#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse causale des defaillances d'un essaim de drones simule, sur plusieurs vols :
graphe d'interactions NRI (Kipf et al. 2018), regression logistique, tests de Granger et
propagation entre drones. Les causes candidates sont exogenes (vent, GNSS, manoeuvre du
leader, voisins) ; la validation par intervention est dans causal_validation.py.

    python analysis/causal_analysis.py --log_dir runs/mc --output_dir causal_out
    python analysis/causal_analysis.py --log_dir runs/base runs/gnss --output_dir out
    python analysis/causal_analysis.py --log_dir logs --all_figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import grangercausalitytests

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


OUTCOMES = ("formation_loss", "nav_degradation", "near_miss")
FACTORS = ("wind", "gnss", "leader_maneuver", "interaction")
# La pression des voisins vient des memes distances que la quasi-collision et l'ecart en
# formation, donc exclue pour ces deux-la. Pour la navigation elle sert de controle negatif.
EXCLUDED_FACTORS = {"near_miss": {"interaction"}, "formation_loss": {"interaction"}}


# Utilitaires


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_torch_threads(n: int) -> None:
    """Peu de threads : plus rapide sur de petits tenseurs."""
    n = int(max(1, n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def persist(mask: np.ndarray, n_steps: int) -> np.ndarray:
    """Garde un 1 seulement apres n_steps valeurs vraies consecutives (par colonne)."""
    if n_steps <= 1:
        return mask.astype(np.int8)
    out = np.zeros_like(mask, dtype=np.int8)
    run = np.zeros(mask.shape[1:], dtype=np.int32)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run >= n_steps
    return out


def onsets(active: np.ndarray) -> np.ndarray:
    """Fronts montants d'un signal binaire [T, N]."""
    a = active.astype(np.int8)
    o = np.zeros_like(a)
    o[1:] = (a[1:] == 1) & (a[:-1] == 0)
    o[0] = a[0] == 1
    return o


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """q : [..., 4] au format [x, y, z, w] -> lacet (rad)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# Chargement des journaux


@dataclass
class SwarmLogs:
    run_id: str
    times: np.ndarray
    drone_names: List[str]
    data: Dict[str, np.ndarray]  # cle -> [T, N, d] ou [T, N]
    dt: float
    notes: List[str] = field(default_factory=list)


_MAIN_LOG = re.compile(r"^drone_\d+\.csv$")


def _find_drone_logs(log_dir: str) -> List[str]:
    return sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir) if _MAIN_LOG.match(f))


def find_runs(log_dir: str) -> List[str]:
    """Un vol (repertoire contenant drone_*.csv) ou un repertoire de vols."""
    if _find_drone_logs(log_dir):
        return [log_dir]
    runs = []
    for d in sorted(os.listdir(log_dir), key=lambda s: (len(s), s)):
        p = os.path.join(log_dir, d)
        if os.path.isdir(p) and _find_drone_logs(p):
            runs.append(p)
    if not runs:
        raise FileNotFoundError(f"Aucun journal drone_*.csv dans {log_dir} ni dans ses sous-repertoires")
    return runs


def _asof(base_t: np.ndarray, df: pd.DataFrame, cols: List[str], direction: str = "nearest") -> np.ndarray:
    left = pd.DataFrame({"time": base_t})
    right = df[["time"] + cols].sort_values("time")
    m = pd.merge_asof(left, right, on="time", direction=direction)
    return m[cols].to_numpy(dtype=np.float64)


def load_run(run_dir: str, downsample: int = 1) -> SwarmLogs:
    """Charge un vol : journaux principaux alignes, completes par ceux du filtre et de verite."""
    paths = _find_drone_logs(run_dir)
    notes: List[str] = []
    dfs, names = [], []
    for p in paths:
        df = pd.read_csv(p)
        if "time" not in df.columns:
            raise ValueError(f"colonne 'time' absente dans {p}")
        df["time"] = df["time"].round(3)
        df = df.drop_duplicates("time").sort_values("time")
        dfs.append(df)
        names.append(os.path.splitext(os.path.basename(p))[0])

    common = dfs[0]["time"].to_numpy()
    for df in dfs[1:]:
        common = np.intersect1d(common, df["time"].to_numpy())
    if len(common) < 20:
        raise ValueError(f"{run_dir} : trop peu d'instants communs ({len(common)})")
    common = common[:: max(1, int(downsample))]  # aligner avant de sous-echantillonner
    dfs = [df.set_index("time").loc[common].reset_index().ffill().bfill() for df in dfs]
    T, N = len(common), len(dfs)
    dt = float(np.median(np.diff(common)))

    def cols(c):
        return np.stack([df[c].to_numpy(np.float64) for df in dfs], axis=1)

    data: Dict[str, np.ndarray] = {
        "pos": np.stack([cols(c) for c in ("gt_x", "gt_y", "gt_z")], axis=-1),
        "vel": np.stack([cols(c) for c in ("gt_vx", "gt_vy", "gt_vz")], axis=-1),
        "wind": np.stack([cols(c) for c in ("wind_x", "wind_y", "wind_z")], axis=-1),
        "contact": cols("collision_flag"),
    }
    data["wind_mag"] = np.linalg.norm(data["wind"], axis=-1)

    nav = np.full((T, N), np.nan)
    gnss = np.full((T, N), np.nan)
    yaw = np.full((T, N), np.nan)
    for i, nm in enumerate(names):
        fpath = os.path.join(run_dir, f"{nm}_filter.csv")
        if os.path.exists(fpath):
            fdf = pd.read_csv(fpath)
            e = _asof(common, fdf, ["e_px", "e_py", "e_pz"])
            nav[:, i] = np.linalg.norm(e, axis=1)
            if "gnss_err" in fdf.columns and fdf["gnss_err"].notna().any():
                upd = fdf[(fdf.get("gnss_update", 1) == 1) & fdf["gnss_err"].notna()]
                gnss[:, i] = _asof(common, upd, ["gnss_err"], direction="backward")[:, 0]
        tpath = os.path.join(run_dir, f"{nm}_truth.csv")
        if os.path.exists(tpath):
            tdf = pd.read_csv(tpath)
            q = _asof(common, tdf, ["qx", "qy", "qz", "qw"])
            yaw[:, i] = yaw_from_quat(q)

    if np.isnan(nav).all():
        nav = cols("ekf_pos_error_mag")
        notes.append("journal du filtre absent : erreur de navigation lue dans le journal principal")
    if np.isnan(gnss).all():
        gnss = cols("gnss_error_mag")
        notes.append(
            "erreur GNSS a l'instant de mesure absente : valeur du journal principal "
            "(SURESTIMEE, mesure maintenue entre deux mises a jour)"
        )
    if np.isnan(yaw).all():
        v = data["vel"]
        yaw = np.arctan2(v[..., 1], v[..., 0])
        notes.append("journal de verite absent : lacet approche par la direction de la vitesse")

    data["nav_err"] = np.nan_to_num(pd.DataFrame(nav).ffill().bfill().to_numpy(), nan=0.0)
    data["gnss_err"] = np.nan_to_num(pd.DataFrame(gnss).ffill().bfill().to_numpy(), nan=0.0)
    data["yaw"] = pd.DataFrame(yaw).ffill().bfill().to_numpy()
    return SwarmLogs(
        run_id=os.path.basename(os.path.normpath(run_dir)),
        times=common,
        drone_names=names,
        data=data,
        dt=dt,
        notes=notes,
    )


def load_runs(log_dir, downsample: int = 1) -> List[SwarmLogs]:
    """Un ou plusieurs repertoires (campagnes) ; identifiant 'campagne/vol' si plusieurs."""
    dirs = [log_dir] if isinstance(log_dir, str) else list(log_dir)
    runs = []
    for d in dirs:
        for r in find_runs(d):
            lg = load_run(r, downsample)
            if len(dirs) > 1:
                lg.run_id = f"{os.path.basename(os.path.normpath(d))}/{lg.run_id}"
            runs.append(lg)
    names = runs[0].drone_names
    runs = [r for r in runs if r.drone_names == names]
    return runs


# Structure connue de l'essaim (pour verifier le graphe NRI)


@dataclass
class SwarmStructure:
    leader: Optional[int]
    followers: List[int]
    independents: List[int]
    source: str

    def known_adjacency(self, n: int) -> np.ndarray:
        """A[i, j] (recepteur i, emetteur j) : 1 influence attendue, 0 aucune, NaN entre suiveurs."""
        A = np.zeros((n, n))
        np.fill_diagonal(A, np.nan)
        if self.leader is not None:
            for f in self.followers:
                A[f, self.leader] = 1.0
        for a in self.followers:
            for b in self.followers:
                if a != b:
                    A[a, b] = np.nan
        return A


def swarm_structure(config_path: Optional[str], names: List[str], leader_index: int = 0) -> SwarmStructure:
    if config_path and os.path.exists(config_path) and yaml is not None:
        cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
        swarms = cfg.get("swarm") or []
        agents = {a.get("name"): a for a in cfg.get("agents", [])}
        if swarms:
            sw = swarms[0]
            members = [nm for nm in names if (agents.get(nm) or {}).get("swarm_id") == sw.get("id")]
            leader = names.index(sw["leader"]) if sw.get("leader") in names else None
            followers = [names.index(m) for m in members if names.index(m) != leader]
            indep = [i for i in range(len(names)) if i != leader and i not in followers]
            return SwarmStructure(leader, followers, indep, f"config ({os.path.basename(config_path)})")
    others = [i for i in range(len(names)) if i != leader_index]
    return SwarmStructure(leader_index, others, [], "hypothese : leader_index, tous les autres suiveurs")


# Defaillances (variables a expliquer)


@dataclass
class Outcomes:
    active: Dict[str, np.ndarray]  # [T, N] binaire, apres stabilisation
    continuous: Dict[str, np.ndarray]
    applicable: Dict[str, np.ndarray]  # [N] defaillance definie pour ce drone


def formation_error(logs: SwarmLogs, st: SwarmStructure, settle_s: float) -> np.ndarray:
    """Ecart de chaque suiveur a sa place (mediane apres stabilisation), dans le repere du leader."""
    T, N = logs.data["pos"].shape[:2]
    err = np.zeros((T, N))
    if st.leader is None or not st.followers:
        return err
    L = st.leader
    p = logs.data["pos"]
    psi = logs.data["yaw"][:, L]
    c, s = np.cos(psi), np.sin(psi)
    settled = logs.times >= settle_s
    if settled.sum() < 5:
        settled = np.ones(T, bool)
    for f in st.followers:
        d = p[:, f, :2] - p[:, L, :2]
        body = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
        slot = np.median(body[settled], axis=0)
        world = np.stack([c * slot[0] - s * slot[1], s * slot[0] + c * slot[1]], axis=1)
        e_xy = d - world
        e_z = p[:, f, 2] - p[:, L, 2]
        err[:, f] = np.sqrt((e_xy**2).sum(axis=1) + e_z**2)
    return err


def compute_outcomes(logs: SwarmLogs, st: SwarmStructure, a: argparse.Namespace) -> Outcomes:
    T, N = logs.data["pos"].shape[:2]

    def k(sec):
        return max(1, int(round(sec / logs.dt)))

    settled = (logs.times >= a.settle_time)[:, None]

    fe = formation_error(logs, st, a.settle_time)
    p = logs.data["pos"]
    d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)
    idx = np.arange(N)
    d[:, idx, idx] = np.inf
    min_d = d.min(axis=-1)
    airborne = p[..., 2] > 0.3

    active = {
        "formation_loss": persist(fe > a.formation_thresh, k(a.formation_persist)) * settled,
        "nav_degradation": persist(logs.data["nav_err"] > a.nav_thresh, k(a.nav_persist)) * settled,
        "near_miss": ((min_d < a.near_miss_dist) & airborne).astype(np.int8) * settled,
    }
    is_f = np.zeros(N, bool)
    is_f[st.followers] = True
    applicable = {"formation_loss": is_f, "nav_degradation": np.ones(N, bool), "near_miss": np.ones(N, bool)}
    for kk in active:
        active[kk] = (active[kk] * applicable[kk][None, :]).astype(np.int8)
    continuous = {
        "formation_loss": fe,
        "nav_degradation": logs.data["nav_err"],
        "near_miss": -np.where(np.isfinite(min_d), min_d, 1e3),
    }
    return Outcomes(active=active, continuous=continuous, applicable=applicable)


# Causes candidates (facteurs exogenes)


def leader_maneuver(logs: SwarmLogs, st: SwarmStructure, smooth_s: float = 0.25) -> np.ndarray:
    """Acceleration horizontale du leader, attribuee a ses suiveurs (0 ailleurs)."""
    T, N = logs.data["pos"].shape[:2]
    out = np.zeros((T, N))
    if st.leader is None:
        return out
    v = logs.data["vel"][:, st.leader, :2]
    acc = np.zeros_like(v)
    acc[1:] = np.diff(v, axis=0) / max(logs.dt, 1e-6)
    w = max(1, int(round(smooth_s / logs.dt)))
    a_mag = pd.Series(np.linalg.norm(acc, axis=1)).rolling(w, min_periods=1).mean().to_numpy()
    for f in st.followers:
        out[:, f] = a_mag
    return out


def interaction_pressure(logs: SwarmLogs, edge_probs: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Somme des voisins j ponderee par l'arete NRI j->i, divisee par la distance."""
    p = logs.data["pos"]
    d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)
    N = d.shape[1]
    d[:, np.arange(N), np.arange(N)] = np.inf
    return np.sum(edge_probs[None, :, :] / (d + eps), axis=-1)


def compute_factors(logs: SwarmLogs, st: SwarmStructure, edge_probs: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "wind": logs.data["wind_mag"],
        "gnss": logs.data["gnss_err"],
        "leader_maneuver": leader_maneuver(logs, st),
        "interaction": interaction_pressure(logs, edge_probs),
    }


# NRI (Kipf et al. 2018)


class MLP(nn.Module):
    def __init__(self, i, h, o, dropout=0.0):
        super().__init__()
        self.fc1, self.fc2, self.bn = nn.Linear(i, h), nn.Linear(h, o), nn.LayerNorm(o)
        self.dropout = dropout

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.dropout, self.training)
        return self.bn(F.elu(self.fc2(x)))


def edge_index(n: int):
    off = np.ones((n, n)) - np.eye(n)
    recv, send = np.where(off)
    return torch.tensor(send, dtype=torch.long), torch.tensor(recv, dtype=torch.long)


class NRIEncoder(nn.Module):
    """Encodeur MLP de NRI : noeud -> arete -> noeud -> arete, sur toute la fenetre."""

    def __init__(self, n_in, hidden, k, dropout=0.0):
        super().__init__()
        self.mlp1 = MLP(n_in, hidden, hidden, dropout)
        self.mlp2 = MLP(2 * hidden, hidden, hidden, dropout)
        self.mlp3 = MLP(hidden, hidden, hidden, dropout)
        self.mlp4 = MLP(3 * hidden, hidden, hidden, dropout)
        self.out = nn.Linear(hidden, k)

    def forward(self, x, send, recv, n):
        # x : [B, N, L*S]
        h = self.mlp1(x)
        e = self.mlp2(torch.cat([h[:, send], h[:, recv]], -1))
        skip = e
        agg = torch.zeros(x.shape[0], n, e.shape[-1], dtype=e.dtype)
        agg.index_add_(1, recv, e)
        h2 = self.mlp3(agg / max(n - 1, 1))
        e2 = self.mlp4(torch.cat([h2[:, send], h2[:, recv], skip], -1))
        return self.out(e2)  # [B, E, K]


class NRIDecoder(nn.Module):
    """
    Decodeur a passage de messages qui predit la vitesse au pas suivant. Le type 0 n'envoie
    aucun message (skip_first) et un message j -> i ne voit que des grandeurs relatives.
    Le cap et les vitesses passees du noeud evitent que les suiveurs servent a predire le leader.
    """

    def __init__(self, u_dim, hidden, k, dropout=0.0, hist=2):
        super().__init__()
        self.k = k
        self.msg = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
                for _ in range(k)
            ]
        )
        self.hist = hist
        self.node = nn.Sequential(
            nn.Linear(6 + 3 * hist + u_dim + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.hidden = hidden

    def forward(self, x, u, z, send, recv, perm_edge=None, perm_idx=None, vh=None):
        """x : [B, N, 8] (position, vitesse, cos/sin du cap), vh : [B, N, 3*hist] vitesses passees."""
        B, N, _ = x.shape
        p, v, head = x[..., :3], x[..., 3:6], x[..., 6:8]
        rel = torch.cat([v[:, send] - v[:, recv], p[:, send] - p[:, recv]], -1)
        if perm_edge is not None:
            rel = rel.clone()
            rel[:, perm_edge] = rel[perm_idx, perm_edge]
        pair = torch.cat([v[:, recv], rel], -1)
        agg = torch.zeros(B, N, self.hidden, dtype=x.dtype)
        for t in range(1, self.k):  # type 0 : pas de message
            agg.index_add_(1, recv, z[..., t : t + 1] * self.msg[t](pair))
        return v + self.node(torch.cat([v, p[..., 2:3], head, vh, u, agg], -1))


@dataclass
class NRIResults:
    edge_probs: np.ndarray  # [N, N] 1 - p(type 0), moyenne sur les graines
    edge_probs_std: np.ndarray
    importance: np.ndarray  # [N, N] hausse relative de l'erreur de i quand j est permute
    importance_std: np.ndarray
    loss_curves: List[List[float]]
    heldout: Dict[str, float]  # MSE sur vols non vus, avec / sans aretes
    validation: Dict[str, object]


def nri_dataset(runs: List[SwarmLogs], nri_dt: float, settle_s: float = 0.0):
    """Par vol : x [T, N, 8] (position, vitesse, cos/sin cap) et u [T, N, 4] (vent, GNSS), sans decollage."""
    out = []
    for r in runs:
        step = max(1, int(round(nri_dt / r.dt)))
        keep = r.times >= settle_s
        p = r.data["pos"][keep][::step]
        v = r.data["vel"][keep][::step]
        psi = r.data["yaw"][keep][::step]
        x = np.concatenate(
            [p - p.reshape(-1, 3).mean(0), v, np.cos(psi)[..., None], np.sin(psi)[..., None]], axis=-1
        )
        u = np.concatenate(
            [r.data["wind"][keep][::step], r.data["gnss_err"][keep][::step][..., None]], axis=-1
        )
        out.append((x.astype(np.float32), u.astype(np.float32)))
    return out


def _windows(series, L, n, rng):
    """Fenetres de L+1 pas, jamais a cheval sur deux vols."""
    lens = np.array([s.shape[0] - L - 1 for s, _ in series])
    ok = np.where(lens > 0)[0]
    if len(ok) == 0:
        raise ValueError("vols trop courts pour la fenetre NRI")
    w = lens[ok] / lens[ok].sum()
    S, U = [], []
    for r in rng.choice(ok, size=n, p=w):
        t0 = rng.randint(0, lens[r])
        S.append(series[r][0][t0 : t0 + L + 1])
        U.append(series[r][1][t0 : t0 + L + 1])
    return np.stack(S), np.stack(U)


def _gumbel(logits, tau, hard=False):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + g) / tau, dim=-1)
    if not hard:
        return y
    # Straight-through : aretes discretes en avant, gradient de la version douce
    y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(-1, keepdim=True), 1.0)
    return (y_hard - y).detach() + y


def train_nri_once(train, test, n, a, seed):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    L, K = a.seq_len, a.n_edge_types
    S_dim, U_dim = train[0][0].shape[-1], train[0][1].shape[-1]
    x_all = np.concatenate([x.reshape(-1, S_dim) for x, _ in train])
    u_all = np.concatenate([u.reshape(-1, U_dim) for _, u in train])
    # Echelles isotropes pour garder la geometrie relative entre drones
    sp, sv = float(x_all[:, :3].std()) or 1.0, float(x_all[:, 3:6].std()) or 1.0
    scale = np.array([sp] * 3 + [sv] * 3 + [1.0] * (S_dim - 6), dtype=np.float32)
    su = StandardScaler().fit(u_all)

    def norm(d):
        return [
            (
                (x / scale).astype(np.float32),
                su.transform(u.reshape(-1, U_dim)).reshape(u.shape).astype(np.float32),
            )
            for x, u in d
        ]

    train_n, test_n = norm(train), norm(test)

    send, recv = edge_index(n)
    enc = NRIEncoder(L * S_dim, a.hidden, K, a.dropout)
    H = a.own_history
    if L < H + 1:
        raise ValueError("seq_len doit depasser own_history")
    dec = NRIDecoder(U_dim, a.hidden, K, a.dropout, hist=H)

    def hist(w):
        """Vitesses propres aux H pas precedant le dernier pas de la fenetre."""
        return torch.cat([w[:, L - 2 - h, :, 3:6] for h in range(H)], -1)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)

    prior = np.full(K, (1.0 - a.edge_prior) / max(K - 1, 1))
    prior[0] = a.edge_prior
    log_prior = torch.tensor(np.log(prior), dtype=torch.float32)

    Sw, Uw = _windows(train_n, L, a.max_train_windows, rng)
    Sw, Uw = torch.from_numpy(Sw), torch.from_numpy(Uw)
    curve = []
    for step in range(1, a.nri_steps + 1):
        tau = a.tau_start + (a.tau_end - a.tau_start) * min(1.0, step / a.tau_anneal_steps)
        b = torch.from_numpy(rng.randint(0, Sw.shape[0], a.batch_size))
        s, u = Sw[b], Uw[b]
        ctx = s[:, :L].permute(0, 2, 1, 3).reshape(s.shape[0], n, -1)
        logits = enc(ctx, send, recv, n)
        z = _gumbel(logits, tau, hard=a.hard_edges)
        pred = dec(s[:, L - 1], u[:, L - 1], z, send, recv, vh=hist(s))
        nll = F.mse_loss(pred, s[:, L, :, 3:6])  # vitesse au pas suivant
        q = F.softmax(logits, -1)
        kl = (q * (torch.log(q + 1e-12) - log_prior)).sum(-1).mean()
        loss = nll + a.kl_weight * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 5.0)
        opt.step()
        curve.append(float(nll.item()))
        if a.verbose_every and step % a.verbose_every == 0:
            print(f"[NRI graine {seed}] pas {step}/{a.nri_steps}  mse={nll.item():.5f}  kl={kl.item():.4f}")

    enc.eval()
    dec.eval()
    with torch.no_grad():
        # Aretes moyennes sur des fenetres d'apprentissage
        b = torch.from_numpy(rng.randint(0, Sw.shape[0], min(1024, Sw.shape[0])))
        ctx = Sw[b][:, :L].permute(0, 2, 1, 3).reshape(len(b), n, -1)
        q = F.softmax(enc(ctx, send, recv, n), -1).mean(0).numpy()
        # Vols non vus : graphe infere, puis aucune arete
        mse_g, mse_0 = float("nan"), float("nan")
        D = np.full((n, n), np.nan)
        if test_n:
            St, Ut = _windows(test_n, L, 1024, np.random.RandomState(seed + 1))
            St, Ut = torch.from_numpy(St), torch.from_numpy(Ut)
            ctx = St[:, :L].permute(0, 2, 1, 3).reshape(St.shape[0], n, -1)
            zq = F.softmax(enc(ctx, send, recv, n), -1)
            if a.hard_edges:  # meme regime qu'a l'apprentissage
                zq = torch.zeros_like(zq).scatter_(-1, zq.argmax(-1, keepdim=True), 1.0)
            s_t, u_t, tgt, vh = St[:, L - 1], Ut[:, L - 1], St[:, L, :, 3:6], hist(St)
            err_g = ((dec(s_t, u_t, zq, send, recv, vh=vh) - tgt) ** 2).mean(dim=(0, 2))
            mse_g = float(err_g.mean())
            z0 = torch.zeros_like(zq)
            z0[..., 0] = 1.0
            mse_0 = float(F.mse_loss(dec(s_t, u_t, z0, send, recv, vh=vh), tgt))
            # Importance de j -> i : grandeurs relatives de j prises dans une autre fenetre
            g = torch.Generator().manual_seed(seed + 7)
            reps = 5
            for e, (j, i) in enumerate(zip(send.numpy(), recv.numpy())):
                inc = 0.0
                for _ in range(reps):
                    perm = torch.randperm(s_t.shape[0], generator=g)
                    err_p = (
                        (dec(s_t, u_t, zq, send, recv, perm_edge=e, perm_idx=perm, vh=vh) - tgt) ** 2
                    ).mean(dim=(0, 2))
                    inc += float(err_p[i] - err_g[i]) / max(float(err_g[i]), 1e-12)
                D[i, j] = inc / reps
    P = np.full((n, n), np.nan)
    for e, (j, i) in enumerate(zip(send.numpy(), recv.numpy())):
        P[i, j] = 1.0 - q[e, 0]
    return P, curve, mse_g, mse_0, D


def validate_graph(P: np.ndarray, A: np.ndarray, names: List[str]) -> Dict[str, object]:
    mask = np.isfinite(A) & np.isfinite(P)
    y, s = A[mask], P[mask]
    res: Dict[str, object] = {"n_aretes_jugees": int(mask.sum()), "n_aretes_attendues": int((y == 1).sum())}
    if 0 < y.sum() < len(y):
        res["auroc"] = float(roc_auc_score(y, s))
        res["min_attendues"] = float(s[y == 1].min())
        res["max_absentes"] = float(s[y == 0].max())
        res["separation_parfaite"] = bool(s[y == 1].min() > s[y == 0].max())
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append(
                    {
                        "emetteur": names[j],
                        "recepteur": names[i],
                        "p_arete": float(P[i, j]),
                        "attendu": (None if not np.isfinite(A[i, j]) else int(A[i, j])),
                    }
                )
    res["aretes"] = sorted(pairs, key=lambda d: -d["p_arete"])
    return res


def run_nri(runs: List[SwarmLogs], st: SwarmStructure, a: argparse.Namespace) -> NRIResults:
    set_torch_threads(a.torch_threads)
    n = len(runs[0].drone_names)
    data = nri_dataset(runs, a.nri_dt, a.settle_time)
    n_test = max(1, len(data) // 5) if len(data) >= 2 else 0
    test, train = data[:n_test], data[n_test:]
    Ps, Ds, curves, hg, h0 = [], [], [], [], []
    for k in range(a.nri_seeds):
        P, c, mg, m0, D = train_nri_once(train, test, n, a, a.seed + k)
        Ps.append(P)
        Ds.append(D)
        curves.append(c)
        hg.append(mg)
        h0.append(m0)
    Ps, Ds = np.stack(Ps), np.stack(Ds)
    with warnings.catch_warnings():  # diagonale NaN : pas d'auto-arete
        warnings.simplefilter("ignore", RuntimeWarning)
        P, Pstd = np.nanmean(Ps, 0), np.nanstd(Ps, 0)
        D, Dstd = np.nanmean(Ds, 0), np.nanstd(Ds, 0)
    heldout = {
        "n_vols_test": n_test,
        "mse_avec_graphe": float(np.nanmean(hg)),
        "mse_sans_aretes": float(np.nanmean(h0)),
    }
    if np.isfinite(heldout["mse_sans_aretes"]) and heldout["mse_sans_aretes"] > 0:
        heldout["gain_relatif"] = 1.0 - heldout["mse_avec_graphe"] / heldout["mse_sans_aretes"]
    A = st.known_adjacency(n)
    val = validate_graph(P, A, runs[0].drone_names)
    val["structure"] = st.source
    val["stabilite_ecart_type_moyen"] = float(np.nanmean(Pstd))
    if np.isfinite(D).any():
        vd = validate_graph(D, A, runs[0].drone_names)
        val["importance_permutation"] = {k: v for k, v in vd.items() if k != "aretes"}
    return NRIResults(P, Pstd, D, Dstd, curves, heldout, val)


# Influence signee (heuristique)


def signed_influence(runs: List[SwarmLogs], P: np.ndarray, thresh: float) -> np.ndarray:
    """Signe de j -> i (+1 attraction, -1 repulsion, 0 si |t| < 2 ou arete faible), hors mode commun."""
    n = P.shape[0]
    num = np.zeros((n, n))
    cnt = np.zeros((n, n))
    sq = np.zeros((n, n))
    for r in runs:
        p, v = r.data["pos"], r.data["vel"]
        acc = np.zeros_like(v)
        acc[1:] = np.diff(v, axis=0) / max(r.dt, 1e-6)
        acc = acc - acc.mean(axis=1, keepdims=True)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rel = p[:, j] - p[:, i]
                dist = np.linalg.norm(rel, axis=1) + 1e-6
                x = np.sum(acc[:, i] * rel, axis=1) / dist
                num[i, j] += x.sum()
                sq[i, j] += (x**2).sum()
                cnt[i, j] += len(x)
    mean = num / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(sq / np.maximum(cnt, 1) - mean**2, 1e-12))
    tstat = mean / (sd / np.sqrt(np.maximum(cnt, 1)))
    sign = np.where(np.abs(tstat) >= 2.0, np.sign(tstat), 0.0)
    out = np.where(P >= thresh, sign * P, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


# Modeles d'evenements (regression logistique) et attribution des causes


def _past_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Moyenne de x sur [t-w, t-1] (strictement passe), [T, N]."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[w:] = (c[w:-1] - c[: -w - 1]) / w
    return out


def build_event_table(runs, outs, facs, st, a):
    """Une ligne par (vol, instant, drone) sans defaillance sur l'horizon passe ; causes moyennees dessus."""
    rows = {ev: [] for ev in OUTCOMES}
    for ri, (r, o, fx) in enumerate(zip(runs, outs, facs)):
        w = max(1, int(round(a.horizon / r.dt)))
        X = np.stack([_past_mean(fx[f], w) for f in FACTORS], axis=-1)  # [T, N, F]
        for ev in OUTCOMES:
            act = o.active[ev]
            prev = _past_mean(act.astype(float), w)
            ok = np.isfinite(X).all(-1) & np.isfinite(prev) & (prev == 0)
            ok &= o.applicable[ev][None, :]
            ok &= (r.times >= a.settle_time + a.horizon)[:, None]
            t_idx, i_idx = np.where(ok)
            rows[ev].append(
                pd.DataFrame(
                    {
                        "run": ri,
                        "t": t_idx,
                        "time": r.times[t_idx],
                        "drone": i_idx,
                        **{f: X[t_idx, i_idx, k] for k, f in enumerate(FACTORS)},
                        "y": act[t_idx, i_idx].astype(int),
                    }
                )
            )
    return {ev: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for ev, v in rows.items()}


@dataclass
class EventResults:
    metrics: Dict[str, Dict]
    attribution: Dict[str, Dict[str, List[float]]]  # defaillance -> cause -> [N] parts moyennes
    instances: List[Dict]


def fit_event_models(tables, names, a) -> EventResults:
    metrics, attribution, instances = {}, {}, []
    rng = np.random.RandomState(a.seed)
    n = len(names)
    for ev, df in tables.items():
        feats = [f for f in FACTORS if f not in EXCLUDED_FACTORS.get(ev, set())]
        m: Dict[str, object] = {
            "causes_candidates": feats,
            "n_echantillons": int(len(df)),
            "n_apparitions": int(df["y"].sum()) if len(df) else 0,
        }
        metrics[ev] = m
        if len(df) == 0 or df["y"].sum() < a.min_events or df["y"].sum() == len(df):
            m["statut"] = "trop peu d'apparitions pour un modele"
            continue
        X = df[feats].to_numpy(float)
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        y = df["y"].to_numpy(int)
        groups = df["run"].to_numpy()

        # Validation croisee par vol
        uniq = np.unique(groups)
        pred = np.full(len(y), np.nan)
        if len(uniq) >= 2:
            folds = np.array_split(rng.permutation(uniq), min(5, len(uniq)))
            for fo in folds:
                te = np.isin(groups, fo)
                if y[~te].sum() == 0 or y[~te].sum() == (~te).sum():
                    continue
                clf = LogisticRegression(max_iter=2000).fit(Xs[~te], y[~te])
                pred[te] = clf.predict_proba(Xs[te])[:, 1]
        ok = np.isfinite(pred)
        if ok.sum() and 0 < y[ok].sum() < ok.sum():
            m["auroc_vols_non_vus"] = float(roc_auc_score(y[ok], pred[ok]))
            m["brier_vols_non_vus"] = float(brier_score_loss(y[ok], pred[ok]))
            m["brier_reference_taux_constant"] = float(
                brier_score_loss(y[ok], np.full(ok.sum(), y[ok].mean()))
            )

        # Modele final, IC par bootstrap sur les vols
        clf = LogisticRegression(max_iter=2000).fit(Xs, y)
        coef = clf.coef_[0]
        boots = []
        for _ in range(a.n_boot if len(uniq) >= 2 else 0):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            ix = np.concatenate([np.where(groups == g)[0] for g in pick])
            if 0 < y[ix].sum() < len(ix):
                boots.append(LogisticRegression(max_iter=2000).fit(Xs[ix], y[ix]).coef_[0])
        boots = np.array(boots) if boots else np.full((1, len(feats)), np.nan)
        lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
        m["coefficients"] = {
            f: {
                "odds_ratio_par_ecart_type": float(np.exp(coef[k])),
                "ic95": [float(np.exp(lo[k])), float(np.exp(hi[k]))],
                "significatif": bool(np.isfinite(lo[k]) and (lo[k] > 0 or hi[k] < 0)),
            }
            for k, f in enumerate(feats)
        }

        # Parts des contributions positives
        contrib = np.maximum(Xs * coef[None, :], 0.0)
        tot = contrib.sum(axis=1)
        pos = np.where(y == 1)[0]
        acc = {f: np.zeros(n) for f in feats}
        acc["inexpliquee"] = np.zeros(n)
        cnt = np.zeros(n)
        for r in pos:
            i = int(df["drone"].iat[r])
            cnt[i] += 1
            if tot[r] < a.attr_min:
                acc["inexpliquee"][i] += 1.0
                share = {f: 0.0 for f in feats}
                share["inexpliquee"] = 1.0
            else:
                share = {f: float(contrib[r, k] / tot[r]) for k, f in enumerate(feats)}
                share["inexpliquee"] = 0.0
                for f in feats:
                    acc[f][i] += share[f]
            if len(instances) < a.max_events_to_explain:
                instances.append(
                    {
                        "defaillance": ev,
                        "vol": int(df["run"].iat[r]),
                        "temps": float(df["time"].iat[r]),
                        "drone": names[i],
                        "parts_des_causes": share,
                    }
                )
        attribution[ev] = {f: (acc[f] / np.maximum(cnt, 1)).tolist() for f in acc}
        attribution[ev]["_n"] = cnt.tolist()
        m["statut"] = "ok"
        print(
            f"[Evenements] {ev}: {int(y.sum())} apparitions, AUROC vols non vus = "
            f"{m.get('auroc_vols_non_vus', float('nan')):.3f}"
        )
    return EventResults(metrics, attribution, instances)


# Granger


def _granger_tests(z: np.ndarray, maxlag: int):
    try:
        return grangercausalitytests(z, maxlag=maxlag, verbose=False)
    except TypeError:
        return grangercausalitytests(z, maxlag=maxlag)


def granger(runs, outs, facs, st, a) -> pd.DataFrame:
    """Test de Granger cause -> grandeur continue, par vol et par drone, seuil de Bonferroni global."""
    recs = []
    n_tests = 0
    for r, o, fx in zip(runs, outs, facs):
        k = max(1, int(round(a.granger_dt / r.dt)))
        t_ok = r.times >= a.settle_time
        for ev in OUTCOMES:
            for f in FACTORS:
                if f in EXCLUDED_FACTORS.get(ev, set()):
                    continue
                for i in range(len(r.drone_names)):
                    if not o.applicable[ev][i]:
                        continue
                    y = o.continuous[ev][t_ok, i][::k]
                    x = fx[f][t_ok, i][::k]
                    rec = {"vol": r.run_id, "defaillance": ev, "cause": f, "drone": r.drone_names[i]}
                    n_tests += 1
                    if len(y) < 3 * a.granger_maxlag + 10 or np.std(y) < 1e-9 or np.std(x) < 1e-9:
                        rec["statut"] = "degenere"
                        recs.append(rec)
                        continue
                    try:
                        z = np.column_stack([(y - y.mean()) / y.std(), (x - x.mean()) / x.std()])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = _granger_tests(z, a.granger_maxlag)
                        best = min(res, key=lambda L: res[L][0]["ssr_ftest"][1])
                        Fv, p, dfd, dfn = res[best][0]["ssr_ftest"]
                        rec.update(
                            statut="ok",
                            p_min=float(p),
                            retard=int(best),
                            p_corrige_retards=float(min(1.0, p * a.granger_maxlag)),
                            r2_partiel=float(Fv * dfn / (Fv * dfn + dfd)),
                        )
                    except Exception as exc:
                        rec.update(statut=f"echec: {type(exc).__name__}")
                    recs.append(rec)
    df = pd.DataFrame(recs)
    if len(df) and "p_corrige_retards" in df:
        alpha = 0.05 / max(n_tests, 1)
        df["significatif_bonferroni"] = df["p_corrige_retards"] < alpha
        df.attrs["alpha_bonferroni"] = alpha
    return df


# Impact systemique (contre une hypothese nulle)


def systemic_impact(runs, outs, a, rng):
    """impact[i, j] = P(defaillance de j dans ]t, t+H] | apparition chez i en t) - P avec t decale au hasard."""
    n = len(runs[0].drone_names)
    any_on = []
    for o in outs:
        act = np.zeros_like(o.active[OUTCOMES[0]])
        for ev in OUTCOMES:
            act |= o.active[ev]
        any_on.append(onsets(act))
    result = {}
    for ev in OUTCOMES:
        cond = np.zeros((n, n))
        null = np.zeros((a.n_perm, n, n))
        cnt = np.zeros(n)
        for r, o, tgt in zip(runs, outs, any_on):
            T = len(r.times)
            H = max(1, int(round(a.impact_horizon / r.dt)))
            src = onsets(o.active[ev])
            csum = np.vstack([np.zeros((1, n)), np.cumsum(tgt, axis=0)])

            def hit(tt, csum=csum, H=H, T=T):
                return (csum[np.minimum(tt + H + 1, T)] - csum[np.minimum(tt + 1, T)]) > 0

            for i in range(n):
                tt = np.where(src[:, i])[0]
                if len(tt) == 0:
                    continue
                cnt[i] += len(tt)
                cond[i] += hit(tt).sum(axis=0)
                for b in range(a.n_perm):
                    sh = (tt + rng.randint(H, max(H + 1, T - H))) % T
                    null[b, i] += hit(sh).sum(axis=0)
        c = cond / np.maximum(cnt, 1)[:, None]
        nl = null / np.maximum(cnt, 1)[None, :, None]
        imp = c - nl.mean(0)
        pval = (1 + (nl >= c[None]).sum(0)) / (1 + a.n_perm)
        np.fill_diagonal(imp, np.nan)
        np.fill_diagonal(pval, np.nan)
        imp[cnt == 0] = np.nan
        result[ev] = {"impact": imp, "p": pval, "n_sources": cnt}
    return result


# Figures


def heatmap(mat, title, xl, yl, xt, yt, path, vmin=None, vmax=None, annot=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.4 * len(xt) + 3, 1.0 * len(yt) + 2.4))
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels(xt, rotation=30, ha="right")
    ax.set_yticks(range(len(yt)))
    ax.set_yticklabels(yt)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontsize=10)
    if annot is not None:
        for (r, c), s in np.ndenumerate(annot):
            if s:
                ax.text(c, r, s, ha="center", va="center", fontsize=8, color="w")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_nri(res: NRIResults, A: np.ndarray, names, out, imp_thresh, details=None):
    """Graphe principal dans `out` ; cartes de chaleur et apprentissage dans `details` (si fourni)."""
    n = len(names)

    def ann(M, Ms, fmt):
        an = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if i == j:
                    an[i, j] = ""
                    continue
                exp = "" if not np.isfinite(A[i, j]) else (" ✓" if A[i, j] == 1 else " ·")
                an[i, j] = fmt.format(M[i, j], Ms[i, j]) + exp
        return an

    v = res.validation
    auc = v.get("auroc")
    if details:
        _plot_nri_details(res, names, details, ann, v, auc)
    _plot_graph(res, A, names, out, imp_thresh)


def _plot_nri_details(res, names, out, ann, v, auc):
    heatmap(
        res.edge_probs,
        "NRI : probabilite d'arete j -> i (moyenne ± ecart-type entre graines)\n"
        "✓ influence attendue, · aucune attendue"
        + (f" | AUROC vs structure connue = {auc:.2f}" if auc is not None else ""),
        "emetteur j",
        "recepteur i",
        names,
        names,
        os.path.join(out, "nri_edge_probs_heatmap.png"),
        0,
        1,
        ann(res.edge_probs, res.edge_probs_std, "{:.2f}±{:.2f}"),
    )
    if np.isfinite(res.importance).any():
        vi = v.get("importance_permutation", {})
        auc_i = vi.get("auroc")
        heatmap(
            100 * res.importance,
            "NRI : hausse de l'erreur de prediction de i sur des vols NON vus\n"
            "quand l'etat de j est remplace par celui d'une autre fenetre [%]"
            + (f" | AUROC vs structure connue = {auc_i:.2f}" if auc_i is not None else ""),
            "emetteur j",
            "recepteur i",
            names,
            names,
            os.path.join(out, "nri_edge_importance_heatmap.png"),
            0,
            max(1.0, float(np.nanmax(100 * res.importance))),
            ann(100 * res.importance, 100 * res.importance_std, "{:.0f}±{:.0f}%"),
            cmap="magma",
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in enumerate(res.loss_curves):
        ax.plot(pd.Series(c).rolling(50, min_periods=1).mean(), lw=1, label=f"graine {k}")
    ax.set_xlabel("pas d'apprentissage")
    ax.set_ylabel("MSE (moyenne glissante)")
    ax.set_title("Apprentissage NRI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nri_training_loss.png"), dpi=160)
    plt.close(fig)


def _swarm_layout(A, names):
    n = len(names)
    A0 = np.nan_to_num(A, nan=0.0)
    leaders = [j for j in range(n) if A0[:, j].sum() > 0]
    if len(leaders) != 1:
        return None
    L = leaders[0]
    fol = [i for i in range(n) if A0[i, L] == 1]
    ind = [i for i in range(n) if i != L and i not in fol]
    pos = {names[L]: np.array([0.0, 1.0])}
    xs = np.linspace(-1.0, 1.0, len(fol)) if len(fol) > 1 else [0.0]
    for x, i in zip(xs, fol):
        pos[names[i]] = np.array([x, -0.4])
    for k, i in enumerate(ind):
        pos[names[i]] = np.array([2.1, 0.3 - 0.8 * k])
    return pos


def _plot_graph(res, A, names, out, imp_thresh):
    n = len(names)
    W = res.importance if np.isfinite(res.importance).any() else res.edge_probs
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(W[i, j]) and W[i, j] >= imp_thresh:
                G.add_edge(names[j], names[i], w=float(W[i, j]))
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = _swarm_layout(A, names) or nx.circular_layout(G)
    A0 = np.nan_to_num(A, nan=0.0)
    role = {}
    for k, nm in enumerate(names):
        role[nm] = "leader" if A0[:, k].sum() > 0 else "follower" if A0[k].sum() > 0 else "independent"
    nx.draw_networkx_nodes(G, pos, node_size=1900, node_color="#cfe2f3", ax=ax)
    nx.draw_networkx_labels(G, pos, {nm: f"{nm}\n({role[nm]})" for nm in names}, ax=ax, font_size=8)
    if G.number_of_edges():
        wmax = max(G[u][v]["w"] for u, v in G.edges())
        idx = {nm: k for k, nm in enumerate(names)}
        col = []
        for u_, v_ in G.edges():  # u_ emetteur, v_ recepteur
            exp = A[idx[v_], idx[u_]]
            col.append("#999999" if not np.isfinite(exp) else ("#2e8b57" if exp == 1 else "#cc3333"))
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=18,
            edge_color=col,
            width=[1 + 5 * G[u][v]["w"] / wmax for u, v in G.edges()],
            connectionstyle="arc3,rad=0.12",
            node_size=1900,
        )
        lab = {(u, v): f"+{100 * G[u][v]['w']:.0f}%" for u, v in G.edges()}
        try:  # connectionstyle accepte a partir de networkx 3.2
            nx.draw_networkx_edge_labels(G, pos, lab, font_size=8, ax=ax, connectionstyle="arc3,rad=0.12")
        except TypeError:
            nx.draw_networkx_edge_labels(G, pos, lab, font_size=8, ax=ax)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color="#2e8b57", lw=3, label="real link (leader -> follower)"),
            Line2D([], [], color="#cc3333", lw=3, label="false link (does not exist)"),
            Line2D([], [], color="#999999", lw=3, label="follower <-> follower (avoidance)"),
        ],
        loc="upper center",
        fontsize=8,
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
    )
    ax.margins(0.15)
    title = (
        "Who influences whom? (graph learned by the NRI)\n"
        f"arrow j -> i: knowing j improves the prediction of i's motion by more than "
        f"{100 * imp_thresh:.0f} %"
    )
    h = res.heldout or {}
    if (
        h.get("mse_avec_graphe") is not None
        and h.get("mse_sans_aretes") is not None
        and h["mse_avec_graphe"] >= h["mse_sans_aretes"]
    ):
        title += "\nNOT RELIABLE: on unseen flights the graph predicts no better than a model without links"
        ax.set_facecolor("#fff3f3")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "graphe_interactions.png"), dpi=160)
    plt.close(fig)


CAUSE_LABELS = {
    "wind": "wind",
    "gnss": "GNSS error",
    "leader_maneuver": "leader manoeuvre",
    "interaction": "close neighbours",
    "inexpliquee": "unexplained",
}
CAUSE_COLORS = {
    "wind": "#4c9be8",
    "gnss": "#e8833a",
    "leader_maneuver": "#6aa84f",
    "interaction": "#a64d79",
    "inexpliquee": "#cccccc",
}
OUTCOME_LABELS = {
    "formation_loss": "Loss of formation",
    "nav_degradation": "Navigation degradation",
    "near_miss": "Near miss",
}


def plot_causes(ev: "EventResults", names, out, title_suffix=""):
    """Part de chaque cause dans les defaillances de chaque drone, une sous-figure par defaillance."""
    evs = [e for e in OUTCOMES if e in ev.attribution]
    if not evs:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Too few failures to estimate their causes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(os.path.join(out, "causes_defaillances.png"), dpi=130)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(evs), figsize=(5.2 * len(evs), 3.6), squeeze=False)
    for ax, e in zip(axes[0], evs):
        at = ev.attribution[e]
        nn = np.array(at["_n"])
        keep = [i for i in range(len(names)) if nn[i] > 0]
        left = np.zeros(len(keep))
        for c in [c for c in at if not c.startswith("_")]:
            vals = np.array([at[c][i] for i in keep]) * 100
            if np.all(vals == 0):
                continue
            ax.barh(
                range(len(keep)),
                vals,
                left=left,
                color=CAUSE_COLORS.get(c, "#888"),
                label=CAUSE_LABELS.get(c, c),
            )
            left += vals
        ax.set_yticks(range(len(keep)), [f"{names[i]}\n({int(nn[i])} cases)" for i in keep])
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of cases [%]")
        ax.invert_yaxis()
        ax.set_title(OUTCOME_LABELS.get(e, e), fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False)
    fig.suptitle("Which cause does the analysis attribute to each failure?" + title_suffix, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "causes_defaillances.png"), dpi=150)
    plt.close(fig)


def plot_event_models(ev: EventResults, out):
    rows = [(e, f, c) for e, m in ev.metrics.items() for f, c in (m.get("coefficients") or {}).items()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.5))
    for y, (_e, _f, c) in enumerate(rows):
        lo, hi = c["ic95"]
        col = "#d62728" if c["significatif"] else "0.5"
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.plot(c["odds_ratio_par_ecart_type"], y, "o", color=col)
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{e} <- {f}" for e, f, _ in rows], fontsize=8)
    ax.set_xlabel("rapport de cotes pour +1 ecart-type de la cause (IC 95 %, bootstrap par vol)")
    ax.set_title(
        "Effet de chaque cause candidate sur l'apparition des defaillances\n"
        "rouge : intervalle qui exclut 1 (effet significatif)",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "event_model_odds_ratios.png"), dpi=160)
    plt.close(fig)


# Pipeline


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _plot_details(a, out, names, signed, ev, gr, imp):
    heatmap(
        signed,
        "Influence signee j -> i (+ attraction, - repulsion, 0 non significatif)",
        "emetteur j",
        "recepteur i",
        names,
        names,
        os.path.join(out, "nri_signed_influence_heatmap.png"),
        -1,
        1,
        cmap="coolwarm",
    )
    plot_event_models(ev, out)

    for e, at in ev.attribution.items():
        causes = [c for c in at if not c.startswith("_")]
        mat = np.array([at[c] for c in causes])
        heatmap(
            mat,
            f"Part moyenne de chaque cause dans les apparitions de '{e}'\n"
            f"(nombre d'apparitions par drone : {[int(x) for x in at['_n']]})",
            "drone",
            "cause",
            names,
            causes,
            os.path.join(out, f"avg_root_cause_{e}.png"),
            0,
            1,
        )
    if len(gr) and "significatif_bonferroni" in gr:
        ok = gr[gr.statut == "ok"]
        for e in OUTCOMES:
            sub = ok[ok.defaillance == e]
            if sub.empty:
                continue
            piv = (
                sub.groupby(["cause", "drone"])["significatif_bonferroni"]
                .mean()
                .unstack()
                .reindex(columns=names)
            )
            heatmap(
                piv.to_numpy(float),
                f"Granger : part des vols ou la cause predit '{e}'\n"
                f"(seuil de Bonferroni alpha = {gr.attrs.get('alpha_bonferroni', float('nan')):.1e})",
                "drone",
                "cause",
                names,
                list(piv.index),
                os.path.join(out, f"granger_{e}_heatmap.png"),
                0,
                1,
            )
    for e, d in imp.items():
        star = np.where(np.nan_to_num(d["p"], nan=1) < 0.05, "*", "")
        lim = np.nanmax(np.abs(d["impact"])) if np.isfinite(d["impact"]).any() else 1.0
        heatmap(
            d["impact"],
            f"Impact : '{e}' sur le drone i -> nouvelle defaillance de j sous "
            f"{a.impact_horizon:g} s\n(ecart a l'hypothese nulle, * p < 0.05)",
            "drone affecte j",
            "drone declencheur i",
            names,
            names,
            os.path.join(out, f"impact_{e}_heatmap.png"),
            -lim,
            lim,
            star,
            cmap="coolwarm",
        )


def run_pipeline(a: argparse.Namespace) -> Dict:
    ensure_dir(a.output_dir)
    runs = load_runs(a.log_dir, a.downsample)
    names = runs[0].drone_names
    n = len(names)
    st = swarm_structure(a.config, names, a.leader_index)
    notes = sorted({x for r in runs for x in r.notes})
    print(f"[Chargement] {len(runs)} vol(s), {n} drones, dt={runs[0].dt:.3f} s ; structure : {st.source}")
    for x in notes:
        print(f"  ! {x}")

    outs = [compute_outcomes(r, st, a) for r in runs]

    nri = run_nri(runs, st, a)
    print(
        f"[NRI] AUROC vs structure connue = {nri.validation.get('auroc')} (probabilites), "
        f"{nri.validation.get('importance_permutation', {}).get('auroc')} (importance par permutation), "
        f"MSE vols non vus avec/sans graphe = {nri.heldout['mse_avec_graphe']:.4f} / "
        f"{nri.heldout['mse_sans_aretes']:.4f}"
    )
    # Aretes retenues par importance : l'a priori parcimonieux comprime les probabilites
    W = nri.importance if np.isfinite(nri.importance).any() else nri.edge_probs
    signed = signed_influence(runs, W, a.importance_thresh if W is nri.importance else a.graph_thresh)

    facs = [compute_factors(r, st, np.nan_to_num(nri.edge_probs)) for r in runs]
    tables = build_event_table(runs, outs, facs, st, a)
    ev = fit_event_models(tables, names, a)
    gr = granger(runs, outs, facs, st, a)
    imp = systemic_impact(runs, outs, a, np.random.RandomState(a.seed))

    # Sorties
    out = a.output_dir
    dat = os.path.join(out, "donnees")
    ensure_dir(dat)
    pd.DataFrame(nri.edge_probs, index=names, columns=names).to_csv(os.path.join(dat, "nri_edge_probs.csv"))
    pd.DataFrame(nri.edge_probs_std, index=names, columns=names).to_csv(
        os.path.join(dat, "nri_edge_probs_std.csv")
    )
    pd.DataFrame(nri.importance, index=names, columns=names).to_csv(
        os.path.join(dat, "nri_edge_importance.csv")
    )
    pd.DataFrame(signed, index=names, columns=names).to_csv(os.path.join(dat, "nri_signed_influence.csv"))
    json.dump(
        {"validation": nri.validation, "prediction_vols_non_vus": nri.heldout},
        open(os.path.join(dat, "nri_validation.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=_json,
    )
    json.dump(
        ev.metrics,
        open(os.path.join(dat, "event_models.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=_json,
    )
    json.dump(
        ev.attribution,
        open(os.path.join(dat, "event_root_cause_averages.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=_json,
    )
    json.dump(
        ev.instances,
        open(os.path.join(dat, "event_root_cause_explanations.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=_json,
    )
    gr.to_csv(os.path.join(dat, "granger_tests.csv"), index=False)
    for e, d in imp.items():
        pd.DataFrame(d["impact"], index=names, columns=names).to_csv(os.path.join(dat, f"impact_{e}.csv"))
        pd.DataFrame(d["p"], index=names, columns=names).to_csv(os.path.join(dat, f"impact_{e}_pvalues.csv"))

    # Figures
    A = st.known_adjacency(n)
    det = os.path.join(out, "details") if a.all_figures else None
    if det:
        ensure_dir(det)
    plot_nri(nri, A, names, out, a.importance_thresh, det)
    plot_causes(ev, names, out)
    if det:
        _plot_details(a, det, names, signed, ev, gr, imp)

    rates = {
        e: {
            names[i]: float(np.mean([o.active[e][:, i].mean() for o in outs]))
            for i in range(n)
            if outs[0].applicable[e][i]
        }
        for e in OUTCOMES
    }
    summary = {
        "log_dir": a.log_dir if isinstance(a.log_dir, str) or len(a.log_dir) > 1 else a.log_dir[0],
        "n_vols": len(runs),
        "drones": names,
        "dt": runs[0].dt,
        "structure": {
            "source": st.source,
            "leader": names[st.leader] if st.leader is not None else None,
            "suiveurs": [names[i] for i in st.followers],
            "independants": [names[i] for i in st.independents],
        },
        "avertissements": notes,
        "seuils": {
            "formation_m": a.formation_thresh,
            "navigation_m": a.nav_thresh,
            "quasi_collision_m": a.near_miss_dist,
            "stabilisation_s": a.settle_time,
        },
        "part_du_temps_en_defaillance": rates,
        "nri": {
            "validation": {k: v for k, v in nri.validation.items() if k != "aretes"},
            "prediction_vols_non_vus": nri.heldout,
        },
        "modeles_evenements": {
            e: {k: v for k, v in m.items() if k != "coefficients"} for e, m in ev.metrics.items()
        },
    }
    json.dump(
        summary,
        open(os.path.join(out, "donnees", "summary_report.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=_json,
    )
    print(f"[Termine] Sorties dans {os.path.abspath(out)}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyse causale (NRI, logistique, Granger, impact) d'un essaim.")
    p.add_argument(
        "--log_dir",
        nargs="+",
        default=["logs"],
        help="un vol (drone_*.csv), un repertoire de vols, ou plusieurs campagnes "
        "(regroupees : des vols avec intervention aident a separer cause et correlation)",
    )
    p.add_argument("--output_dir", default="causal_out")
    p.add_argument(
        "--config",
        default=None,
        help="config.yaml pour connaitre la structure de l'essaim (defaut : celui du projet)",
    )
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument(
        "--all_figures",
        action="store_true",
        help="ecrire aussi les figures detaillees (cartes NRI, Granger, impact...) dans details/",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="conserve pour compatibilite (CPU uniquement)")
    p.add_argument("--torch_threads", type=int, default=1)
    p.add_argument("--leader_index", type=int, default=0, help="si aucune config n'est disponible")

    g = p.add_argument_group("defaillances (seuils physiques)")
    g.add_argument("--formation_thresh", type=float, default=1.0, help="ecart a la place en formation [m]")
    g.add_argument("--formation_persist", type=float, default=0.5, help="duree minimale [s]")
    g.add_argument("--nav_thresh", type=float, default=0.3, help="erreur de navigation [m]")
    g.add_argument("--nav_persist", type=float, default=0.25, help="duree minimale [s]")
    g.add_argument(
        "--near_miss_dist",
        "--collision_dist",
        type=float,
        default=0.3,
        help="distance entre drones en vol [m]",
    )
    g.add_argument("--settle_time", type=float, default=3.0, help="decollage ignore [s]")

    g = p.add_argument_group("NRI")
    g.add_argument("--nri_dt", type=float, default=0.2, help="pas de temps du NRI [s]")
    g.add_argument("--seq_len", type=int, default=10, help="fenetre de l'encodeur (en pas NRI)")
    g.add_argument("--n_edge_types", type=int, default=2, help="type 0 = pas d'arete")
    g.add_argument(
        "--own_history", type=int, default=2, help="vitesses propres passees donnees au decodeur (en pas NRI)"
    )
    g.add_argument("--edge_prior", type=float, default=0.8, help="probabilite a priori de 'pas d'arete'")
    g.add_argument("--kl_weight", type=float, default=0.01)
    g.add_argument(
        "--soft_edges",
        dest="hard_edges",
        action="store_false",
        help="aretes douces (par defaut : echantillonnage dur straight-through)",
    )
    g.add_argument("--hidden", type=int, default=64)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--nri_steps", type=int, default=2000)
    g.add_argument("--nri_seeds", type=int, default=3, help="nombre d'apprentissages independants")
    g.add_argument("--max_train_windows", type=int, default=8000)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--tau_start", type=float, default=1.0)
    g.add_argument("--tau_end", type=float, default=0.5)
    g.add_argument("--tau_anneal_steps", type=int, default=1500)
    g.add_argument("--verbose_every", type=int, default=500)
    g.add_argument(
        "--graph_thresh",
        type=float,
        default=0.3,
        help="probabilite d'arete minimale (seulement si l'importance est indisponible)",
    )
    g.add_argument(
        "--importance_thresh",
        type=float,
        default=0.05,
        help="hausse relative d'erreur minimale pour retenir une arete (graphe, influence signee)",
    )

    g = p.add_argument_group("evenements, Granger, impact")
    g.add_argument("--horizon", type=float, default=1.0, help="horizon passe des causes [s]")
    g.add_argument("--min_events", type=int, default=10)
    g.add_argument("--n_boot", type=int, default=200)
    g.add_argument("--attr_min", type=float, default=0.05, help="evidence minimale pour attribuer une cause")
    g.add_argument("--max_events_to_explain", type=int, default=2000)
    g.add_argument("--granger_dt", type=float, default=0.2)
    g.add_argument("--granger_maxlag", type=int, default=5)
    g.add_argument("--impact_horizon", type=float, default=2.0, help="[s]")
    g.add_argument("--n_perm", type=int, default=200)
    return p


def main():
    a = build_argparser().parse_args()
    if a.config is None:
        guess = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        a.config = guess if os.path.exists(guess) else None
    run_pipeline(a)


if __name__ == "__main__":
    main()
