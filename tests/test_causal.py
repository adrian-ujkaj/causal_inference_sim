"""
Tests de causal_analysis.py et causal_validation.py sur donnees synthetiques.
Lancer depuis la racine du depot :  python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))

torch = pytest.importorskip("torch")
import causal_analysis as ca  # noqa: E402
import causal_validation as cv  # noqa: E402

NAMES = ["drone_0", "drone_1", "drone_2", "drone_3"]
ST = ca.SwarmStructure(leader=0, followers=[1, 2], independents=[3], source="test")


def _args(**kw):
    a = ca.build_argparser().parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# Briques elementaires
def test_persist_and_onsets():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)[:, None]
    p = ca.persist(m, 3)
    assert p[:, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]
    assert ca.onsets(np.array([1, 1, 0, 1, 0])[:, None])[:, 0].tolist() == [1, 0, 0, 1, 0]


def test_known_adjacency():
    A = ST.known_adjacency(4)
    assert A[1, 0] == 1 and A[2, 0] == 1  # leader -> suiveurs
    assert A[0, 1] == 0 and A[0, 2] == 0  # pas de suiveur -> leader
    assert np.isnan(A[1, 2]) and np.isnan(A[2, 1])  # suiveur <-> suiveur indetermine
    assert np.all(A[3, :3] == 0) and np.all(A[:3, 3] == 0)


def _logs(T=400, dt=0.05, yaw_rate=0.3, follower_noise=0.0, seed=0):
    """Leader en virage, suiveurs fixes dans le repere du leader, drone_3 immobile."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) * dt
    psi = yaw_rate * t
    pos = np.zeros((T, 4, 3))
    pos[:, 0, 0], pos[:, 0, 1], pos[:, 0, 2] = 5 * np.cos(0.2 * t), 5 * np.sin(0.2 * t), 2.0
    for f, off in ((1, (-1.0, 1.0)), (2, (-1.0, -1.0))):
        c, s = np.cos(psi), np.sin(psi)
        pos[:, f, 0] = pos[:, 0, 0] + c * off[0] - s * off[1]
        pos[:, f, 1] = pos[:, 0, 1] + s * off[0] + c * off[1]
        pos[:, f, 2] = 2.0
    pos[:, 3] = [-5.0, 0.0, 10.0]
    pos[:, 1:3, :2] += follower_noise * rng.standard_normal((T, 2, 2))
    vel = np.gradient(pos, dt, axis=0)
    yaw = np.repeat(psi[:, None], 4, axis=1)
    data = {
        "pos": pos,
        "vel": vel,
        "wind": np.zeros((T, 4, 3)),
        "wind_mag": np.zeros((T, 4)),
        "contact": np.zeros((T, 4)),
        "nav_err": np.zeros((T, 4)),
        "gnss_err": np.zeros((T, 4)),
        "yaw": yaw,
    }
    return ca.SwarmLogs("s1", t, NAMES, data, dt)


def test_formation_error_in_leader_frame():
    """Formation rigide qui tourne avec le leader : erreur nulle dans son repere."""
    fe = ca.formation_error(_logs(), ST, settle_s=1.0)
    assert np.abs(fe[:, 1:3]).max() < 1e-6
    assert np.all(fe[:, [0, 3]] == 0)


def test_outcomes_thresholds():
    lg = _logs()
    lg.data["nav_err"][200:240, 3] = 1.0  # 2 s au-dessus de 0.3 m
    out = ca.compute_outcomes(lg, ST, _args())
    nav = out.active["nav_degradation"][:, 3]
    assert nav[:200].sum() == 0 and nav[210:240].all()
    assert out.active["formation_loss"].sum() == 0
    assert not out.applicable["formation_loss"][0]


def test_validate_graph_auroc():
    A = ST.known_adjacency(4)
    P = np.where(A == 1, 0.9, 0.1)
    np.fill_diagonal(P, np.nan)
    v = ca.validate_graph(P, A, NAMES)
    assert v["auroc"] == 1.0 and v["separation_parfaite"]
    v2 = ca.validate_graph(1 - P, A, NAMES)
    assert v2["auroc"] == 0.0


# Decodeur NRI
def test_decoder_permutation_identity_and_type0():
    torch.manual_seed(0)
    send, recv = ca.edge_index(4)
    send, recv = torch.as_tensor(send), torch.as_tensor(recv)
    E = len(send)
    dec = ca.NRIDecoder(u_dim=4, hidden=16, k=2, hist=2)
    B = 8
    x, u, vh = torch.randn(B, 4, 8), torch.randn(B, 4, 4), torch.randn(B, 4, 6)
    z = torch.zeros(B, E, 2)
    z[..., 1] = 1.0
    base = dec(x, u, z, send, recv, vh=vh)
    same = dec(x, u, z, send, recv, perm_edge=0, perm_idx=torch.arange(B), vh=vh)
    assert torch.allclose(base, same)
    perm = dec(x, u, z, send, recv, perm_edge=0, perm_idx=torch.roll(torch.arange(B), 1), vh=vh)
    r = int(recv[0])
    others = [i for i in range(4) if i != r]
    assert torch.allclose(base[:, others], perm[:, others])  # seul le recepteur de l'arete bouge
    assert not torch.allclose(base[:, r], perm[:, r])
    z0 = torch.zeros(B, E, 2)
    z0[..., 0] = 1.0  # type 0 = aucune arete
    p0 = dec(x, u, z0, send, recv, perm_edge=0, perm_idx=torch.roll(torch.arange(B), 1), vh=vh)
    assert torch.allclose(p0, dec(x, u, z0, send, recv, vh=vh))


def test_nri_dataset_shapes_and_takeoff_excluded():
    lg = _logs(T=200)
    ((x, u),) = ca.nri_dataset([lg], nri_dt=0.2, settle_s=3.0)
    assert x.shape[1:] == (4, 8) and u.shape[1:] == (4, 4)
    assert x.shape[0] == len(np.arange(200)[lg.times >= 3.0][::4])


# Validation interventionnelle
def test_parse_intervention_and_window():
    assert cv.parse_intervention("j=runs/x:10:20") == ("j", "runs/x", 10.0, 20.0)
    assert cv.parse_intervention("w=runs/y") == ("w", "runs/y", None, None)
    assert cv.parse_intervention(r"j=C:\Users\a (4)\x:10:20") == ("j", r"C:\Users\a (4)\x", 10.0, 20.0)
    assert cv.parse_intervention(r"w=C:\Users\a\y") == ("w", r"C:\Users\a\y", None, None)
    with pytest.raises(ValueError):
        cv.parse_intervention("runs/x")
    with pytest.raises(ValueError):
        cv.parse_intervention("j=runs/x:10:abc")
    m = cv.window_mask(np.arange(30.0), settle=3.0, t0=10.0, t1=20.0)
    assert m.sum() == 10 and m[10] and not m[20]


def test_bootstrap_ci_contains_mean():
    rng = np.random.RandomState(0)
    m, lo, hi = cv.bootstrap_ci(np.r_[np.ones(10), 3 * np.ones(10)], 500, rng)
    assert m == 2.0 and lo < 2.0 < hi


def _write_run(d, lg):
    os.makedirs(d, exist_ok=True)
    for i, nm in enumerate(lg.drone_names):
        p, v = lg.data["pos"][:, i], lg.data["vel"][:, i]
        df = pd.DataFrame(
            {
                "time": lg.times,
                "gt_x": p[:, 0],
                "gt_y": p[:, 1],
                "gt_z": p[:, 2],
                "gt_vx": v[:, 0],
                "gt_vy": v[:, 1],
                "gt_vz": v[:, 2],
                "wind_x": 0.0,
                "wind_y": 0.0,
                "wind_z": 0.0,
                "collision_flag": 0,
                "ekf_pos_error_mag": lg.data["nav_err"][:, i],
                "gnss_error_mag": 0.0,
            }
        )
        df.to_csv(os.path.join(d, f"{nm}.csv"), index=False)


def test_paired_intervention_recovers_affected_drone(tmp_path):
    """Seul drone_1, deplace entre 10 et 20 s, est detecte ; la paire rompue est exclue."""
    for k in range(1, 7):
        b = _logs(T=600, follower_noise=0.02, seed=k)
        i = _logs(T=600, follower_noise=0.02, seed=k)
        w = (i.times >= 10) & (i.times < 20)
        i.data["pos"][w, 1, 0] += 0.5
        i.data["nav_err"][w, 1] = 1.0
        if k == 6:  # paire rompue des le depart
            i.data["pos"][:, :3, 0] += 5.0
        _write_run(str(tmp_path / "base" / f"s{k}"), b)
        _write_run(str(tmp_path / "int" / f"s{k}"), i)
    out = tmp_path / "val"
    rep = cv.main(
        [
            "--base",
            str(tmp_path / "base"),
            "--intervention",
            f"j={tmp_path / 'int'}:10:20",
            "--output_dir",
            str(out),
            "--n_boot",
            "200",
            "--config",
            "absent.yaml",
            "--settle_time",
            "1",
        ]
    )
    r = rep["j"]
    assert r["paires_rompues_exclues"] == ["s6"]
    assert r["drones_affectes"] == ["drone_1"]
    ace = {(x["mesure"], x["drone"]) for x in r["ace_significatifs"]}
    assert ("ace_nav_degradation", "drone_1") in ace
    # near_miss exclu : grandeur de paire, deplacer drone_1 la change aussi pour les autres
    assert not any(d == "drone_0" for m, d in ace if "near_miss" not in m)
    assert (out / "resume_etude.png").exists()


def test_granger_detects_lagged_cause():
    """Le test de Granger trouve la cause retardee x -> y."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = np.r_[0.0, 0.0, 0.8 * x[:-2]] + 0.3 * rng.standard_normal(300)
    res = ca._granger_tests(np.column_stack([y, x]), 3)
    assert res[2][0]["ssr_ftest"][1] < 1e-6
