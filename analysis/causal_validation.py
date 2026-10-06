"""Validation par intervention de l'analyse causale.

On rejoue les memes vols (memes graines) en changeant une seule chose, et on mesure
l'effet sur chaque drone : difference intervenu - base vol par vol, IC par bootstrap.
Les vols ne sont pas reproductibles au bit pres (messages ZMQ) : on utilise une double
difference (pendant - avant la fenetre) et on exclut les paires qui divergent avant
l'intervention (--pair_tol).

    python analysis/causal_validation.py --base runs/causal/base --intervention gnss_fort=runs/causal/gnss_fort:8:22 --output_dir runs/causal/validation

Format : nom=repertoire[:debut:fin], fenetre en secondes (par defaut tout le vol)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import causal_analysis as ca  # noqa: E402


def parse_intervention(spec: str) -> Tuple[str, str, Optional[float], Optional[float]]:
    """'nom=repertoire[:debut:fin]' -> (nom, repertoire, debut, fin)."""
    if "=" not in spec:
        raise ValueError(f"intervention mal formee (nom=repertoire[:debut:fin]) : {spec}")
    name, rest = spec.split("=", 1)
    # Fenetre lue a droite (le chemin contient "C:" sous Windows)
    m = re.match(r"^(.+):([^:\\/]*):([^:\\/]*)$", rest)
    if m:
        try:
            return name, m.group(1), float(m.group(2)), float(m.group(3))
        except ValueError:
            raise ValueError(f"fenetre mal formee : {spec}") from None
    return name, rest, None, None


def paired_runs(base_dir: str, int_dir: str) -> List[Tuple[str, str]]:
    """Couples (vol de base, vol intervenu) de meme nom de graine."""
    b = {os.path.basename(p): p for p in ca.find_runs(base_dir)}
    i = {os.path.basename(p): p for p in ca.find_runs(int_dir)}
    common = sorted(set(b) & set(i), key=lambda s: (len(s), s))
    if not common:
        raise ValueError(f"aucune graine commune entre {base_dir} et {int_dir}")
    return [(b[k], i[k]) for k in common]


def align(b: ca.SwarmLogs, i: ca.SwarmLogs) -> Tuple[np.ndarray, np.ndarray]:
    """Indices des instants communs aux deux vols (horodatages au ms)."""
    tb, ti = np.round(b.times, 3), np.round(i.times, 3)
    common, ib, ii = np.intersect1d(tb, ti, return_indices=True)
    if len(common) < 10:
        raise ValueError(f"{b.run_id} : trop peu d'instants communs entre les deux vols")
    return ib, ii


def window_mask(t: np.ndarray, settle: float, t0: Optional[float], t1: Optional[float]) -> np.ndarray:
    lo = settle if t0 is None else max(settle, t0)
    hi = np.inf if t1 is None else t1
    return (t >= lo) & (t < hi)


def bootstrap_ci(x: np.ndarray, n_boot: int, rng: np.random.RandomState, alpha: float = 0.05):
    """Moyenne et IC a 95 % par bootstrap sur les vols (lignes de x)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan
    bs = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return m, float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def per_flight_effects(pairs, st, a, t0, t1):
    """Effets vol par vol : dict de tableaux [n_vols, N], et paires exclues."""
    rows = {"deplacement_m": [], "deplacement_brut_m": [], "ecart_avant_m": []}
    excluded = []
    for e in ca.OUTCOMES:
        rows[f"ace_{e}"] = []
        rows[f"ace_continu_{e}"] = []
        rows[f"base_{e}"] = []
    for pb, pi in pairs:
        b, i = ca.load_run(pb, a.downsample), ca.load_run(pi, a.downsample)
        if b.drone_names != i.drone_names:
            raise ValueError(f"drones differents entre {pb} et {pi}")
        ob, oi = ca.compute_outcomes(b, st, a), ca.compute_outcomes(i, st, a)
        ib, ii = align(b, i)
        w = window_mask(b.times[ib], a.settle_time, t0, t1)
        if not w.any():
            raise ValueError(f"fenetre vide pour {pb}")
        d2 = ((i.data["pos"][ii] - b.data["pos"][ib]) ** 2).sum(-1)  # [T, N]
        tt = b.times[ib]
        pre = (tt >= a.settle_time) & (tt < t0) if t0 is not None else np.zeros_like(w)
        before = np.sqrt(d2[pre].mean(0)) if pre.sum() >= 5 else np.full(d2.shape[1], np.nan)
        if np.nanmax(before, initial=0.0) > a.pair_tol:
            excluded.append(os.path.basename(pb))
            continue
        during = np.sqrt(d2[w].mean(0))
        rows["deplacement_brut_m"].append(during)
        rows["ecart_avant_m"].append(before)
        rows["deplacement_m"].append(during - before if np.isfinite(before).all() else during)
        for e in ca.OUTCOMES:
            app = ob.applicable[e]
            rb = ob.active[e][ib][w].mean(0).astype(float)
            ri = oi.active[e][ii][w].mean(0).astype(float)
            cb = ob.continuous[e][ib][w].mean(0)
            ci = oi.continuous[e][ii][w].mean(0)
            rows[f"ace_{e}"].append(np.where(app, ri - rb, np.nan))
            rows[f"ace_continu_{e}"].append(np.where(app, ci - cb, np.nan))
            rows[f"base_{e}"].append(np.where(app, rb, np.nan))
    if not rows["deplacement_m"]:
        raise ValueError("toutes les paires sont rompues avant l'intervention (voir --pair_tol)")
    return {k: np.array(v) for k, v in rows.items()}, excluded


def summarize(
    eff: Dict[str, np.ndarray],
    names: List[str],
    st,
    n_boot: int,
    seed: int,
    did: bool,
    min_effect: float = 0.05,
) -> pd.DataFrame:
    """Tableau des effets. Significatif : IC 95 % qui exclut 0 ; pour le deplacement,
    effet > min_effect m (ou, sans fenetre, au-dessus du drone independant)."""
    rng = np.random.RandomState(seed)
    out = []
    indep = list(st.independents)
    floor = None
    if not did and indep:
        d = eff["deplacement_m"][:, indep].ravel()
        floor = float(np.quantile(d[np.isfinite(d)], 0.95)) if np.isfinite(d).any() else None
    for key, arr in eff.items():
        for j, nm in enumerate(names):
            m, lo, hi = bootstrap_ci(arr[:, j], n_boot, rng)
            row = dict(
                mesure=key,
                drone=nm,
                moyenne=m,
                ic_bas=lo,
                ic_haut=hi,
                n_vols=int(np.isfinite(arr[:, j]).sum()),
            )
            if key.startswith("ace_"):
                row["significatif"] = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
            elif key == "deplacement_m" and did:
                row["significatif"] = bool(np.isfinite(lo) and lo > 0 and m > min_effect)
            elif key == "deplacement_m":
                row["plancher_temoin_m"] = floor
                row["significatif"] = bool(
                    floor is not None and np.isfinite(lo) and lo > floor and j not in indep
                )
            out.append(row)
    return pd.DataFrame(out)


# Confrontation avec l'analyse observationnelle
def compare_with_analysis(analysis_dir: str, names: List[str]) -> Dict:
    """Conclusions de causal_analysis.py sur les vols intervenus."""
    res: Dict[str, object] = {"repertoire_analyse": analysis_dir}
    if not analysis_dir or not os.path.isdir(analysis_dir):
        res["note"] = "analyse observationnelle absente"
        return res

    def find(f):
        for d in (os.path.join(analysis_dir, "donnees"), analysis_dir):
            if os.path.exists(os.path.join(d, f)):
                return os.path.join(d, f)
        return os.path.join(analysis_dir, f)

    def load_json(f):
        p = find(f)
        return json.load(open(p)) if os.path.exists(p) else None

    att = load_json("event_root_cause_averages.json") or {}
    models = load_json("event_models.json") or {}
    res["attribution"] = {e: {c: dict(zip(names, v)) for c, v in d.items()} for e, d in att.items()}
    res["modeles"] = {
        e: {
            k: m.get(k)
            for k in (
                "statut",
                "n_apparitions",
                "auroc_vols_non_vus",
                "brier_vols_non_vus",
                "brier_reference_taux_constant",
            )
            if k in m
        }
        | {"coefficients": m.get("coefficients")}
        for e, m in models.items()
    }
    gp = find("granger_tests.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        if "significatif_bonferroni" in g:
            ok = g[g.statut == "ok"]
            res["granger_part_significative"] = (
                ok.groupby(["defaillance", "cause", "drone"])["significatif_bonferroni"]
                .mean()
                .reset_index()
                .to_dict("records")
            )
    ip = find("nri_edge_importance.csv")
    if os.path.exists(ip):
        res["nri_importance"] = pd.read_csv(ip, index_col=0).to_dict()
    return res


# Figures
def plot_effects(tables: Dict[str, pd.DataFrame], names: List[str], path: str) -> None:
    keys = [
        ("deplacement_m", "Deplacement de trajectoire [m]\n(double difference si fenetre)"),
        ("ace_continu_nav_degradation", "Effet sur l'erreur de navigation [m]"),
        ("ace_formation_loss", "Effet sur la part du temps hors formation"),
        ("ace_near_miss", "Effet sur la part du temps en quasi-collision"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8))
    width = 0.8 / max(1, len(tables))
    x = np.arange(len(names))
    for ax, (k, title) in zip(axes, keys):
        for n_i, (iv, tab) in enumerate(tables.items()):
            sub = tab[tab.mesure == k].set_index("drone").reindex(names)
            m = sub.moyenne.to_numpy(float)
            lo, hi = sub.ic_bas.to_numpy(float), sub.ic_haut.to_numpy(float)
            err = np.vstack([np.nan_to_num(m - lo), np.nan_to_num(hi - m)])
            ax.bar(
                x + (n_i - (len(tables) - 1) / 2) * width,
                np.nan_to_num(m),
                width,
                yerr=err,
                capsize=3,
                label=iv,
            )
            if k == "deplacement_m" and "plancher_temoin_m" in sub and sub.plancher_temoin_m.notna().any():
                ax.axhline(float(sub.plancher_temoin_m.dropna().iloc[0]), ls=":", lw=0.8, color="k")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, names, rotation=30)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("Effets causaux mesures par intervention (vols apparies, IC 95 % bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


LABELS = {
    "gnss_fort": "Strong GNSS jamming on drone_1 only (8-22 s)",
    "vent_fort": "Very strong wind on drone_2 only (8-22 s)",
}
FAIL = [
    ("ace_formation_loss", "out of formation", "#9467bd"),
    ("ace_nav_degradation", "degraded navigation", "#d62728"),
]


def _attribution(analysis_dir: str, names: List[str]):
    """Parts des causes par drone, toutes defaillances confondues (ponderees par le nombre de cas)."""
    p = None
    for d in (os.path.join(analysis_dir or "", "donnees"), analysis_dir or ""):
        if os.path.exists(os.path.join(d, "event_root_cause_averages.json")):
            p = os.path.join(d, "event_root_cause_averages.json")
    if p is None:
        return None
    att = json.load(open(p))
    tot, cnt = {}, np.zeros(len(names))
    for d in att.values():
        n = np.array(d["_n"], float)
        cnt += n
        for c, v in d.items():
            if not c.startswith("_"):
                tot[c] = tot.get(c, np.zeros(len(names))) + np.array(v) * n
    return {c: v / np.maximum(cnt, 1) for c, v in tot.items()}, cnt


def plot_summary(tables: Dict[str, pd.DataFrame], names: List[str], analyses: Dict[str, str], path: str):
    """Une ligne par intervention : a gauche l'effet mesure, a droite les causes trouvees par l'analyse."""
    rows = list(tables)
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.4 * len(rows)), squeeze=False)
    x = np.arange(len(names))
    for r, iv in enumerate(rows):
        tab = tables[iv]
        ax = axes[r, 0]
        w = 0.38
        for k, (key, lab, col) in enumerate(FAIL):
            sub = tab[tab.mesure == key].set_index("drone").reindex(names)
            m = 100 * sub.moyenne.to_numpy(float)
            lo, hi = 100 * sub.ic_bas.to_numpy(float), 100 * sub.ic_haut.to_numpy(float)
            err = np.vstack([np.nan_to_num(m - lo), np.nan_to_num(hi - m)])
            ax.bar(x + (k - 0.5) * w, np.nan_to_num(m), w, yerr=err, capsize=3, color=col, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, names)
        ax.set_ylabel("extra time in failure\n[% of the time]")
        ax.set_title(f"{LABELS.get(iv, iv)}\nWhat really happened (same flights replayed)", fontsize=9)
        ax.legend(fontsize=7)

        ax = axes[r, 1]
        res = _attribution(analyses.get(iv, ""), names)
        if res is None:
            ax.text(0.5, 0.5, "no analysis", ha="center", va="center")
            ax.axis("off")
            continue
        sh, cnt = res
        keep = [i for i in range(len(names)) if cnt[i] > 0]
        left = np.zeros(len(keep))
        for c, v in sh.items():
            vals = 100 * v[keep]
            if np.all(vals == 0):
                continue
            ax.barh(
                range(len(keep)),
                vals,
                left=left,
                color=ca.CAUSE_COLORS.get(c, "#888"),
                label=ca.CAUSE_LABELS.get(c, c),
            )
            left += vals
        ax.set_yticks(range(len(keep)), [f"{names[i]} ({int(cnt[i])} cases)" for i in keep])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of failures [%]")
        ax.set_title("Cause found by the analysis (which does not know the perturbation)", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="repertoire des vols de base")
    p.add_argument(
        "--intervention", action="append", required=True, help="nom=repertoire[:debut:fin] (repetable)"
    )
    p.add_argument(
        "--analysis",
        action="append",
        default=[],
        help="nom=repertoire de sortie de causal_analysis.py sur les vols intervenus (option)",
    )
    p.add_argument("--output_dir", default="runs/causal/validation")
    p.add_argument("--config", default=os.path.join(os.path.dirname(_HERE), "config.yaml"))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument(
        "--pair_tol",
        type=float,
        default=1.0,
        help="ecart max de l'essaim avant intervention pour garder une paire [m]",
    )
    p.add_argument(
        "--min_effect",
        type=float,
        default=0.05,
        help="deplacement minimal (double difference) juge physiquement notable [m]",
    )
    p.add_argument("--seed", type=int, default=0)
    # memes seuils que l'analyse (on reutilise ses valeurs par defaut)
    ref = ca.build_argparser()
    for act in ref._actions:
        if act.dest in (
            "formation_thresh",
            "formation_persist",
            "nav_thresh",
            "nav_persist",
            "near_miss_dist",
            "settle_time",
            "downsample",
            "leader_index",
        ):
            p.add_argument(*act.option_strings, type=act.type, default=act.default, help=act.help)
    return p


def main(argv=None) -> Dict:
    a = build_argparser().parse_args(argv)
    ca.ensure_dir(a.output_dir)
    dat = os.path.join(a.output_dir, "donnees")
    ca.ensure_dir(dat)
    analyses = dict(s.split("=", 1) for s in a.analysis)
    report, tables = {}, {}
    names = None
    for spec in a.intervention:
        name, d, t0, t1 = parse_intervention(spec)
        pairs = paired_runs(a.base, d)
        first = ca.load_run(pairs[0][0], a.downsample)
        names = first.drone_names
        st = ca.swarm_structure(a.config, names, a.leader_index)
        eff, excluded = per_flight_effects(pairs, st, a, t0, t1)
        did = t0 is not None
        tab = summarize(eff, names, st, a.n_boot, a.seed, did, a.min_effect)
        tab.insert(0, "intervention", name)
        tables[name] = tab
        tab.to_csv(os.path.join(dat, f"effets_{name}.csv"), index=False)
        dep = tab[tab.mesure == "deplacement_m"].set_index("drone")
        report[name] = {
            "repertoire": d,
            "fenetre_s": [t0, t1],
            "n_paires": len(pairs),
            "paires_rompues_exclues": excluded,
            "deplacement": "double difference (pendant - avant)"
            if did
            else "ecart brut, temoin = drone independant",
            "deplacement_m": {
                k: [round(v.moyenne, 4), round(v.ic_bas, 4), round(v.ic_haut, 4)] for k, v in dep.iterrows()
            },
            "drones_affectes": [k for k, v in dep.iterrows() if v.get("significatif")],
            "ace_significatifs": tab[(tab.mesure.str.startswith("ace_")) & (tab.significatif == True)][  # noqa: E712
                ["mesure", "drone", "moyenne", "ic_bas", "ic_haut"]
            ]
            .round(4)
            .to_dict("records"),
            "analyse_observationnelle": compare_with_analysis(analyses.get(name, ""), names),
        }
        print(
            f"[{name}] {len(pairs) - len(excluded)} paires valides ({len(excluded)} rompues : "
            f"{excluded}) ; drones affectes : {report[name]['drones_affectes']}"
        )
        for r in report[name]["ace_significatifs"]:
            print(
                f"    {r['mesure']:<32} {r['drone']:<8} {r['moyenne']:+.4f} [{r['ic_bas']:+.4f}, {r['ic_haut']:+.4f}]"
            )
    plot_summary(tables, names, analyses, os.path.join(a.output_dir, "resume_etude.png"))
    plot_effects(tables, names, os.path.join(dat, "effets_interventions.png"))
    json.dump(
        report,
        open(os.path.join(dat, "validation_interventionnelle.json"), "w"),
        indent=2,
        ensure_ascii=False,
        default=ca._json,
    )
    print(f"[Termine] {os.path.abspath(a.output_dir)}")
    return report


if __name__ == "__main__":
    main()
