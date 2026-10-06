#!/usr/bin/env python3
"""
Coupure GNSS : derive inertielle et reconvergence, filtre 6 etats contre ESKF.

On rejoue une trajectoire VRAIE (enregistree pendant un vol PyBullet, ou
synthetique) en regenerant les mesures capteurs N fois avec des bruits et des
biais differents, et on fait tourner les deux filtres sur EXACTEMENT les memes
mesures. On coupe le GNSS sur une fenetre choisie.

Ce que la figure doit montrer
-----------------------------
1. Pendant la coupure, le filtre n'a plus que l'inertie : l'erreur derive.
   La vitesse de derive depend de ce que le filtre a appris AVANT la coupure :
   un filtre qui a estime les biais capteurs derive moins.
2. L'enveloppe 3-sigma doit s'ouvrir au meme rythme que l'erreur. Un filtre
   dont l'erreur sort de son enveloppe ne sait pas qu'il derive : c'est un
   defaut d'integrite, plus grave qu'une derive annoncee.
3. Au retour du GNSS, l'erreur doit reconverger en quelques mises a jour.

Precision importante sur la comparaison
---------------------------------------
Le filtre 6 etats n'estime pas l'attitude : on lui fournit l'attitude VRAIE.
C'est un avantage qu'aucun systeme reel n'a. Malgre cela, avec un MEMS
realiste, l'ESKF derive nettement moins, parce qu'il a appris les biais.

Usage
-----
  python analysis/gnss_outage.py --truth runs/mc/s1/drone_1_truth.csv --outage 8 18
  python analysis/gnss_outage.py --synthetic maneuver --duration 60 --outage 30 45
  options : --runs 40  --imu-preset mems_nav|config_biais  --out runs/outage
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.nav_replay import (  # noqa: E402
    IMU_PRESETS,
    load_truth,
    replay,  # noqa: E402
    synthetic_truth,
)

COLORS = {"eskf": "#1f77b4", "kf6": "#d62728"}
LABELS = {"eskf": "15-state ESKF (estimated attitude)", "kf6": "6-state filter (TRUE attitude given)"}


def run(truth, outage, n_runs, imu_cfg, every=2, seed0=1000):
    """Rejoue n_runs tirages. Retourne {filtre: DataFrame concatene avec 'run'}."""
    out = {"eskf": [], "kf6": []}
    for i in range(n_runs):
        res = replay(truth, ("eskf", "kf6"), imu_cfg=imu_cfg, seed=seed0 + i, outages=[outage], every=every)
        for name, df in res.items():
            df = df.copy()
            df["run"] = i
            df["err"] = np.sqrt(df.e_px**2 + df.e_py**2 + df.e_pz**2)
            df["sig3"] = 3.0 * np.sqrt(df.sig_px**2 + df.sig_py**2 + df.sig_pz**2)
            # Test d'integrite PAR AXE : |e_i| <= 3 sigma_i doit tenir 99.7 % du
            # temps. Comparer la norme 3D a 3 * sqrt(somme des variances) est bien
            # plus laxiste (~99.999 % pour un filtre coherent) et laisse passer un
            # filtre dont l'erreur reelle est 1.5 fois plus grande qu'annoncee.
            df["in3s"] = (
                (df.e_px.abs() <= 3 * df.sig_px)
                & (df.e_py.abs() <= 3 * df.sig_py)
                & (df.e_pz.abs() <= 3 * df.sig_pz)
            )
            df["axes_in3s"] = (
                (df.e_px.abs() <= 3 * df.sig_px).astype(float)
                + (df.e_py.abs() <= 3 * df.sig_py)
                + (df.e_pz.abs() <= 3 * df.sig_pz)
            ) / 3.0
            out[name].append(df)
        print(f"\r  tirage {i + 1}/{n_runs}", end="", flush=True)
    print()
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def metrics(res, outage):
    t0, t1 = outage
    rows = []
    for name, df in res.items():
        inside = df[(df.time >= t0) & (df.time <= t1)]
        before = df[(df.time > min(3.0, 0.5 * t0)) & (df.time < t0)]
        end = inside.loc[inside.groupby("run").time.idxmax()]
        nominal = float(np.sqrt((before.err**2).mean()))
        # Reconvergence : premier instant apres la coupure ou l'erreur repasse
        # sous 2x le RMSE nominal (et y reste 1 s)
        reconv = []
        for _, g in df[df.time > t1].groupby("run"):
            g = g.sort_values("time")
            ok = (g.err < 2 * nominal).to_numpy()
            tt = g.time.to_numpy()
            found = np.nan
            for j in range(len(ok)):
                if ok[j]:
                    w = ok[j:][tt[j:] <= tt[j] + 1.0]
                    if w.all():
                        found = tt[j] - t1
                        break
            reconv.append(found)
        rows.append(
            dict(
                filtre=name,
                rmse_nominal=nominal,
                derive_mediane=float(end.err.median()),
                derive_q95=float(end.err.quantile(0.95)),
                dans_3sigma=100.0 * float(inside.axes_in3s.mean()),
                reconvergence=float(np.nanmedian(reconv)) if np.isfinite(reconv).any() else np.nan,
            )
        )
    return pd.DataFrame(rows)


def figure(res, outage, path, title_suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, t1 = outage
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, 1]))

    # (a) faisceau Monte-Carlo de l'erreur
    ax = axes[0]
    ax.axvspan(t0, t1, color="0.88", zorder=0)
    ax.text(
        (t0 + t1) / 2,
        0.97,
        "GNSS outage",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="0.35",
    )
    for name in ("kf6", "eskf"):
        df = res[name]
        g = df.groupby(df.time.round(2))
        q = g.err.quantile([0.05, 0.5, 0.95]).unstack()
        s3 = g.sig3.median()
        ax.fill_between(q.index, q[0.05], q[0.95], color=COLORS[name], alpha=0.15, lw=0)
        ax.plot(q.index, q[0.5], color=COLORS[name], lw=1.8, label=f"{LABELS[name]}: median")
        ax.plot(
            s3.index, s3.values, color=COLORS[name], lw=1.1, ls="--", label=r"reported $3\sigma$ envelope"
        )
    ax.set_yscale("log")
    ax.set_ylabel("position error [m]")
    n = res["eskf"].run.nunique()
    ax.set_title(f"GNSS outage: drift and reconvergence ({n} Monte-Carlo runs, 5-95 % band)" + title_suffix)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # (b) rapport erreur / enveloppe : > 1 = le filtre ne sait pas qu'il derive
    ax = axes[1]
    ax.axvspan(t0, t1, color="0.88", zorder=0)
    for name in ("kf6", "eskf"):
        df = res[name]
        # pire axe : max_i |e_i| / (3 sigma_i), coherent avec le test d'integrite
        worst = np.maximum.reduce(
            [
                df.e_px.abs() / (3 * df.sig_px),
                df.e_py.abs() / (3 * df.sig_py),
                df.e_pz.abs() / (3 * df.sig_pz),
            ]
        )
        ratio = pd.Series(worst).groupby(df.time.round(2).to_numpy())
        q = ratio.quantile([0.5, 0.95]).unstack()
        ax.plot(q.index, q[0.95], color=COLORS[name], lw=1.4, label=f"{name.upper()}: 95th percentile")
        ax.plot(q.index, q[0.5], color=COLORS[name], lw=0.9, ls=":")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.set_ylabel(r"max$_i$ |e$_i$| / 3$\sigma_i$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(
        0.995,
        1.0,
        "above 1: error outside the reported envelope ",
        transform=ax.get_yaxis_transform(),
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_title("Integrity: does the filter know it is drifting? (dotted: median)", fontsize=10)

    # (c) apprentissage des biais par l'ESKF, un tirage
    ax = axes[2]
    ax.axvspan(t0, t1, color="0.88", zorder=0)
    one = res["eskf"][res["eskf"].run == 0].sort_values("time")
    for a, c in zip("xyz", ("#1f77b4", "#2ca02c", "#9467bd")):
        ax.plot(one.time, one[f"ba{a}"], color=c, lw=1.4, label=f"estimated bias {a}")
        ax.plot(one.time, one[f"ba_true{a}"], color=c, lw=1.0, ls="--")
        ax.fill_between(
            one.time,
            one[f"ba{a}"] - 3 * one[f"sig_ba{a}"],
            one[f"ba{a}"] + 3 * one[f"sig_ba{a}"],
            color=c,
            alpha=0.08,
            lw=0,
        )
    ax.set_ylabel(r"accelerometer bias [m/s$^2$]")
    ax.set_xlabel("time [s]")
    ax.set_title("ESKF: estimated (solid) and true (dashed) accelerometer biases, one run", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--truth", help="journal de verite <drone>_truth.csv d'un vol")
    src.add_argument("--synthetic", choices=["maneuver", "hover"])
    ap.add_argument("--duration", type=float, default=60.0, help="(synthetique) duree [s]")
    ap.add_argument("--outage", nargs=2, type=float, required=True, metavar=("DEBUT", "FIN"))
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--imu-preset", choices=sorted(IMU_PRESETS), default="mems_nav")
    ap.add_argument("--out", default="runs/outage")
    args = ap.parse_args()

    truth = load_truth(args.truth) if args.truth else synthetic_truth(args.duration, kind=args.synthetic)
    t_end = float(truth["t"][-1])
    if not (truth["t"][0] < args.outage[0] < args.outage[1] < t_end):
        sys.exit(f"la coupure doit etre comprise dans la trajectoire [0, {t_end:.1f}] s")
    os.makedirs(args.out, exist_ok=True)

    print(
        f"Rejeu : {args.runs} tirages, IMU={args.imu_preset}, coupure "
        f"{args.outage[0]:.0f}-{args.outage[1]:.0f} s sur {t_end:.0f} s de trajectoire"
    )
    res = run(truth, tuple(args.outage), args.runs, IMU_PRESETS[args.imu_preset])
    M = metrics(res, tuple(args.outage))
    dur = args.outage[1] - args.outage[0]
    print(
        f"\n{'filtre':8}{'RMSE nominal':>14}{'derive fin coupure':>22}{'q95':>8}"
        f"{'dans 3 sigma':>15}{'reconvergence':>15}"
    )
    for r in M.itertuples():
        print(
            f"{r.filtre:8}{r.rmse_nominal:12.3f} m{r.derive_mediane:18.2f} m (mediane)"
            f"{r.derive_q95:6.2f} m{r.dans_3sigma:13.1f} %{r.reconvergence:13.2f} s"
        )
    print(
        "\n'dans 3 sigma' : pendant la coupure, part des echantillons ou chaque axe "
        "respecte |e| <= 3 sigma (attendu 99.7 % pour un filtre coherent)."
    )
    tag = f"{args.imu_preset}_{dur:.0f}s"
    fig = figure(
        res,
        tuple(args.outage),
        os.path.join(args.out, f"fig_coupure_gnss_{tag}.png"),
        title_suffix=f"\nIMU: {args.imu_preset}, {dur:.0f} s outage",
    )
    M.to_csv(os.path.join(args.out, f"metriques_{tag}.csv"), index=False)
    print(f"figure : {fig}")


if __name__ == "__main__":
    main()
