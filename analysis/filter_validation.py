#!/usr/bin/env python3
"""
Validation statistique du filtre de navigation par Monte-Carlo de simulations PyBullet
completes : RMSE compare au GNSS brut, enveloppe 3-sigma, NEES (dim 6 et 15) et NIS.
Les bornes chi2 sont trop etroites car les echantillons d'un meme vol sont correles.

Usage :
    python analysis/filter_validation.py --runs 30 --tmax 20
    python analysis/filter_validation.py --runs 30 --filter kf6 --outage 8 18
    python analysis/filter_validation.py --logs runs/mc --no-sim
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PV_DIM, FULL_DIM, MEAS_DIM = 6, 15, 6


_RUNNER_SRC = '''
import os, sys, json, signal, yaml
# Un arret par `timeout` (SIGTERM) sautait le bloc finally et les journaux en
# memoire etaient perdus : on convertit le signal en sortie normale.
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
cfg = yaml.safe_load(open("config.yaml"))
cfg["simulation"].update(connect_mode="direct", realtime=False,
                         max_sim_time=float(os.environ["TMAX"]),
                         seed=int(os.environ["SEED"]))
OFF = int(os.environ.get("PORT_OFFSET", "0"))
if OFF:
    for sw in cfg.get("swarm", []) or []:
        for k in ("port_in", "port_out"):
            if k in sw: sw[k] += OFF
    for a in cfg["agents"]:
        if "port_out" in a: a["port_out"] += OFF
        for r in (a.get("radar") or []):
            if "port" in r: r["port"] += OFF
ftype = os.environ.get("FILTER_TYPE")
override = json.loads(os.environ.get("AGENT_OVERRIDE", "null")) or {}
preset = json.loads(os.environ.get("IMU_PRESET", "null"))
outage = json.loads(os.environ.get("OUTAGE", "null"))
jam = json.loads(os.environ.get("JAM", "null"))
seed = int(os.environ["SEED"])

def deep_merge(dst, src):
    """Fusion recursive : {"sensors": {"gnss": {...}}} ne doit pas effacer sensors.imu."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v

# Surcharges : cles "@<nom>" = pour ce drone seulement, autres cles = pour tous
override_all = {k: v for k, v in override.items() if not k.startswith("@")}
override_one = {k[1:]: v for k, v in override.items() if k.startswith("@")}
names = {a.get("name") for a in cfg["agents"]}
for nm in override_one:
    if nm not in names:
        raise SystemExit(f"surcharge pour un agent inconnu : {nm}")
for i, a in enumerate(cfg["agents"]):
    if a.get("type") != "uav":
        continue
    a["log_dir"] = os.environ["LOGDIR"]
    # Planification synchrone : a graine egale, trajectoire identique, ce qui
    # rend les comparaisons appariees (KF6 / ESKF sur les memes vols) valides.
    a.setdefault("planning_sync", os.environ.get("PLANNING_SYNC", "1") == "1")
    deep_merge(a, override_all)
    deep_merge(a, override_one.get(a.get("name"), {}))
    sens = a.setdefault("sensors", {})
    if ftype:
        a.setdefault("filter", {})["type"] = ftype
    if preset is not None:
        imu = {k: v for k, v in (sens.get("imu") or {}).items()
               if k in ("enabled", "gravity")}
        imu.update(preset)
        imu["seed"] = seed * 100 + i          # biais differents par drone et par tirage
        sens["imu"] = imu
    g = sens.setdefault("gnss", {})
    if outage:
        g["outage_start"], g["outage_end"] = float(outage[0]), float(outage[1])
    if jam:
        g["jam_start"], g["jam_end"] = float(jam[0]), float(jam[1])
        g["jam_multiplier"] = float(jam[2]) if len(jam) > 2 else 30.0
from simulator.simulator_manager import SimulationManager
sim = SimulationManager(cfg)
try: sim.run()
finally: sim.stop()
'''


def run_monte_carlo(
    n_runs: int,
    tmax: float,
    out_dir: str,
    *,
    filter_type=None,
    imu_preset=None,
    outage=None,
    jam=None,
    workers: int = 4,
    seed0: int = 1,
    agent_override: dict | None = None,
) -> str:
    """Lance n_runs simulations headless, chacune dans sa copie du projet avec des ports ZMQ decales."""
    os.makedirs(out_dir, exist_ok=True)
    preset = None
    if imu_preset:
        from analysis.nav_replay import IMU_PRESETS

        preset = IMU_PRESETS[imu_preset]

    pending = list(range(seed0, seed0 + n_runs))
    running, done, failures = [], 0, 0
    print(
        f"Monte-Carlo : {n_runs} tirages de {tmax:.0f} s, {workers} en parallele"
        + (f", filtre={filter_type}" if filter_type else "")
        + (f", IMU={imu_preset}" if imu_preset else "")
        + (f", coupure GNSS {outage[0]}-{outage[1]} s" if outage else "")
    )
    while pending or running:
        while pending and len(running) < workers:
            seed = pending.pop(0)
            work = tempfile.mkdtemp(prefix=f"mc_s{seed}_")
            for d in (
                "Control",
                "entities",
                "environment",
                "simulator",
                "swarm",
                "utilities",
                "assets",
            ):
                shutil.copytree(
                    os.path.join(REPO, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__")
                )
            shutil.copy(os.path.join(REPO, "config.yaml"), work)
            with open(os.path.join(work, "_mc_runner.py"), "w") as f:
                f.write(_RUNNER_SRC)
            env = dict(
                os.environ,
                SEED=str(seed),
                TMAX=str(tmax),
                LOGDIR=os.path.abspath(os.path.join(out_dir, f"s{seed}")),
                PORT_OFFSET=str(20 * (seed % 200 + 1)),
                FILTER_TYPE=filter_type or "",
                AGENT_OVERRIDE=json.dumps(agent_override or {}),
                IMU_PRESET=json.dumps(preset),
                OUTAGE=json.dumps(list(outage) if outage else None),
                JAM=json.dumps(list(jam) if jam else None),
            )
            proc = subprocess.Popen(
                [sys.executable, "_mc_runner.py"],
                cwd=work,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            running.append((proc, work, seed))
        for item in list(running):
            proc, work, seed = item
            if proc.poll() is None:
                continue
            running.remove(item)
            done += 1
            if proc.returncode != 0:
                failures += 1
                err = proc.stderr.read().decode(errors="replace").strip().splitlines()
                print(f"\n  tirage {seed} en ECHEC : {err[-1] if err else '?'}")
            shutil.rmtree(work, ignore_errors=True)
            print(f"\r  {done}/{n_runs} termines", end="", flush=True)
        time.sleep(0.2)
    print(f"\n  {failures} echec(s) d'execution" if failures else "")
    return out_dir


def chi2_bounds(dim: int, n: int, p: float = 0.01) -> tuple[float, float]:
    """Bornes a p et 1-p de la moyenne de n variables chi2 a `dim` ddl."""
    n = max(int(n), 1)
    if not HAVE_SCIPY:
        s = np.sqrt(2.0 * dim / n)
        # p = 0.01 de chaque cote (intervalle a 98 %) -> z = 2.326
        return dim - 2.326 * s, dim + 2.326 * s
    return chi2.ppf(p, dim * n) / n, chi2.ppf(1 - p, dim * n) / n


def collect(log_root: str) -> dict[str, pd.DataFrame]:
    """Agrege les journaux de filtre par drone, tous tirages confondus."""
    per_drone: dict[str, list[pd.DataFrame]] = {}
    for seed_dir in sorted(glob.glob(os.path.join(log_root, "s*"))):
        m = re.search(r"s(\d+)$", seed_dir)
        if not m:
            continue
        seed = int(m.group(1))
        for fpath in sorted(glob.glob(os.path.join(seed_dir, "*_filter.csv"))):
            name = os.path.basename(fpath).replace("_filter.csv", "")
            fdf = pd.read_csv(fpath)
            if fdf.empty:
                continue
            main = os.path.join(seed_dir, f"{name}.csv")
            if os.path.exists(main):
                mdf = pd.read_csv(main)
                if not mdf.empty:
                    # Journal principal moins frequent : plus proche voisin en temps
                    fdf = pd.merge_asof(
                        fdf.sort_values("time"),
                        mdf[["time", "gnss_error_mag"]].sort_values("time"),
                        on="time",
                        direction="nearest",
                    )
            fdf["seed"] = seed
            per_drone.setdefault(name, []).append(fdf)
    return {k: pd.concat(v, ignore_index=True) for k, v in per_drone.items()}


def _norm(df, cols):
    return np.sqrt((df[cols].to_numpy() ** 2).sum(axis=1))


def report(per_drone: dict[str, pd.DataFrame], skip: float = 2.0) -> pd.DataFrame:
    """Tableau de validation. `skip` : secondes initiales ignorees (convergence)."""
    rows = []
    for name, df in sorted(per_drone.items()):
        d = df[df.time > skip]
        avail = d.gnss_available == 1 if "gnss_available" in d else np.ones(len(d), bool)
        e_pos = _norm(d, ["e_px", "e_py", "e_pz"])
        e_vel = _norm(d, ["e_vx", "e_vy", "e_vz"])
        avail = np.asarray(avail, dtype=bool)
        # Erreur GNSS a l'instant de la mesure
        if "gnss_err" in d and d.gnss_err.notna().any():
            gn = d.loc[d.gnss_update == 1, "gnss_err"].dropna()
            gnss_src = "instant de mesure"
        else:
            gn = d.loc[avail, "gnss_error_mag"] if "gnss_error_mag" in d else pd.Series(dtype=float)
            gnss_src = "journal principal (SURESTIME : mesure maintenue)"
        # Indicateurs sur les memes echantillons, hors coupure GNSS
        da = d[avail]
        with np.errstate(invalid="ignore"):
            out = np.abs(da[["e_px", "e_py", "e_pz"]].to_numpy()) > 3 * np.maximum(
                da[["sig_px", "sig_py", "sig_pz"]].to_numpy(), 1e-12
            )
        has_full = "nees_full" in d and d.nees_full.notna().any()
        rows.append(
            dict(
                drone=name,
                seeds=d.seed.nunique(),
                rmse_pos=float(np.sqrt(np.mean(e_pos[avail] ** 2))),
                rmse_vel=float(np.sqrt(np.mean(e_vel[avail] ** 2))),
                rmse_gnss=float(np.sqrt(np.mean(gn**2))) if len(gn) else np.nan,
                pct_out=100.0 * float(np.nanmean(out)),
                nees_pv=float(da.nees.mean()),
                nees_full=float(da.nees_full.mean()) if has_full else np.nan,
                nis=float(da.nis.dropna().mean()) if da.nis.notna().any() else np.nan,
                n=len(da),
                n_nis=int(da.nis.notna().sum()),
                gnss_src=gnss_src,
                att_rp=(
                    float(np.degrees(np.sqrt(np.nanmean(d.e_rx**2 + d.e_ry**2) / 2))) if has_full else np.nan
                ),
                att_yaw=float(np.degrees(np.sqrt(np.nanmean(d.e_rz**2)))) if has_full else np.nan,
            )
        )
    R = pd.DataFrame(rows)
    R["gain"] = R.rmse_gnss / R.rmse_pos

    print("\n" + "=" * 92)
    print("VALIDATION DU FILTRE DE NAVIGATION  (indicateurs hors coupure GNSS)")
    print("=" * 92)
    print(
        f"{'drone':9}{'RMSE pos':>10}{'RMSE vit':>11}{'RMSE GNSS':>11}{'gain':>7}"
        f"{'>3sig':>8}{'NEES pv':>9}{'NEES 15':>9}{'NIS':>7}{'att rp':>8}{'lacet':>7}"
    )
    print("-" * 92)
    for r in R.itertuples():
        f15 = f"{r.nees_full:9.2f}" if np.isfinite(r.nees_full) else f"{'-':>9}"
        arp = f"{r.att_rp:7.2f}d" if np.isfinite(r.att_rp) else f"{'-':>8}"
        ayw = f"{r.att_yaw:6.2f}d" if np.isfinite(r.att_yaw) else f"{'-':>7}"
        print(
            f"{r.drone:9}{r.rmse_pos:9.3f}m{r.rmse_vel:8.3f}m/s{r.rmse_gnss:10.3f}m"
            f"{r.gain:7.2f}{r.pct_out:7.2f}%{r.nees_pv:9.2f}{f15}{r.nis:7.2f}{arp}{ayw}"
        )

    print(f"\nReference GNSS : {R.gnss_src.iloc[0]}")
    print("Attendus : gain > 1 | >3sig ~0.3 % | NEES pv = 6 | NEES 15 = 15 | NIS = 6")
    print("Bornes chi2 a 98 % (optimistes, echantillons correles) :")
    n = int(R.n.mean()) if len(R) else 1
    for lbl, dim, nn in [
        ("NEES pv", PV_DIM, n),
        ("NEES 15", FULL_DIM, n),
        ("NIS", MEAS_DIM, int(R.n_nis.mean()) if len(R) else 1),
    ]:
        lo, hi = chi2_bounds(dim, nn)
        print(f"  {lbl:8}: [{lo:.2f}, {hi:.2f}]")

    print("\nVerdict (tolerance +/- 25 % autour de l'attendu, plus robuste que les bornes chi2) :")
    for r in R.itertuples():
        msg = []
        if np.isfinite(r.gain) and r.gain <= 1.0:
            msg.append("le filtre fait MOINS BIEN que le GNSS brut")
        for lbl, val, dim in [
            ("NEES pv", r.nees_pv, PV_DIM),
            ("NEES 15", r.nees_full, FULL_DIM),
            ("NIS", r.nis, MEAS_DIM),
        ]:
            if not np.isfinite(val):
                continue
            if val > 1.25 * dim:
                msg.append(f"{lbl}={val:.1f} > {dim} : trop confiant")
            elif val < 0.75 * dim:
                msg.append(f"{lbl}={val:.1f} < {dim} : trop prudent")
        if r.pct_out > 2.0:
            msg.append(f"{r.pct_out:.1f} % hors 3-sigma")
        print(f"  {r.drone:9} " + ("COHERENT" if not msg else "; ".join(msg)))
    return R


def figures(
    per_drone: dict[str, pd.DataFrame], out_dir: str, drone: str | None = None, outage=None
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = drone or sorted(per_drone)[0]
    df = per_drone[name]
    n_s = df.seed.nunique()
    ref = df[df.seed == df.seed.min()].sort_values("time")
    paths = []

    def shade(ax):
        if outage:
            ax.axvspan(outage[0], outage[1], color="0.85", zorder=0, label="coupure GNSS")

    # Erreur et enveloppe 3-sigma, un tirage
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, a in zip(axes, "xyz"):
        shade(ax)
        ax.plot(ref.time, ref[f"e_p{a}"], lw=0.9, color="#1f77b4", label="erreur (estime - vrai)")
        ax.plot(
            ref.time,
            3 * ref[f"sig_p{a}"],
            lw=1.0,
            color="#d62728",
            ls="--",
            label=r"$\pm 3\sigma$ annonce par le filtre",
        )
        ax.plot(ref.time, -3 * ref[f"sig_p{a}"], lw=1.0, color="#d62728", ls="--")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel(f"erreur {a} [m]")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{name} : erreur de position et enveloppe 3-sigma (un tirage)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("temps [s]")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig1_enveloppe_3sigma.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # NEES et NIS moyens sur tous les tirages
    g = df.groupby(df.time.round(2))
    has_full = "nees_full" in df and df.nees_full.notna().any()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    shade(axes[0])
    nees = g.nees.mean().dropna()
    axes[0].plot(nees.index, nees.values, lw=0.9, color="#1f77b4", label="NEES position-vitesse")
    axes[0].axhline(PV_DIM, color="#1f77b4", lw=1.2, ls=":", label=f"attendu {PV_DIM}")
    if has_full:
        nf = g.nees_full.mean().dropna()
        axes[0].plot(nf.index, nf.values, lw=0.9, color="#9467bd", label="NEES 15 etats")
        axes[0].axhline(FULL_DIM, color="#9467bd", lw=1.2, ls=":", label=f"attendu {FULL_DIM}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NEES moyen")
    axes[0].set_title(f"{name} : coherence du filtre, moyenne sur {n_s} tirages")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    shade(axes[1])
    nis = g.nis.mean().dropna()
    axes[1].plot(nis.index, nis.values, lw=0.9, color="#ff7f0e", label="NIS")
    axes[1].axhline(MEAS_DIM, color="#2ca02c", lw=1.2, ls=":", label=f"attendu {MEAS_DIM}")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("NIS moyen")
    axes[1].set_xlabel("temps [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig2_nees_nis.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    # Faisceau Monte-Carlo de l'erreur de position
    fig, ax = plt.subplots(figsize=(10, 5))
    shade(ax)
    err = _norm(df, ["e_px", "e_py", "e_pz"])
    tmp = pd.DataFrame({"time": df.time.round(2).to_numpy(), "err": err, "seed": df.seed.to_numpy()})
    for _, sub in tmp.groupby("seed"):
        ax.plot(sub.time, sub.err, lw=0.4, color="0.75")
    q = tmp.groupby("time").err.quantile([0.5, 0.95]).unstack()
    ax.plot(q.index, q[0.5], lw=1.6, color="#1f77b4", label="mediane")
    ax.plot(q.index, q[0.95], lw=1.4, color="#d62728", ls="--", label="quantile 95 %")
    ax.set_xlabel("temps [s]")
    ax.set_ylabel("erreur de position [m]")
    ax.set_title(f"{name} : norme de l'erreur de position, {n_s} tirages")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "fig3_faisceau_monte_carlo.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    paths.append(p3)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30, help="nombre de tirages")
    ap.add_argument("--tmax", type=float, default=20.0, help="duree simulee [s]")
    ap.add_argument("--logs", default="runs/mc", help="repertoire des journaux")
    ap.add_argument("--no-sim", action="store_true", help="reutiliser les journaux existants")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--filter", choices=["eskf", "kf6"], default=None, help="force le type de filtre pour tous les drones"
    )
    ap.add_argument(
        "--imu-preset",
        choices=["config_biais", "mems_nav"],
        default=None,
        help="remplace le modele d'erreur IMU du config.yaml",
    )
    ap.add_argument(
        "--outage", nargs=2, type=float, metavar=("DEBUT", "FIN"), help="coupure GNSS [s] (aucune mesure)"
    )
    ap.add_argument(
        "--jam", nargs="+", type=float, metavar="X", help="brouillage GNSS : DEBUT FIN [MULTIPLICATEUR]"
    )
    ap.add_argument(
        "--agent-override",
        default=None,
        help='surcharge JSON (fusion recursive) appliquee a chaque drone, ex. '
        '\'{"max_tilt_deg": 45}\' ; une cle "@drone_1" cible un seul drone',
    )
    ap.add_argument("--drone", default=None, help="drone des figures")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    if not args.no_sim:
        if os.path.isdir(args.logs):
            shutil.rmtree(args.logs)
        run_monte_carlo(
            args.runs,
            args.tmax,
            args.logs,
            filter_type=args.filter,
            imu_preset=args.imu_preset,
            outage=args.outage,
            jam=args.jam,
            workers=args.workers,
            agent_override=json.loads(args.agent_override) if args.agent_override else None,
        )

    per_drone = collect(args.logs)
    if not per_drone:
        sys.exit(f"aucun journal de filtre dans {args.logs}")
    report(per_drone)
    if not args.no_fig:
        for pth in figures(per_drone, args.logs, args.drone, outage=args.outage):
            print(f"figure : {pth}")


if __name__ == "__main__":
    main()
