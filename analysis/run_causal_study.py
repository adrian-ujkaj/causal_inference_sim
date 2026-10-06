"""
Etude causale complete, reproductible (Windows / Linux) :

  1. trois campagnes de vols APPARIES (memes graines 1..N) :
       base      : configuration nominale ;
       gnss_fort : GNSS de drone_1 seul fortement brouille entre 8 et 22 s (bruit x100) ;
       vent_fort : turbulence tres forte (80 au lieu de 10) sur drone_2 seul,
                   entre 8 et 22 s ;
  2. analyse causale observationnelle de chaque campagne ;
  3. validation interventionnelle : effet mesure de chaque intervention sur
     chaque drone, confronte a ce que l'analyse observationnelle a conclu.

Les 3 figures a regarder sont copiees dans runs/causal/RESULTATS/ :
  1_verification.png        ce qui s'est vraiment passe / ce que l'analyse en deduit
  2_qui_influence_qui.png   graphe des interactions appris par le NRI (vols normaux)
  3_causes_vol_normal.png   causes des defaillances en vol normal

  python analysis/run_causal_study.py                 # 24 vols x 30 s par campagne
  python analysis/run_causal_study.py --runs 8 --skip_sim   # analyses seules
Duree indicative : 10 a 40 min par campagne selon la machine, ~5 min par analyse.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

CAMPAIGNS = {
    "base": None,
    "gnss_fort": {"@drone_1": {"sensors": {"gnss": {"jam_start": 8, "jam_end": 22, "jam_multiplier": 100}}}},
    "vent_fort": {"@drone_2": {"wind": {"burst_start": 8, "burst_end": 22, "burst_turbulence": 80}}},
}
WINDOWS = {"gnss_fort": "8:22", "vent_fort": "8:22"}


def run(cmd, log):
    print(">", " ".join(cmd))
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"echec ({r.returncode}), voir {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join("runs", "causal"))
    p.add_argument("--runs", type=int, default=24)
    p.add_argument("--tmax", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip_sim", action="store_true", help="reutiliser les vols deja simules")
    a = p.parse_args()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    if not a.skip_sim:
        for name, ov in CAMPAIGNS.items():
            cmd = [
                PY,
                "analysis/filter_validation.py",
                "--runs",
                str(a.runs),
                "--tmax",
                str(a.tmax),
                "--workers",
                str(a.workers),
                "--logs",
                os.path.join(a.out, name),
                "--no-fig",
            ]
            if ov:
                cmd += ["--agent-override", json.dumps(ov)]
            run(cmd, os.path.join(out, f"{name}.log"))

    ana = os.path.join(a.out, "analyse")
    for name in CAMPAIGNS:
        run(
            [
                PY,
                "analysis/causal_analysis.py",
                "--log_dir",
                os.path.join(a.out, name),
                "--output_dir",
                os.path.join(ana, name),
                "--verbose_every",
                "0",
            ],
            os.path.join(out, f"analyse_{name}.log"),
        )

    cmd = [
        PY,
        "analysis/causal_validation.py",
        "--base",
        os.path.join(a.out, "base"),
        "--output_dir",
        os.path.join(a.out, "validation"),
    ]
    for name, w in WINDOWS.items():
        cmd += [
            "--intervention",
            f"{name}={os.path.join(a.out, name)}" + (f":{w}" if w else ""),
            "--analysis",
            f"{name}={os.path.join(ana, name)}",
        ]
    run(cmd, os.path.join(out, "validation.log"))
    print(open(os.path.join(out, "validation.log")).read())

    res = os.path.join(out, "RESULTATS")
    os.makedirs(res, exist_ok=True)
    for src, dst in (
        (os.path.join(out, "validation", "resume_etude.png"), "1_verification.png"),
        (os.path.join(ROOT, ana, "base", "graphe_interactions.png"), "2_qui_influence_qui.png"),
        (os.path.join(ROOT, ana, "base", "causes_defaillances.png"), "3_causes_vol_normal.png"),
    ):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(res, dst))
    print(f"Figures a regarder : {res}")


if __name__ == "__main__":
    main()
