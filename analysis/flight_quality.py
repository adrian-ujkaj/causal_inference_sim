"""
Qualite de vol sur un ensemble de tirages : decollage rate (altitude max < 50 % de la cible),
retournement (inclinaison vraie > 90 deg), chute (au sol en contact apres >= 80 % de la cible)
et contact en vol (z > 0.3 m, avec un batiment ou un drone).
"""

import glob
import os
import sys
import pandas as pd

EXPECT = {'drone_0': 1.0, 'drone_1': 1.0, 'drone_2': 1.0, 'drone_3': 10.0}


def analyse(root):
    rows = []
    for sd in sorted(glob.glob(os.path.join(root, 's*'))):
        for f in sorted(glob.glob(os.path.join(sd, 'drone_*.csv'))):
            if f.endswith(('_filter.csv', '_truth.csv')):
                continue
            n = os.path.basename(f)[:-4]
            m = pd.read_csv(f)
            if m.empty:
                rows.append(dict(seed=sd, drone=n, takeoff=1, flip=0, crash=0, wall=0))
                continue
            exp = EXPECT.get(n, 1.0)
            flip = 0
            tf = os.path.join(sd, f'{n}_truth.csv')
            if os.path.exists(tf):
                t = pd.read_csv(tf)
                flip = int(((1 - 2 * (t.qx**2 + t.qy**2)) < 0).any())
            zmax = m.gt_z.max()
            after = m.loc[m.gt_z.idxmax() :]
            crash = int(
                zmax >= 0.8 * exp and after.gt_z.iloc[-1] < 0.15 and after.collision_flag.iloc[-1] == 1
            )
            rows.append(
                dict(
                    seed=sd,
                    drone=n,
                    takeoff=int(zmax < 0.5 * exp),
                    flip=flip,
                    crash=crash,
                    wall=int((m[m.gt_z > 0.3].collision_flag == 1).any()),
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for root in sys.argv[1:]:
        R = analyse(root)
        if R.empty:
            print(f"{root}: vide")
            continue
        bad = R[(R.takeoff | R.flip | R.crash) == 1]
        print(
            f"{os.path.basename(root):18} vols {len(R):3d} | decollage rate {R.takeoff.sum():2d} | "
            f"retournements {R.flip.sum():2d} | chutes {R.crash.sum():2d} | contact en vol {R.wall.sum():2d} "
            f"| VOLS REUSSIS {len(R) - len(bad)}/{len(R)}"
        )
