"""
Banc de rejeu de navigation, hors boucle de controle.

Principe : on part d'une trajectoire VRAIE (enregistree pendant une simulation
PyBullet, ou synthetique), on regenere les mesures capteurs avec les memes
classes que la simulation (IMUSensor, GNSSensor), puis on fait tourner un ou
plusieurs filtres sur EXACTEMENT les memes mesures.

Pourquoi hors boucle :
  - comparaison equitable : en boucle fermee, le drone vole sur l'estimee du
    filtre, donc deux filtres differents produisent deux trajectoires
    differentes et la comparaison melange estimation et controle ;
  - Monte-Carlo rapide : pas de physique a recalculer, des centaines de
    tirages de bruit sur la meme trajectoire en quelques secondes ;
  - scenarios maitrises : on place une coupure GNSS exactement ou on veut.

C'est la pratique courante en navigation : on valide un filtre en rejeu sur
trajectoire de reference avant de le mettre dans la boucle.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utilities import quaternion as Q  # noqa: E402
from entities.sensor import IMUSensor, GNSSensor  # noqa: E402
from Control.EKF import INSGNSSFilter  # noqa: E402
from Control.ESKF import ESKF  # noqa: E402

# Scenarios capteurs.
#  config_biais : bruits du config.yaml (accel 0.3 m/s^2, gyro 0.03 rad/s par
#                 echantillon a 80 Hz, soit 3400 ug/rtHz et 11.5 deg/rt h : un
#                 gyrometre environ 40 fois plus bruite qu'un MEMS grand public)
#                 auxquels on ajoute des biais de mise sous tension realistes.
#  mems_nav     : MEMS de navigation, ordres de grandeur d'une fiche technique
#                 type Bosch BMI088 (200 ug/rtHz, 0.9 deg/rt h).
DEFAULT_IMU = dict(
    accel_noise_std=0.3,
    gyro_noise_std=0.03,
    accel_bias_std=0.05,
    accel_bias_rw=0.01,
    gyro_bias_std=0.002,
    gyro_bias_rw=0.0005,
    gravity=9.81,
)
MEMS_NAV = dict(
    accel_noise_density=0.002,
    gyro_noise_density=0.00025,
    accel_bias_std=0.05,
    accel_bias_rw=0.001,
    gyro_bias_std=0.005,
    gyro_bias_rw=1e-5,
    gravity=9.81,
)
IMU_PRESETS = {"config_biais": DEFAULT_IMU, "mems_nav": MEMS_NAV}
DEFAULT_GNSS = dict(position_noise_std=0.1, velocity_noise_std=0.05)


# ----------------------------------------------------------------------
# Trajectoires
# ----------------------------------------------------------------------
def synthetic_truth(duration: float = 60.0, dt: float = 1 / 80, kind: str = "maneuver"):
    """
    Trajectoire vraie analytique.
      kind="maneuver" : boucles horizontales + variations d'altitude et
                        d'attitude (lacet observable).
      kind="hover"    : quasi stationnaire (lacet peu observable).
    Retourne un dict de tableaux : t, p, v, q.
    """
    t = np.arange(0.0, duration + 1e-9, dt)
    if kind == "hover":
        A, B, C, w = 0.2, 0.2, 0.1, 0.3
    else:
        A, B, C, w = 8.0, 6.0, 1.5, 0.25
    # Position et vitesse analytiques (rampe douce au depart pour partir au repos)
    ramp = 1.0 - np.exp(-t / 2.0)
    dramp = np.exp(-t / 2.0) / 2.0
    px = A * np.sin(w * t) * ramp
    py = B * (1 - np.cos(w * t)) * ramp
    pz = 1.0 + C * np.sin(2 * w * t) * ramp
    vx = A * (w * np.cos(w * t) * ramp + np.sin(w * t) * dramp)
    vy = B * (w * np.sin(w * t) * ramp + (1 - np.cos(w * t)) * dramp)
    vz = C * (2 * w * np.cos(2 * w * t) * ramp + np.sin(2 * w * t) * dramp)
    # Attitude : inclinaison coherente avec l'acceleration + lacet qui tourne
    roll = 0.15 * np.sin(0.7 * t) * ramp
    pitch = 0.12 * np.cos(0.5 * t) * ramp
    yaw = (0.6 * np.sin(0.2 * t) if kind != "hover" else 0.05 * np.sin(0.1 * t)) * ramp
    q = np.array([Q.from_euler(r, pp, y) for r, pp, y in zip(roll, pitch, yaw)])
    return dict(t=t, p=np.c_[px, py, pz], v=np.c_[vx, vy, vz], q=q)


def load_truth(csv_path: str):
    """Charge un journal de verite enregistre par la simulation (<drone>_truth.csv)."""
    df = pd.read_csv(csv_path)
    return dict(
        t=df.time.to_numpy(),
        p=df[["px", "py", "pz"]].to_numpy(),
        v=df[["vx", "vy", "vz"]].to_numpy(),
        q=df[["qx", "qy", "qz", "qw"]].to_numpy(),
    )


# ----------------------------------------------------------------------
# Rejeu
# ----------------------------------------------------------------------
def make_filter(kind: str, dt: float, imu_cfg: dict, gnss_cfg: dict, filter_cfg: dict | None = None):
    kind = kind.lower()
    if kind in ("kf6", "kf", "ekf", "insgnss"):
        return INSGNSSFilter(dt, gnss_config=gnss_cfg, imu_config=imu_cfg, config=filter_cfg)
    if kind == "eskf":
        return ESKF(dt, gnss_config=gnss_cfg, imu_config=imu_cfg, config=filter_cfg)
    raise ValueError(f"filtre inconnu : {kind}")


def replay(
    truth: dict,
    filters=("kf6", "eskf"),
    imu_cfg: dict | None = None,
    gnss_cfg: dict | None = None,
    filter_cfg: dict | None = None,
    seed: int = 0,
    outages=None,
    gnss_rate: float = 10.0,
    gnss_jitter: tuple[float, float] = (0.0, 0.0),
    every: int = 1,
):
    """
    Fait tourner les filtres demandes sur une trajectoire vraie.

    Retourne {nom_filtre: DataFrame} avec, a chaque pas enregistre :
      time, e_px..e_pz, e_vx..e_vz (estime - vrai), sig_px..sig_vz,
      e_rx..e_rz (erreur d'attitude, ESKF), e_bax.., e_bgx.. (biais, ESKF),
      nees_pv, nees_full, nis, gnss_update, gnss_available.
    """
    imu_cfg = dict(DEFAULT_IMU if imu_cfg is None else imu_cfg)
    gnss_cfg = dict(DEFAULT_GNSS if gnss_cfg is None else gnss_cfg)
    t, P, V, QQ = truth["t"], truth["p"], truth["v"], truth["q"]
    dt0 = float(np.median(np.diff(t)))

    imu = IMUSensor({**imu_cfg, "seed": int(seed)}, dt=dt0)
    gnss = GNSSensor({**gnss_cfg, "seed": int(seed) + 7919, "outages": [list(w) for w in (outages or [])]})
    rng = np.random.default_rng(int(seed) + 104729)

    yaw0 = float(Q.to_euler(QQ[0])[2])
    fl = {}
    for name in filters:
        fcfg = dict(filter_cfg or {})
        fcfg.setdefault("initial_yaw", yaw0)
        f = make_filter(name, dt0, imu_cfg, gnss_cfg, fcfg)
        if hasattr(f, "init_state"):
            f.init_state(P[0], V[0])
        fl[name] = f
    rows = {name: [] for name in fl}

    gnss_dt = 1.0 / gnss_rate
    nominal = t[0]
    next_trigger = t[0]
    for k in range(len(t)):
        dt = dt0 if k == 0 else float(t[k] - t[k - 1])
        acc, gyr = imu.measure(V[k], QQ[k], dt=dt)
        for f in fl.values():
            if isinstance(f, ESKF):
                f.predict(acc, gyr, dt=dt)
            else:
                f.predict(acc, imu.q_mid, dt=dt)

        updated = False
        if t[k] >= next_trigger:
            meas_p, meas_v = gnss.measure(P[k], V[k], t=t[k])
            if meas_p is not None:
                for f in fl.values():
                    f.update(meas_p, meas_v, pos_std=gnss.last_pos_std, vel_std=gnss.last_vel_std)
                updated = True
            nominal += gnss_dt  # echeancier strictement periodique (cf. uav.py)
            jit = max(0.0, rng.normal(*gnss_jitter)) if gnss_jitter[1] > 0 else gnss_jitter[0]
            next_trigger = nominal + jit

        if k % every and not updated:  # ne jamais perdre une ligne de mise a jour
            continue
        ba_true = imu.accel_error_det
        bg_true = imu.gyro_error_det
        for name, f in fl.items():
            sig = f.sigmas() if hasattr(f, "sigmas") else np.sqrt(np.clip(np.diag(f.P), 0, None))
            row = dict(time=t[k], gnss_update=int(updated), gnss_available=int(gnss.is_available(t[k])))
            # Convention des journaux : erreur = ESTIME - VRAI (error_vector
            # renvoie vrai - estime, convention interne de l'ESKF).
            if isinstance(f, ESKF):
                e = -f.error_vector(P[k], V[k], QQ[k], ba_true, bg_true)
                nees_pv, nees_full = f.nees(P[k], V[k], QQ[k], ba_true, bg_true)
                for i, a in enumerate("xyz"):
                    row[f"e_r{a}"] = e[6 + i]
                    row[f"e_ba{a}"] = e[9 + i]
                    row[f"e_bg{a}"] = e[12 + i]
                    row[f"sig_r{a}"] = sig[6 + i]
                    row[f"sig_ba{a}"] = sig[9 + i]
                    row[f"sig_bg{a}"] = sig[12 + i]
                    row[f"ba{a}"] = f.ba[i]
                    row[f"ba_true{a}"] = ba_true[i]
            else:
                e = np.r_[f.x[:3] - P[k], f.x[3:6] - V[k]]
                nees_pv = f.nees(P[k], V[k])
                nees_full = np.nan
            for i, a in enumerate("xyz"):
                row[f"e_p{a}"] = e[i]
                row[f"e_v{a}"] = e[3 + i]
                row[f"sig_p{a}"] = sig[i]
                row[f"sig_v{a}"] = sig[3 + i]
            row["nees_pv"] = nees_pv
            row["nees_full"] = nees_full
            row["nis"] = f.last_nis if updated else np.nan
            rows[name].append(row)

    return {name: pd.DataFrame(r) for name, r in rows.items()}
