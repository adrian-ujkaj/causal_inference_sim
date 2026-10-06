"""
Tests de l'ESKF 15 etats et des capteurs associes, hors simulation PyBullet.
Lancer depuis la racine du depot :  python -m pytest tests -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import quaternion as Q  # noqa: E402
from entities.sensor import IMUSensor, GNSSensor  # noqa: E402
from Control.ESKF import ESKF  # noqa: E402
from analysis.nav_replay import synthetic_truth, replay, DEFAULT_IMU  # noqa: E402

MEMS_NAV = dict(
    accel_noise_density=0.002,
    gyro_noise_density=0.00025,
    accel_bias_std=0.05,
    accel_bias_rw=0.001,
    gyro_bias_std=0.005,
    gyro_bias_rw=1e-5,
    gravity=9.81,
)


def _concat(results):
    import pandas as pd

    return pd.concat(results, ignore_index=True)


# ----------------------------------------------------------------------
# Capteurs
# ----------------------------------------------------------------------
def test_imu_gyro_reproduces_exact_rotation():
    """Le gyro doit delivrer l'increment angulaire qui reproduit la rotation vraie."""
    imu = IMUSensor({}, dt=0.0125)
    q0 = Q.from_euler(0.1, -0.2, 0.3)
    q1 = Q.mul(q0, Q.exp([0.02, -0.01, 0.03]))
    imu.measure([0, 0, 0], q0, dt=0.0125)
    _, gyr = imu.measure([0, 0, 0], q1, dt=0.0125)
    assert np.allclose(Q.to_rot(Q.mul(q0, Q.exp(gyr * 0.0125))), Q.to_rot(q1), atol=1e-12)


def test_gnss_outage_and_reported_accuracy():
    g = GNSSensor(
        dict(
            position_noise_std=0.1,
            velocity_noise_std=0.05,
            outage_start=5.0,
            outage_end=8.0,
            jam_start=10.0,
            jam_end=12.0,
            jam_multiplier=20.0,
            seed=1,
        )
    )
    assert g.measure([0, 0, 0], [0, 0, 0], t=6.0) == (None, None)
    p, v = g.measure([0, 0, 0], [0, 0, 0], t=9.0)
    assert p is not None and np.isclose(g.last_pos_std, 0.1)
    g.measure([0, 0, 0], [0, 0, 0], t=11.0)
    assert np.isclose(g.last_pos_std, 2.0) and np.isclose(g.last_vel_std, 1.0)


# ----------------------------------------------------------------------
# ESKF
# ----------------------------------------------------------------------
def test_alignment_from_accelerometer():
    f = ESKF(0.0125)
    q_true = Q.from_euler(0.2, -0.15, 0.0)
    f.align(Q.to_rot(q_true).T @ np.array([0, 0, 9.81]), yaw=0.0)
    assert np.allclose(Q.to_euler(f.q), [0.2, -0.15, 0.0], atol=1e-12)


def test_mechanization_is_exact_without_sensor_errors():
    """30 s de vol inertiel pur, capteurs parfaits : l'erreur doit rester nulle."""
    tr = synthetic_truth(30.0, kind="maneuver")
    df = replay(tr, ("eskf",), imu_cfg=dict(gravity=9.81), outages=[(0.0, 1e9)], seed=1)["eskf"]
    last = df.iloc[-1]
    assert np.linalg.norm([last.e_vx, last.e_vy, last.e_vz]) < 1e-9
    assert np.degrees(np.abs(df[["e_rx", "e_ry", "e_rz"]].to_numpy())).max() < 1e-9
    assert np.linalg.norm([last.e_px, last.e_py, last.e_pz]) < 1e-3


def test_eskf_is_statistically_consistent_with_biases():
    """NEES 15 etats, NEES position-vitesse et NIS dans leurs plages attendues."""
    tr = synthetic_truth(30.0, kind="maneuver")
    df = _concat([replay(tr, ("eskf",), imu_cfg=DEFAULT_IMU, seed=200 + s)["eskf"] for s in range(8)])
    late = df[df.time > 5.0]
    assert 12.5 < late.nees_full.mean() < 17.5  # attendu 15
    assert 5.0 < late.nees_pv.mean() < 7.2  # attendu 6
    assert 5.2 < late.nis.dropna().mean() < 6.8  # attendu 6


def test_eskf_handles_large_bias_that_breaks_kf6():
    """Cas drone_2 (offset accelerometre 0.5 m/s^2) : l'ESKF reste coherent."""
    tr = synthetic_truth(30.0, kind="maneuver")
    cfg = dict(DEFAULT_IMU, accel_noise_mean=0.5)
    res = [replay(tr, ("eskf", "kf6"), imu_cfg=cfg, seed=300 + s) for s in range(5)]
    eskf = _concat([r["eskf"] for r in res])
    kf6 = _concat([r["kf6"] for r in res])
    assert eskf[eskf.time > 5].nees_pv.mean() < 7.5
    assert kf6[kf6.time > 5].nees_pv.mean() > 20.0


def test_outage_drift_and_honest_uncertainty_with_realistic_mems():
    """Coupure GNSS de 10 s : l'ESKF derive moins et son enveloppe 3-sigma est vraie."""
    tr = synthetic_truth(45.0, kind="maneuver")
    T0, T1 = 25.0, 35.0
    drift = {"eskf": [], "kf6": []}
    cover = []
    for s in range(8):
        r = replay(tr, ("eskf", "kf6"), imu_cfg=MEMS_NAV, seed=400 + s, outages=[(T0, T1)])
        for n, df in r.items():
            w = df[(df.time >= T0) & (df.time <= T1)]
            e = np.sqrt(w.e_px**2 + w.e_py**2 + w.e_pz**2)
            drift[n].append(e.iloc[-1])
            if n == "eskf":
                sg = np.sqrt(w.sig_px**2 + w.sig_py**2 + w.sig_pz**2)
                cover.append(np.mean(e <= 3 * sg))
    assert np.median(drift["eskf"]) < 0.5 * np.median(drift["kf6"])
    assert np.mean(cover) > 0.97
