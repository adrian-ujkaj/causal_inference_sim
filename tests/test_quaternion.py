"""Tests du module de quaternions, verifies contre PyBullet (reference des conventions)."""

import numpy as np
import pybullet as p
from utilities import quaternion as Q

rng = np.random.default_rng(0)


def rand_q():
    return Q.normalize(rng.normal(size=4))


def same_rot(q1, q2, tol=1e-9):
    return np.allclose(Q.to_rot(q1), Q.to_rot(q2), atol=tol)


def test_to_rot_matches_pybullet():
    for _ in range(200):
        q = rand_q()
        R_pb = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        assert np.allclose(Q.to_rot(q), R_pb, atol=1e-9)


def test_mul_matches_pybullet():
    for _ in range(200):
        a, b = rand_q(), rand_q()
        _, q_pb = p.multiplyTransforms([0, 0, 0], a, [0, 0, 0], b)
        # multiplyTransforms renvoie un resultat en simple precision
        assert same_rot(Q.mul(a, b), q_pb, tol=1e-6)
        assert np.allclose(Q.to_rot(Q.mul(a, b)), Q.to_rot(a) @ Q.to_rot(b), atol=1e-9)


def test_euler_matches_pybullet():
    for _ in range(200):
        e = rng.uniform([-np.pi, -1.5, -np.pi], [np.pi, 1.5, np.pi])
        assert same_rot(Q.from_euler(*e), p.getQuaternionFromEuler(e))
        q = rand_q()
        assert np.allclose(Q.to_euler(q), p.getEulerFromQuaternion(q), atol=1e-7)


def test_exp_log_roundtrip():
    for _ in range(500):
        th = rng.normal(size=3) * rng.choice([1e-10, 1e-5, 0.1, 1.0, 3.0])
        if np.linalg.norm(th) >= np.pi:
            continue
        assert np.allclose(Q.log(Q.exp(th)), th, atol=1e-9)


def test_exp_is_rodrigues():
    for _ in range(100):
        th = rng.normal(size=3)
        a = np.linalg.norm(th)
        k = th / a
        K = Q.skew(k)
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        assert np.allclose(Q.rot_exp(th), R, atol=1e-9)


def test_kinematics_body_rate_right_composition():
    """q(k+1) = q(k) (x) Exp(w dt) doit integrer Rdot = R [w]x (vitesse corps)."""
    q = rand_q()
    R = Q.to_rot(q)
    w = np.array([0.3, -0.7, 1.1])
    dt = 1e-4
    for _ in range(1000):
        q = Q.mul(q, Q.exp(w * dt))
        R = R @ Q.rot_exp(w * dt)
    assert np.allclose(Q.to_rot(q), R, atol=1e-9)


def test_attitude_error_convention():
    """q_true = q_est (x) Exp(dtheta)  <=>  attitude_error(q_est, q_true) = dtheta."""
    for _ in range(200):
        q_est = rand_q()
        dth = rng.normal(size=3) * 0.2
        q_true = Q.mul(q_est, Q.exp(dth))
        assert np.allclose(Q.attitude_error(q_est, q_true), dth, atol=1e-9)


def test_first_order_rotation_perturbation():
    """R_true = R_est (I + [dtheta]x) au premier ordre (base des jacobiennes ESKF)."""
    q_est = rand_q()
    dth = np.array([1e-4, -2e-4, 3e-4])
    R_true = Q.to_rot(Q.mul(q_est, Q.exp(dth)))
    approx = Q.to_rot(q_est) @ (np.eye(3) + Q.skew(dth))
    assert np.allclose(R_true, approx, atol=1e-7)


def test_slerp_endpoints_and_midpoint():
    for _ in range(100):
        a, b = rand_q(), rand_q()
        assert same_rot(Q.slerp(a, b, 0.0), a)
        assert same_rot(Q.slerp(a, b, 1.0), b)
        m = Q.slerp(a, b, 0.5)
        # le milieu est a egale distance angulaire des deux bouts
        da = np.linalg.norm(Q.attitude_error(a, m))
        db = np.linalg.norm(Q.attitude_error(m, b))
        assert abs(da - db) < 1e-9


def test_canonical_sign():
    q = rand_q()
    assert np.allclose(Q.normalize(-q), Q.normalize(q))
