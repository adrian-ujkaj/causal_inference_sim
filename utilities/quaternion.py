"""
Outils de quaternions pour la navigation inertielle.

Conventions (identiques a PyBullet, a respecter partout dans le projet) :
  - stockage [x, y, z, w], partie scalaire EN DERNIER ;
  - produit de Hamilton ;
  - q represente la rotation CORPS -> MONDE : v_monde = R(q) @ v_corps ;
  - cinematique : q_dot = 0.5 * q (x) [omega_corps, 0], donc l'increment
    d'attitude se compose A DROITE : q(k+1) = q(k) (x) Exp(omega * dt).

Attention : beaucoup de references (dont Sola, "Quaternion kinematics for the
error-state Kalman filter") stockent [w, x, y, z]. Les formules sont les memes,
seul l'ordre de stockage change ; c'est la source d'erreur la plus frequente
quand on recopie une equation.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def skew(v) -> np.ndarray:
    """Matrice antisymetrique [v]x telle que [v]x @ u = v x u."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    # Representant canonique (w >= 0) : q et -q decrivent la meme rotation.
    return q if q[3] >= 0.0 else -q


def mul(q1, q2) -> np.ndarray:
    """Produit de Hamilton q1 (x) q2, stockage [x, y, z, w]."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=float).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def conj(q) -> np.ndarray:
    """Conjugue = inverse pour un quaternion unitaire."""
    x, y, z, w = np.asarray(q, dtype=float).reshape(4)
    return np.array([-x, -y, -z, w])


def exp(theta) -> np.ndarray:
    """
    Vecteur de rotation (axe * angle, rad) -> quaternion unitaire.
    Serie de Taylor pres de zero pour eviter la division par |theta|.
    """
    theta = np.asarray(theta, dtype=float).reshape(3)
    a = float(np.linalg.norm(theta))
    if a < 1e-8:
        # sin(a/2)/a ~ 1/2 - a^2/48
        q = np.array([*(0.5 * theta), 1.0 - a * a / 8.0])
        return q / np.linalg.norm(q)
    s = np.sin(0.5 * a) / a
    return np.array([*(s * theta), np.cos(0.5 * a)])


def log(q) -> np.ndarray:
    """Quaternion unitaire -> vecteur de rotation (rad), angle dans [0, pi]."""
    q = normalize(q)  # impose w >= 0 : plus court chemin
    v, w = q[:3], q[3]
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return 2.0 * v / max(w, _EPS)  # 2*v pour les petits angles
    angle = 2.0 * np.arctan2(n, w)
    return angle * v / n


def to_rot(q) -> np.ndarray:
    """Quaternion [x, y, z, w] -> matrice de rotation corps -> monde."""
    x, y, z, w = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_exp(theta) -> np.ndarray:
    """Formule de Rodrigues : vecteur de rotation -> matrice de rotation."""
    return to_rot(exp(theta))


def slerp(q0, q1, t: float) -> np.ndarray:
    """Interpolation spherique entre q0 et q1, t dans [0, 1]."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    d = mul(conj(q0), q1)  # rotation relative q0 -> q1
    return normalize(mul(q0, exp(t * log(d))))


def from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Angles d'Euler (convention PyBullet, rotations fixes X puis Y puis Z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return normalize(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def to_euler(q) -> np.ndarray:
    """Quaternion -> [roulis, tangage, lacet] (rad), meme convention."""
    R = to_rot(q)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def attitude_error(q_est, q_true) -> np.ndarray:
    """
    Erreur d'attitude LOCALE (repere corps) telle que
        q_true = q_est (x) Exp(dtheta).
    C'est la convention de l'ESKF : la composante d'erreur angulaire du NEES
    doit etre calculee ainsi, sinon le test de coherence est faux.
    """
    return log(mul(conj(q_est), q_true))
