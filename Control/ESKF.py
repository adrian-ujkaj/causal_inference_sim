"""
ESKF : filtre de Kalman a etat d'erreur, hybridation inertie / GNSS a 15 etats.

Pourquoi un filtre a etat d'erreur
----------------------------------
L'attitude vit sur SO(3), pas dans R^n : un quaternion unitaire ne supporte pas
une correction additive. On separe donc
  - un etat NOMINAL propage par les mesures inertielles (il porte le quaternion) ;
  - un etat d'ERREUR de dimension 15, petit, sur lequel on applique le Kalman,
    et dont la partie attitude est une rotation infinitesimale (3 composantes).
Apres chaque correction, l'erreur estimee est INJECTEE dans le nominal (par
composition de quaternions pour l'attitude) puis remise a zero.

Etats
-----
Nominal : p (3) position monde, v (3) vitesse monde,
          q (4) attitude corps -> monde [x, y, z, w],
          b_a (3) biais accelerometre corps, b_g (3) biais gyrometre corps.
Erreur  : dx = [dp | dv | dtheta | db_a | db_g]   (15)
Convention d'erreur LOCALE (repere corps) : q_vrai = q (x) Exp(dtheta).

Mecanisation (pas dt, mesures a_m, w_m)
---------------------------------------
    w      = w_m - b_g                  vitesse angulaire corrigee
    a_b    = a_m - b_a                  acceleration propre corrigee (corps)
    q_mid  = q (x) Exp(w dt / 2)        attitude a mi-intervalle
    a_w    = R(q_mid) a_b + g_vec       g_vec = [0, 0, -g]
    p     += v dt + a_w dt^2 / 2
    v     += a_w dt
    q      = q (x) Exp(w dt)
L'attitude a mi-intervalle est celle avec laquelle l'IMU simulee a projete son
increment de vitesse : la mecanisation est exacte en l'absence d'erreur capteur.

Dynamique de l'erreur (Sola 2017, erreur angulaire locale), discretisee
-----------------------------------------------------------------------
    dp'      = dp + dv dt - R[a_b]x dtheta dt^2/2 - R db_a dt^2/2
    dv'      = dv - R[a_b]x dtheta dt - R db_a dt + bruit
    dtheta'  = R{w dt}^T dtheta - db_g dt + bruit
    db_a'    = db_a + marche aleatoire
    db_g'    = db_g + marche aleatoire

Mesure GNSS
-----------
Position et vitesse : h(x) = [p; v], lineaire dans l'etat d'erreur,
H = [I3 0 0 0 0 ; 0 I3 0 0 0]. R peut etre fourni mesure par mesure (precision
rapportee par le recepteur), sinon il est derive de la configuration GNSS.

Observabilite, a garder en tete pour interpreter les resultats
--------------------------------------------------------------
- Roulis et tangage sont observables en permanence grace a la gravite : une
  erreur d'inclinaison projette une fraction de g sur l'horizontale, ce qui
  fait deriver la vitesse que le GNSS corrige.
- Le lacet n'est observable que pendant les accelerations HORIZONTALES. En vol
  stationnaire il derive au rythme du biais gyro z. Sans magnetometre, c'est
  une limite physique, pas un defaut du filtre.
- Biais accelerometre horizontal et erreur d'inclinaison se confondent en
  stationnaire (meme signature) ; c'est la manoeuvre qui les separe.

Reference : J. Sola, "Quaternion kinematics for the error-state Kalman
filter", 2017 (attention : stockage [w, x, y, z] chez Sola, [x, y, z, w] ici).
"""

from __future__ import annotations

import numpy as np

from utilities import quaternion as Q

_P = slice(0, 3)
_V = slice(3, 6)
_TH = slice(6, 9)
_BA = slice(9, 12)
_BG = slice(12, 15)


class ESKF:
    STATE_DIM = 15
    name = "eskf"

    def __init__(
        self, dt, gnss_config: dict | None = None, imu_config: dict | None = None, config: dict | None = None
    ):
        cfg = dict(config or {})
        imu = dict(imu_config or {})
        gnss = dict(gnss_config or {})

        self.dt = float(dt)
        self.g = float(cfg.get("gravity", imu.get("gravity", 9.81)))
        self.g_vec = np.array([0.0, 0.0, -self.g])

        # --- Etat nominal ---
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.aligned = False
        # Cap initial suppose connu (compas a l'initialisation), avec
        # l'incertitude init_yaw_std_deg. Utilise par l'alignement automatique.
        self.initial_yaw = float(cfg.get("initial_yaw", 0.0))

        # --- Bruits capteur, derives de la configuration IMU declaree ---
        self._a_std = float(imu.get("accel_noise_std", 0.0))
        self._a_den = imu.get("accel_noise_density", None)
        self._g_std = float(imu.get("gyro_noise_std", 0.0))
        self._g_den = imu.get("gyro_noise_density", None)
        # Planchers : un bruit nul rendrait P singuliere et le filtre sur-confiant.
        self._a_floor = float(cfg.get("accel_noise_floor", 0.02))
        self._g_floor = float(cfg.get("gyro_noise_floor", 0.002))
        # Marge pour ce que le modele ignore (facteurs d'echelle, integration).
        self._a_margin = float(cfg.get("accel_model_margin", 0.05))
        self._g_margin = float(cfg.get("gyro_model_margin", 0.0))

        # Marches aleatoires des biais. Un plancher non nul laisse le filtre
        # continuer a apprendre les biais ; sans lui, P_biais tend vers 0 et le
        # filtre cesse de les corriger.
        self.sig_ba_rw = max(
            float(imu.get("accel_bias_rw", 0.0)), float(cfg.get("accel_bias_rw_floor", 1e-3))
        )
        self.sig_bg_rw = max(float(imu.get("gyro_bias_rw", 0.0)), float(cfg.get("gyro_bias_rw_floor", 1e-4)))

        # --- Covariance initiale ---
        sp0 = float(cfg.get("init_pos_std", 0.5))
        sv0 = float(cfg.get("init_vel_std", 0.1))
        srp0 = np.radians(float(cfg.get("init_roll_pitch_std_deg", 3.0)))
        syaw0 = np.radians(float(cfg.get("init_yaw_std_deg", 5.0)))
        # Biais : le filtre doit savoir quel biais est plausible. On reprend ce
        # que declare le capteur (biais de mise sous tension ET offset constant,
        # cas de drone_2 avec accel_noise_mean = 0.5).
        sba0 = float(
            cfg.get(
                "init_accel_bias_std",
                np.sqrt(
                    float(imu.get("accel_bias_std", 0.0)) ** 2
                    + float(imu.get("accel_noise_mean", 0.0)) ** 2
                    + 0.05**2
                ),
            )
        )
        sbg0 = float(cfg.get("init_gyro_bias_std", np.hypot(float(imu.get("gyro_bias_std", 0.0)), 0.005)))
        self.P = np.diag(
            [sp0**2] * 3 + [sv0**2] * 3 + [srp0**2, srp0**2, syaw0**2] + [sba0**2] * 3 + [sbg0**2] * 3
        )

        # --- Bruit de mesure GNSS par defaut ---
        self.sigma_p = float(gnss.get("position_noise_std", 0.1)) or 0.1
        self.sigma_v = float(gnss.get("velocity_noise_std", 0.05)) or 0.05
        self.R = np.diag([self.sigma_p**2] * 3 + [self.sigma_v**2] * 3)

        # --- Rejet des mesures aberrantes, avec sortie de blocage ---
        self.nis_gate = cfg.get("nis_gate", 22.46)  # chi2(6), p = 0.001
        self.max_consecutive_rejected = int(cfg.get("max_consecutive_rejected", 5))
        self.n_consecutive_rejected = 0

        # --- Diagnostic ---
        self.last_innovation = np.zeros(6)
        self.last_S = np.zeros((6, 6))
        self.last_nis = np.nan
        self.n_updates = 0
        self.n_rejected = 0
        self.n_gate_recoveries = 0
        self.last_acc_world = np.zeros(3)

    # ------------------------------------------------------------------
    # Interface commune avec INSGNSSFilter
    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.p

    @property
    def velocity(self) -> np.ndarray:
        return self.v

    @property
    def attitude(self) -> np.ndarray:
        return self.q

    @property
    def x(self) -> np.ndarray:
        """Compatibilite avec le code qui lit ekf.x[:3] / ekf.x[3:6] (lecture seule)."""
        return np.concatenate([self.p, self.v])

    def init_state(self, position, velocity=None, attitude=None):
        self.p = np.asarray(position, dtype=float).reshape(3).copy()
        self.v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3).copy()
        if attitude is not None:
            self.q = Q.normalize(attitude)

    def sigmas(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    # ------------------------------------------------------------------
    def align(self, acc_body, yaw: float = 0.0) -> None:
        """
        Alignement grossier, a faire sur une mesure accelerometre au repos.

        Au repos l'accelerometre ne mesure que la gravite :
            f = R^T [0, 0, g] = g [-sin(tangage), cos(tangage) sin(roulis),
                                   cos(tangage) cos(roulis)]
        d'ou roulis et tangage. Le lacet n'est PAS observable par nivellement :
        il vient d'une autre source (ici la configuration, avec l'incertitude
        init_yaw_std_deg, comme le ferait un compas a l'initialisation).
        """
        f = np.asarray(acc_body, dtype=float).reshape(3)
        roll = np.arctan2(f[1], f[2])
        pitch = np.arctan2(-f[0], np.hypot(f[1], f[2]))
        self.q = Q.from_euler(roll, pitch, yaw)
        self.aligned = True

    # ------------------------------------------------------------------
    def _noise_std(self, std, density, floor, margin, dt) -> float:
        if density is not None and float(density) > 0.0 and dt > 0.0:
            s = float(density) / np.sqrt(dt)
        else:
            s = float(std)
        return float(np.sqrt(max(s, floor) ** 2 + margin**2))

    def predict(self, acc_body, gyro_body, dt: float | None = None) -> None:
        """Propagation par les mesures inertielles sur l'intervalle dt."""
        dt = self.dt if (dt is None or float(dt) <= 0.0) else float(dt)
        a_m = np.asarray(acc_body, dtype=float).reshape(3)
        w_m = np.asarray(gyro_body, dtype=float).reshape(3)

        if not self.aligned:
            # Premier echantillon : l'IMU simulee le delivre a vitesse constante
            # (pas d'increment de vitesse), ce qui est le cas du drone au sol.
            self.align(a_m, yaw=self.initial_yaw)

        w = w_m - self.bg
        a_b = a_m - self.ba

        # --- Etat nominal ---
        q_mid = Q.mul(self.q, Q.exp(0.5 * w * dt))
        R_mid = Q.to_rot(q_mid)
        a_w = R_mid @ a_b + self.g_vec
        self.last_acc_world = a_w

        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = Q.normalize(Q.mul(self.q, Q.exp(w * dt)))

        # --- Transition de l'etat d'erreur ---
        I3 = np.eye(3)
        RA = R_mid @ Q.skew(a_b)
        F = np.eye(15)
        F[_P, _V] = I3 * dt
        F[_P, _TH] = -0.5 * RA * dt * dt
        F[_P, _BA] = -0.5 * R_mid * dt * dt
        F[_V, _TH] = -RA * dt
        F[_V, _BA] = -R_mid * dt
        F[_TH, _TH] = Q.rot_exp(w * dt).T
        F[_TH, _BG] = -I3 * dt

        # --- Bruit de propagation ---
        sa = self._noise_std(self._a_std, self._a_den, self._a_floor, self._a_margin, dt)
        sg = self._noise_std(self._g_std, self._g_den, self._g_floor, self._g_margin, dt)
        qa = sa * sa
        Qd = np.zeros((15, 15))
        # bruit accelerometre : agit sur v (dt) et sur p (dt^2/2), correle
        Qd[_P, _P] = qa * (dt**4) / 4.0 * I3
        Qd[_P, _V] = qa * (dt**3) / 2.0 * I3
        Qd[_V, _P] = qa * (dt**3) / 2.0 * I3
        Qd[_V, _V] = qa * (dt**2) * I3
        Qd[_TH, _TH] = sg * sg * (dt**2) * I3
        Qd[_BA, _BA] = self.sig_ba_rw**2 * dt * I3
        Qd[_BG, _BG] = self.sig_bg_rw**2 * dt * I3

        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    def update(self, pos_meas, vel_meas=None, pos_std: float | None = None, vel_std: float | None = None):
        """
        Correction GNSS. vel_meas peut etre None (position seule).
        pos_std / vel_std : precision rapportee par le recepteur pour CETTE
        mesure ; a defaut, les valeurs de configuration.
        Retourne (position, vitesse).
        """
        use_vel = vel_meas is not None
        m = 6 if use_vel else 3
        sp = self.sigma_p if pos_std is None else max(float(pos_std), 1e-6)
        sv = self.sigma_v if vel_std is None else max(float(vel_std), 1e-6)

        H = np.zeros((m, 15))
        H[0:3, _P] = np.eye(3)
        if use_vel:
            H[3:6, _V] = np.eye(3)
            z = np.hstack([np.asarray(pos_meas, float), np.asarray(vel_meas, float)])
            h = np.hstack([self.p, self.v])
            R = np.diag([sp * sp] * 3 + [sv * sv] * 3)
        else:
            z = np.asarray(pos_meas, float).reshape(3)
            h = self.p.copy()
            R = np.diag([sp * sp] * 3)

        y = z - h
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)
        try:
            nis = float(y @ np.linalg.solve(S, y))
            K = np.linalg.solve(S, H @ self.P).T  # = P H^T S^-1
        except np.linalg.LinAlgError:
            print("[ESKF] S singuliere : mise a jour ignoree (symptome de divergence)")
            return self.p.copy(), self.v.copy()

        self.last_innovation = y
        self.last_S = S
        self.last_nis = nis
        self.n_updates += 1

        # --- Rejet des aberrations, borne (cf. INSGNSSFilter pour le detail) ---
        gate = self.nis_gate
        if gate and m != 6:
            gate = float(gate) * m / 6.0  # ajustement grossier du seuil
        if gate and nis > float(gate):
            self.n_consecutive_rejected += 1
            if self.n_consecutive_rejected <= self.max_consecutive_rejected:
                self.n_rejected += 1
                return self.p.copy(), self.v.copy()
            # Trop de rejets de suite : le filtre a tort, pas la mesure.
            self.n_gate_recoveries += 1
            self.P = self.P * 10.0
            S = H @ self.P @ H.T + R
            S = 0.5 * (S + S.T)
            K = np.linalg.solve(S, H @ self.P).T
        self.n_consecutive_rejected = 0

        # --- Correction de l'etat d'erreur et covariance (forme de Joseph) ---
        dx = K @ y
        I15 = np.eye(15)
        A = I15 - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

        # --- Injection dans l'etat nominal ---
        self.p = self.p + dx[_P]
        self.v = self.v + dx[_V]
        self.q = Q.normalize(Q.mul(self.q, Q.exp(dx[_TH])))
        self.ba = self.ba + dx[_BA]
        self.bg = self.bg + dx[_BG]

        # --- Remise a zero de l'erreur : jacobienne de reset sur l'attitude ---
        G = np.eye(15)
        G[_TH, _TH] = np.eye(3) - Q.skew(0.5 * dx[_TH])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.p.copy(), self.v.copy()

    # ------------------------------------------------------------------
    def error_vector(self, p_true, v_true, q_true=None, ba_true=None, bg_true=None):
        """Erreur 'vrai moins estime' dans la convention de l'etat d'erreur."""
        e = np.full(15, np.nan)
        e[_P] = np.asarray(p_true, float) - self.p
        e[_V] = np.asarray(v_true, float) - self.v
        if q_true is not None:
            e[_TH] = Q.attitude_error(self.q, q_true)
        if ba_true is not None:
            e[_BA] = np.asarray(ba_true, float) - self.ba
        if bg_true is not None:
            e[_BG] = np.asarray(bg_true, float) - self.bg
        return e

    def nees(self, p_true, v_true, q_true=None, ba_true=None, bg_true=None):
        """
        Renvoie (NEES position-vitesse, NEES complet).
        NEES position-vitesse : dimension 6, comparable au filtre 6 etats.
        NEES complet : dimension 15, NaN si une verite manque.
        """
        e = self.error_vector(p_true, v_true, q_true, ba_true, bg_true)
        try:
            e6 = e[0:6]
            nees_pv = float(e6 @ np.linalg.solve(self.P[0:6, 0:6], e6))
        except np.linalg.LinAlgError:
            nees_pv = float("nan")
        if np.any(np.isnan(e)):
            return nees_pv, float("nan")
        try:
            nees_full = float(e @ np.linalg.solve(self.P, e))
        except np.linalg.LinAlgError:
            nees_full = float("nan")
        return nees_pv, nees_full
