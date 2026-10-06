import numpy as np


class INSGNSSFilter:
    """Filtre de Kalman lineaire (pas un EKF) inertie / GNSS, x = [px, py, pz, vx, vy, vz] en repere monde.
    L'accelerometre sert d'entree ; attitude fournie de l'exterieur, biais capteur non estimes.
    """

    def __init__(
        self, dt, gnss_config: dict | None = None, imu_config: dict | None = None, config: dict | None = None
    ):
        self.dt = float(dt)
        cfg = dict(config or {})

        self.x = np.zeros(6)

        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        p0_pos = float(cfg.get("init_pos_std", 0.5))
        p0_vel = float(cfg.get("init_vel_std", 0.5))
        self.P = np.diag([p0_pos**2] * 3 + [p0_vel**2] * 3)

        imu = dict(imu_config or {})
        sigma_a = float(cfg.get("accel_process_std", imu.get("accel_noise_std", 0.3)))
        # Marge pour biais, erreur d'attitude, trainee et vent, sinon le filtre est trop confiant
        sigma_a = float(np.hypot(sigma_a, float(cfg.get("accel_model_margin", 0.5))))
        self.sigma_a = sigma_a
        self.Q = self._build_Q(self.dt, sigma_a)

        gnss = dict(gnss_config or {})
        sigma_p = float(gnss.get("position_noise_std", 0.1)) or 0.1
        sigma_v = float(gnss.get("velocity_noise_std", 0.05)) or 0.05
        self.sigma_p, self.sigma_v = sigma_p, sigma_v
        self.H = np.eye(6)
        self.R = np.diag([sigma_p**2] * 3 + [sigma_v**2] * 3)

        self.GRAVITY = np.array([0.0, 0.0, float(cfg.get("gravity", 9.81))])

        self.last_innovation = np.zeros(6)
        self.last_S = np.zeros((6, 6))
        self.last_nis = np.nan
        self.n_updates = 0
        self.n_rejected = 0
        self.n_consecutive_rejected = 0
        self.n_gate_recoveries = 0
        # chi2 a 6 ddl, p = 0.001 ; 0 ou None desactive le test
        self.nis_gate = cfg.get("nis_gate", 22.46)
        # Rejets consecutifs bornes, sinon un filtre biaise rejette tout et diverge en inertie pure
        self.max_consecutive_rejected = int(cfg.get("max_consecutive_rejected", 5))

    # Interface commune avec l'ESKF
    name = "kf6"
    STATE_DIM = 6

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def attitude(self):
        return None  # attitude non estimee

    def init_state(self, position, velocity=None, attitude=None):
        self.x[:3] = np.asarray(position, dtype=float).reshape(3)
        self.x[3:6] = 0.0 if velocity is None else np.asarray(velocity, dtype=float).reshape(3)

    def sigmas(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    @staticmethod
    def _build_Q(dt, sigma_a):
        """Bruit de modele discret pour une acceleration blanche d'ecart-type sigma_a."""
        I3 = np.eye(3)
        q = sigma_a**2
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = q * (dt**4) / 4.0 * I3
        Q[0:3, 3:6] = q * (dt**3) / 2.0 * I3
        Q[3:6, 0:3] = q * (dt**3) / 2.0 * I3
        Q[3:6, 3:6] = q * (dt**2) * I3
        return Q

    @staticmethod
    def _quat_to_rot_matrix(q):
        """Quaternion [x, y, z, w] -> matrice de rotation corps vers monde."""
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def predict(self, imu_acc_body, orientation_quat, dt: float | None = None):
        """Propagation ; acceleration propre en repere corps [m/s^2], attitude [x, y, z, w]."""
        step = self.dt if (dt is None or float(dt) <= 0.0) else float(dt)

        # Gravite retiree en repere monde : une erreur d'attitude y laisse une fraction de g
        R = self._quat_to_rot_matrix(orientation_quat)
        acc_linear = (R @ np.asarray(imu_acc_body, dtype=float)) - self.GRAVITY

        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = step

        self.x = F @ self.x
        self.x[0:3] += 0.5 * acc_linear * (step**2)
        self.x[3:6] += acc_linear * step

        self.P = F @ self.P @ F.T + self._build_Q(step, self.sigma_a)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, pos_meas, vel_meas, pos_std: float | None = None, vel_std: float | None = None):
        """Correction GNSS, pos_std / vel_std donnes par le recepteur (sinon R de config). Renvoie (p, v)."""
        z = np.hstack([np.asarray(pos_meas, float), np.asarray(vel_meas, float)])
        y = z - (self.H @ self.x)
        if pos_std is None and vel_std is None:
            R = self.R
        else:
            sp = self.sigma_p if pos_std is None else max(float(pos_std), 1e-6)
            sv = self.sigma_v if vel_std is None else max(float(vel_std), 1e-6)
            R = np.diag([sp**2] * 3 + [sv**2] * 3)
        S = self.H @ self.P @ self.H.T + R
        S = 0.5 * (S + S.T)

        try:
            Sinv_y = np.linalg.solve(S, y)
            nis = float(y @ Sinv_y)
            K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError:
            print("[INSGNSSFilter] S singuliere : mise a jour ignoree (symptome de divergence du filtre)")
            return self.x[:3].copy(), self.x[3:6].copy()

        self.last_innovation = y
        self.last_S = S
        self.last_nis = nis
        self.n_updates += 1

        if self.nis_gate and nis > float(self.nis_gate):
            self.n_consecutive_rejected += 1
            if self.n_consecutive_rejected <= self.max_consecutive_rejected:
                self.n_rejected += 1
                return self.x[:3].copy(), self.x[3:6].copy()
            # Trop de rejets de suite : le filtre a tort, on gonfle P et on se recale sur la mesure
            self.n_gate_recoveries += 1
            self.P = self.P * 10.0
            S = self.H @ self.P @ self.H.T + R
            S = 0.5 * (S + S.T)
            K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        self.n_consecutive_rejected = 0

        self.x = self.x + (K @ y)

        # Forme de Joseph, garde P symetrique semi-definie positive
        I = np.eye(6)
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return self.x[:3].copy(), self.x[3:6].copy()

    def nees(self, true_pos, true_vel):
        """NEES = e^T P^-1 e, de moyenne 6 pour un filtre coherent."""
        e = self.x - np.hstack([np.asarray(true_pos, float), np.asarray(true_vel, float)])
        try:
            return float(e @ np.linalg.solve(self.P, e))
        except np.linalg.LinAlgError:
            return float('nan')
