import numpy as np

from utilities import quaternion as Q


class Sensor:
    """Classe de base pour les capteurs."""

    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée.")


class GNSSensor(Sensor):
    """GNSS simule : position [m] et vitesse [m/s] bruitees, brouillage (jam_*) et coupures (outages).
    last_pos_std / last_vel_std donnent l'ecart-type applique, comme hAcc / sAcc d'un recepteur u-blox.
    """

    def __init__(self, config: dict):
        super().__init__()
        config = dict(config or {})

        self.pos_noise_std = max(0.0, float(config.get("position_noise_std", 0.0)))
        self.vel_noise_std = max(0.0, float(config.get("velocity_noise_std", 0.0)))
        self.pos_noise_std_base = self.pos_noise_std
        self.vel_noise_std_base = self.vel_noise_std

        # Brouillage
        self.jam_start = config.get("jam_start", None)
        self.jam_end = config.get("jam_end", None)
        self.jam_pos_noise_std = config.get("jam_position_noise_std", config.get("jam_pos_noise_std", None))
        self.jam_vel_noise_std = config.get("jam_velocity_noise_std", config.get("jam_vel_noise_std", None))
        self.jam_multiplier = float(config.get("jam_multiplier", 1.0))

        # Coupures
        windows = list(config.get("outages", []) or [])
        if config.get("outage_start") is not None and config.get("outage_end") is not None:
            windows.append([config["outage_start"], config["outage_end"]])
        self.outages = [(float(a), float(b)) for a, b in windows]

        seed = config.get("seed", None)
        self._rng = None if seed is None else np.random.default_rng(int(seed))

        self.last_pos_std = self.pos_noise_std
        self.last_vel_std = self.vel_noise_std
        self.available = True

    def is_available(self, t: float | None) -> bool:
        """False pendant une fenetre de coupure."""
        if t is None:
            return True
        return not any(a <= float(t) <= b for a, b in self.outages)

    def _normal(self, std: float) -> np.ndarray:
        if self._rng is not None:
            return self._rng.normal(0.0, std, 3)
        return np.random.normal(0.0, std, 3)

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray,
        t: float | None = None,
    ):
        """Renvoie (position, vitesse) bruitees, ou (None, None) pendant une coupure."""
        self.available = self.is_available(t)
        if not self.available:
            return None, None

        pos_std = self.pos_noise_std
        vel_std = self.vel_noise_std

        if (t is not None) and (self.jam_start is not None) and (self.jam_end is not None):
            if float(self.jam_start) <= float(t) <= float(self.jam_end):
                if self.jam_pos_noise_std is not None:
                    pos_std = float(self.jam_pos_noise_std)
                else:
                    pos_std = float(self.pos_noise_std_base) * float(self.jam_multiplier)
                if self.jam_vel_noise_std is not None:
                    vel_std = float(self.jam_vel_noise_std)
                else:
                    vel_std = float(self.vel_noise_std_base) * float(self.jam_multiplier)

        self.last_pos_std = pos_std
        self.last_vel_std = vel_std

        meas_pos = np.asarray(ground_truth_position, dtype=float) + self._normal(pos_std)
        meas_vel = np.asarray(ground_truth_velocity, dtype=float) + self._normal(vel_std)
        return meas_pos, meas_vel


class IMUSensor(Sensor):
    """IMU MEMS simulee, en repere corps : mesure = (1 + s) * vrai + biais (constant + marche aleatoire) + bruit.
    Valeurs moyennes sur [k-1, k] : gyro = Log(q(k-1)^-1 (x) q(k)) / dt, accel projete a mi-intervalle.
    accel_error_det, gyro_error_det et q_mid donnent la verite terrain des erreurs pour le NEES.
    """

    def __init__(self, config: dict | None = None, dt: float | None = None):
        super().__init__()
        cfg = dict(config or {})

        dt_cfg = cfg.get("dt", dt)
        if dt_cfg is None or float(dt_cfg) <= 0.0:
            dt_cfg = 1.0 / 100.0
        self.dt_nominal = float(dt_cfg)

        g = float(cfg.get("gravity", 9.81))
        self.g_vector = np.array([0.0, 0.0, g], dtype=float)

        # Generateur dedie, tirages reproductibles en Monte-Carlo
        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))

        # Accelerometre : bruit [m/s^2], densite [m/s^2/sqrt(Hz)], marche aleatoire [m/s^2/sqrt(s)]
        self.accel_noise_std = max(0.0, float(cfg.get("accel_noise_std", 0.0)))
        self.accel_noise_density = cfg.get("accel_noise_density", None)
        self.accel_noise_mean = float(cfg.get("accel_noise_mean", 0.0))
        self.accel_bias_rw = max(0.0, float(cfg.get("accel_bias_rw", 0.0)))
        accel_bias_std = max(0.0, float(cfg.get("accel_bias_std", 0.0)))
        accel_scale_std = max(0.0, float(cfg.get("accel_scale_std", 0.0)))

        # Gyrometre : memes grandeurs en rad/s
        self.gyro_noise_std = max(0.0, float(cfg.get("gyro_noise_std", 0.0)))
        self.gyro_noise_density = cfg.get("gyro_noise_density", None)
        self.gyro_bias_rw = max(0.0, float(cfg.get("gyro_bias_rw", 0.0)))
        gyro_bias_std = max(0.0, float(cfg.get("gyro_bias_std", 0.0)))
        gyro_scale_std = max(0.0, float(cfg.get("gyro_scale_std", 0.0)))

        # Tires une fois par capteur (turn-on bias)
        self.accel_bias = self._rng.normal(0.0, accel_bias_std, 3) if accel_bias_std > 0 else np.zeros(3)
        self.gyro_bias = self._rng.normal(0.0, gyro_bias_std, 3) if gyro_bias_std > 0 else np.zeros(3)
        self.accel_scale = self._rng.normal(0.0, accel_scale_std, 3) if accel_scale_std > 0 else np.zeros(3)
        self.gyro_scale = self._rng.normal(0.0, gyro_scale_std, 3) if gyro_scale_std > 0 else np.zeros(3)

        self.accel_bias_0 = self.accel_bias.copy()
        self.gyro_bias_0 = self.gyro_bias.copy()

        self.last_vel = None  # None au premier appel : acceleration supposee nulle
        self.last_q = None

        # Verite terrain des erreurs
        self.accel_error_det = self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_bias.copy()
        self.q_mid = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_acc_body_true = np.zeros(3)
        self.last_gyro_body_true = np.zeros(3)

    def reset(self, vel: np.ndarray | None = None, orn_q=None) -> None:
        """Reinitialise la derivation (entre deux runs Monte-Carlo)."""
        self.last_vel = None if vel is None else np.asarray(vel, dtype=float).copy()
        self.last_q = None if orn_q is None else Q.normalize(orn_q)

    def _white_std(self, std: float, density, dt: float) -> float:
        if density is not None:
            d = float(density)
            if d > 0.0 and dt > 0.0:
                return d / np.sqrt(dt)
            return 0.0
        return std

    def measure(self, vel, orn_q, ang_vel=None, dt: float | None = None):
        """Renvoie (acc_body, gyro_body) ; vel en repere monde [m/s], orn_q corps -> monde, ang_vel ignore."""
        step = self.dt_nominal if (dt is None or float(dt) <= 0.0) else float(dt)

        vel = np.asarray(vel, dtype=float).reshape(3)
        q_k = Q.normalize(orn_q)
        q_prev = q_k if self.last_q is None else self.last_q

        if self.last_vel is None:
            acc_world = np.zeros(3)
        else:
            acc_world = (vel - self.last_vel) / step
        self.last_vel = vel.copy()
        self.last_q = q_k

        # Acceleration propre f = a - g_vec, avec g_vec = [0, 0, -g]
        acc_proper_world = acc_world + self.g_vector

        # Projection monde -> corps a mi-intervalle, comme la mecanisation de l'ESKF
        self.q_mid = Q.slerp(q_prev, q_k, 0.5)
        acc_body_true = Q.to_rot(self.q_mid).T @ acc_proper_world

        # Vitesse angulaire moyenne qui reproduit exactement la rotation de l'intervalle
        gyro_body_true = Q.log(Q.mul(Q.conj(q_prev), q_k)) / step

        self.last_acc_body_true = acc_body_true
        self.last_gyro_body_true = gyro_body_true

        if self.accel_bias_rw > 0.0:
            self.accel_bias = self.accel_bias + self._rng.normal(0.0, self.accel_bias_rw * np.sqrt(step), 3)
        if self.gyro_bias_rw > 0.0:
            self.gyro_bias = self.gyro_bias + self._rng.normal(0.0, self.gyro_bias_rw * np.sqrt(step), 3)

        self.accel_error_det = self.accel_scale * acc_body_true + self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_scale * gyro_body_true + self.gyro_bias

        a_std = self._white_std(self.accel_noise_std, self.accel_noise_density, step)
        acc_body = acc_body_true + self.accel_error_det
        if a_std > 0.0:
            acc_body = acc_body + self._rng.normal(0.0, a_std, 3)

        g_std = self._white_std(self.gyro_noise_std, self.gyro_noise_density, step)
        gyro_body = gyro_body_true + self.gyro_error_det
        if g_std > 0.0:
            gyro_body = gyro_body + self._rng.normal(0.0, g_std, 3)

        return acc_body, gyro_body
