import numpy as np

from utilities import quaternion as Q


class Sensor:
    """Classe de base pour les capteurs."""

    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée.")


class GNSSensor(Sensor):
    """
    Recepteur GNSS simule : position et vitesse bruitees.

    Parametres de configuration
    ---------------------------
    position_noise_std   [m]     ecart-type du bruit de position
    velocity_noise_std   [m/s]   ecart-type du bruit de vitesse

    Brouillage (degradation : la mesure existe mais elle est plus bruitee)
      jam_start, jam_end [s]
      jam_position_noise_std / jam_velocity_noise_std   valeurs imposees, ou
      jam_multiplier                                    facteur sur le nominal

    Coupure (indisponibilite : aucune mesure n'est delivree)
      outage_start, outage_end [s]       une fenetre, ou
      outages: [[t0, t1], [t2, t3]]      plusieurs fenetres

    seed   graine optionnelle d'un generateur dedie (sinon np.random global,
           pilote par simulation.seed).

    Precision rapportee
    -------------------
    Un recepteur reel publie une estimation de sa propre precision (hAcc, vAcc,
    sAcc chez u-blox). On l'emule : apres chaque mesure, `last_pos_std` et
    `last_vel_std` donnent l'ecart-type reellement applique, brouillage compris.
    Un filtre peut ainsi ajuster R mesure par mesure au lieu de faire une
    confiance aveugle a une valeur fixe.
    """

    def __init__(self, config: dict):
        super().__init__()
        config = dict(config or {})

        # Les valeurs negatives sont ramenees a 0 AVANT de servir de reference
        # au brouillage.
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

        # Precision rapportee par le recepteur pour la derniere mesure
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
        """
        Renvoie (position_mesuree, vitesse_mesuree) avec bruit gaussien,
        ou (None, None) pendant une coupure.
        Optionnel : t (temps simule) pour appliquer brouillage et coupures.
        """
        self.available = self.is_available(t)
        if not self.available:
            return None, None

        pos_std = self.pos_noise_std
        vel_std = self.vel_noise_std

        # Fenetre de brouillage
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
    """
    Centrale inertielle simulee : accelerometre 3 axes + gyrometre 3 axes,
    avec un modele d'erreur representatif d'un capteur MEMS.

    Chaine de mesure
    ----------------
      1. acceleration propre en repere monde : f_world = a_world + g
         (un accelerometre mesure a - g_vec, avec g_vec = [0, 0, -9.81],
          donc f = a + [0, 0, +9.81])
      2. projection en repere corps : f_body = R_wb^T f_world
      3. erreurs capteur appliquees en repere corps :
            mesure = (1 + s) * vrai + b(t) + n(t)
         avec s facteur d'echelle (constant), b(t) biais (constant + marche
         aleatoire), n(t) bruit blanc.

    Parametres de configuration (tous optionnels)
    ---------------------------------------------
    Accelerometre
      accel_noise_std       [m/s^2]           ecart-type du bruit blanc discret
      accel_noise_density   [m/s^2/sqrt(Hz)]  si fourni, prioritaire :
                                              std_discret = density / sqrt(dt)
      accel_noise_mean      [m/s^2]           offset deterministe (toutes voies)
      accel_bias_std        [m/s^2]           biais initial, tire a l'init
      accel_bias_rw         [m/s^2/sqrt(s)]   marche aleatoire du biais
      accel_scale_std       [-]               erreur de facteur d'echelle
    Gyrometre
      gyro_noise_std        [rad/s]
      gyro_noise_density    [rad/s/sqrt(Hz)]
      gyro_bias_std         [rad/s]
      gyro_bias_rw          [rad/s/sqrt(s)]
      gyro_scale_std        [-]
    Commun
      gravity               [m/s^2]           norme de g (defaut 9.81)
      dt                    [s]               pas nominal, surcharge l'argument
      seed                  [int]             graine pour biais/echelle (Monte-Carlo)

    Coherence avec une mecanisation inertielle
    -----------------------------------------
    Une vraie centrale delivre des INCREMENTS sur l'intervalle [k-1, k]
    (increment de vitesse, increment angulaire), pas des valeurs instantanees.
    On emule ce comportement, ce qui rend la propagation du filtre exacte en
    l'absence d'erreurs capteur :

    - gyrometre : vitesse angulaire moyenne qui reproduit EXACTEMENT la rotation
      de l'intervalle, omega = Log(q(k-1)^-1 (x) q(k)) / dt. Un filtre qui
      integre q(k) = q(k-1) (x) Exp(omega dt) retrouve donc l'attitude vraie.
    - accelerometre : increment de vitesse moyen (v(k) - v(k-1)) / dt + g,
      projete en repere corps avec l'attitude au MILIEU de l'intervalle
      (slerp a 1/2). Le filtre doit faire la rotation inverse avec SA propre
      attitude a mi-intervalle, q(k-1) (x) Exp(omega dt / 2), qui coincide.

    Pourquoi c'est important : projeter avec l'attitude de fin d'intervalle
    alors que le filtre tourne avec celle de debut cree une erreur de rotation
    d'un demi-pas. A 2 rad/s et dt = 12.5 ms, c'est 0.0125 rad applique a
    9.81 m/s^2, soit 0.12 m/s^2 : du meme ordre qu'un biais accelerometre. Le
    filtre l'interpreterait comme un biais et son test de coherence serait faux.

    Verite terrain des erreurs (pour le NEES)
    -----------------------------------------
    Apres chaque mesure :
      accel_error_det : erreur deterministe accelerometre, echelle + biais + offset
      gyro_error_det  : erreur deterministe gyrometre, echelle + biais
      q_mid           : attitude vraie a mi-intervalle (sert au filtre 6 etats,
                        qui n'estime pas l'attitude)

    Notes
    -----
    - `dt` est fourni par l'appelant (pas reel de la boucle) ou, a defaut, par
      le pas nominal du constructeur. Indispensable puisque acc = dv/dt.
    - `ang_vel` n'est plus necessaire : le gyrometre est deduit de la variation
      d'attitude. L'argument est conserve pour compatibilite.
    - La derivation a partir de la verite terrain reste une emulation : ce sont
      les erreurs ci-dessus qui rendent le capteur realiste.
    """

    def __init__(self, config: dict | None = None, dt: float | None = None):
        super().__init__()
        cfg = dict(config or {})

        # --- Pas de temps nominal (utilise si l'appelant n'en fournit pas) ---
        dt_cfg = cfg.get("dt", dt)
        if dt_cfg is None or float(dt_cfg) <= 0.0:
            dt_cfg = 1.0 / 100.0
        self.dt_nominal = float(dt_cfg)

        # --- Gravite ---
        g = float(cfg.get("gravity", 9.81))
        self.g_vector = np.array([0.0, 0.0, g], dtype=float)

        # --- Generateur dedie : rend les tirages reproductibles en Monte-Carlo ---
        seed = cfg.get("seed", None)
        self._rng = np.random.default_rng(None if seed is None else int(seed))

        # --- Parametres accelerometre ---
        self.accel_noise_std = max(0.0, float(cfg.get("accel_noise_std", 0.0)))
        self.accel_noise_density = cfg.get("accel_noise_density", None)
        self.accel_noise_mean = float(cfg.get("accel_noise_mean", 0.0))
        self.accel_bias_rw = max(0.0, float(cfg.get("accel_bias_rw", 0.0)))
        accel_bias_std = max(0.0, float(cfg.get("accel_bias_std", 0.0)))
        accel_scale_std = max(0.0, float(cfg.get("accel_scale_std", 0.0)))

        # --- Parametres gyrometre ---
        self.gyro_noise_std = max(0.0, float(cfg.get("gyro_noise_std", 0.0)))
        self.gyro_noise_density = cfg.get("gyro_noise_density", None)
        self.gyro_bias_rw = max(0.0, float(cfg.get("gyro_bias_rw", 0.0)))
        gyro_bias_std = max(0.0, float(cfg.get("gyro_bias_std", 0.0)))
        gyro_scale_std = max(0.0, float(cfg.get("gyro_scale_std", 0.0)))

        # --- Etats d'erreur tires une fois par capteur (turn-on bias) ---
        self.accel_bias = self._rng.normal(0.0, accel_bias_std, 3) if accel_bias_std > 0 else np.zeros(3)
        self.gyro_bias = self._rng.normal(0.0, gyro_bias_std, 3) if gyro_bias_std > 0 else np.zeros(3)
        self.accel_scale = self._rng.normal(0.0, accel_scale_std, 3) if accel_scale_std > 0 else np.zeros(3)
        self.gyro_scale = self._rng.normal(0.0, gyro_scale_std, 3) if gyro_scale_std > 0 else np.zeros(3)

        # Biais initiaux conserves pour diagnostic / comparaison a un filtre
        # qui chercherait a les estimer.
        self.accel_bias_0 = self.accel_bias.copy()
        self.gyro_bias_0 = self.gyro_bias.copy()

        # --- Etat interne de derivation ---
        self.last_vel = None  # None => premier appel, acceleration supposee nulle
        self.last_q = None

        # --- Sorties de diagnostic (verite terrain des erreurs) ---
        self.accel_error_det = self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_bias.copy()
        self.q_mid = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_acc_body_true = np.zeros(3)
        self.last_gyro_body_true = np.zeros(3)

    # ------------------------------------------------------------------
    def reset(self, vel: np.ndarray | None = None, orn_q=None) -> None:
        """Reinitialise la derivation (utile entre deux runs Monte-Carlo)."""
        self.last_vel = None if vel is None else np.asarray(vel, dtype=float).copy()
        self.last_q = None if orn_q is None else Q.normalize(orn_q)

    def _white_std(self, std: float, density, dt: float) -> float:
        """Ecart-type du bruit blanc discret pour un pas dt."""
        if density is not None:
            d = float(density)
            if d > 0.0 and dt > 0.0:
                return d / np.sqrt(dt)
            return 0.0
        return std

    # ------------------------------------------------------------------
    def measure(self, vel, orn_q, ang_vel=None, dt: float | None = None):
        """
        Simule une mesure IMU sur l'intervalle ecoule depuis l'appel precedent.

        Args:
            vel      : vitesse lineaire verite terrain en repere monde [m/s]
            orn_q    : quaternion d'attitude vrai [x, y, z, w] (corps -> monde)
            ang_vel  : ignore (conserve pour compatibilite), cf. docstring
            dt       : pas reel depuis le dernier appel [s]; None -> dt nominal

        Returns:
            (acc_body, gyro_body) : np.ndarray de dimension 3 chacun.
        """
        step = self.dt_nominal if (dt is None or float(dt) <= 0.0) else float(dt)

        vel = np.asarray(vel, dtype=float).reshape(3)
        q_k = Q.normalize(orn_q)
        q_prev = q_k if self.last_q is None else self.last_q

        # --- 1. Increment de vitesse moyen sur l'intervalle ---
        if self.last_vel is None:
            acc_world = np.zeros(3)  # pas d'acceleration fictive au 1er pas
        else:
            acc_world = (vel - self.last_vel) / step
        self.last_vel = vel.copy()
        self.last_q = q_k

        # --- 2. Acceleration propre : f = a - g_vec = a + [0, 0, g] ---
        acc_proper_world = acc_world + self.g_vector

        # --- 3. Attitude a mi-intervalle et projection monde -> corps ---
        self.q_mid = Q.slerp(q_prev, q_k, 0.5)
        acc_body_true = Q.to_rot(self.q_mid).T @ acc_proper_world

        # --- 4. Vitesse angulaire moyenne reproduisant la rotation exacte ---
        gyro_body_true = Q.log(Q.mul(Q.conj(q_prev), q_k)) / step

        self.last_acc_body_true = acc_body_true
        self.last_gyro_body_true = gyro_body_true

        # --- 5. Marche aleatoire des biais ---
        if self.accel_bias_rw > 0.0:
            self.accel_bias = self.accel_bias + self._rng.normal(0.0, self.accel_bias_rw * np.sqrt(step), 3)
        if self.gyro_bias_rw > 0.0:
            self.gyro_bias = self.gyro_bias + self._rng.normal(0.0, self.gyro_bias_rw * np.sqrt(step), 3)

        # --- 6. Erreurs deterministes (verite terrain pour le NEES) ---
        self.accel_error_det = self.accel_scale * acc_body_true + self.accel_bias + self.accel_noise_mean
        self.gyro_error_det = self.gyro_scale * gyro_body_true + self.gyro_bias

        # --- 7. Mesures : vrai + erreur deterministe + bruit blanc ---
        a_std = self._white_std(self.accel_noise_std, self.accel_noise_density, step)
        acc_body = acc_body_true + self.accel_error_det
        if a_std > 0.0:
            acc_body = acc_body + self._rng.normal(0.0, a_std, 3)

        g_std = self._white_std(self.gyro_noise_std, self.gyro_noise_density, step)
        gyro_body = gyro_body_true + self.gyro_error_det
        if g_std > 0.0:
            gyro_body = gyro_body + self._rng.normal(0.0, g_std, 3)

        return acc_body, gyro_body
