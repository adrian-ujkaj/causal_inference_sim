import numpy as np


class DrydenGustModel:
    """Turbulence de Dryden basse altitude (MIL-F-8785C).

    Chaque composante est un processus de Gauss-Markov du premier ordre, avec l'ecart-type
    et la longueur de correlation du modele (W20 en noeuds, h en ft dans les formules).
    u suit la vitesse air horizontale par rapport au vent moyen, w est vertical ; sortie en
    repere monde. burst=(t0, t1, W20) : turbulence plus forte entre t0 et t1 (meme tirage,
    intensite multipliee)."""

    KNOT_TO_FTS = 1.68781
    FT_TO_M = 0.3048
    M_TO_FT = 3.28084

    def __init__(
        self,
        dt,
        turbulence_intensity_knots=15,
        mean_wind=(0.0, 0.0, 0.0),
        seed=None,
        min_airspeed=1.0,
        burst=None,
    ):
        self.dt = float(dt)
        self.turbulence_level = max(0.0, float(turbulence_intensity_knots))
        self.base_level = self.turbulence_level
        # (t0, t1, W20) : intensite W20 entre t0 et t1, None sinon
        self.burst = None if burst is None else tuple(float(x) for x in burst)
        self.mean_wind = np.asarray(mean_wind, dtype=float).reshape(3)
        # V borne par dessous (en stationnaire, L / V serait infini)
        self.min_airspeed = float(min_airspeed)

        # Generateur dedie (reproductible via simulation.seed)
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))

        self._gust = None  # [u, v, w] en m/s, repere du vent
        self._heading = np.array([1.0, 0.0])
        self.last_sigmas = np.zeros(3)  # diagnostic

    def _params(self, h_m: float, V_ms: float):
        """Ecarts-types [m/s] et longueurs de correlation [m] pour (h, V)."""
        h = max(float(h_m) * self.M_TO_FT, 10.0)  # validite du modele : h >= 10 ft
        k = 0.177 + 0.000823 * min(h, 1000.0)
        sigma_w = 0.1 * self.turbulence_level * self.KNOT_TO_FTS  # ft/s
        sigma_u = sigma_w / k**0.4
        L_w = h
        L_u = h / k**1.2
        sig = np.array([sigma_u, sigma_u, sigma_w]) * self.FT_TO_M
        L = np.array([L_u, L_u, L_w]) * self.FT_TO_M
        return sig, L

    def _set_level(self, level: float) -> None:
        """Change l'intensite ; la rafale en cours est remise a l'echelle."""
        level = max(0.0, float(level))
        if level == self.turbulence_level:
            return
        if self._gust is not None and self.turbulence_level > 0:
            self._gust = self._gust * (level / self.turbulence_level)
        else:
            self._gust = None  # reinitialisee au prochain pas dans le regime stationnaire
        self.turbulence_level = level

    def step(self, h_meters, V_ms, airspeed_vec=None, t=None):
        """Avance d'un pas et renvoie le vent total (moyen + rafale) en repere monde.
        airspeed_vec (vitesse sol - vent moyen) oriente u ; t sert a la fenetre burst."""
        if self.burst is not None and t is not None:
            t0, t1, level = self.burst
            self._set_level(level if t0 <= t < t1 else self.base_level)
        V = max(float(V_ms), self.min_airspeed)
        sig, L = self._params(h_meters, V)
        self.last_sigmas = sig

        if airspeed_vec is not None:
            a = np.asarray(airspeed_vec, dtype=float).reshape(3)[:2]
            n = float(np.linalg.norm(a))
            if n > 0.2:
                self._heading = a / n

        if self._gust is None:
            # Premier pas : tirage dans le regime stationnaire
            self._gust = self._rng.normal(0.0, 1.0, 3) * sig
        else:
            a_k = np.exp(-V * self.dt / L)
            self._gust = a_k * self._gust + sig * np.sqrt(1.0 - a_k * a_k) * self._rng.normal(0.0, 1.0, 3)

        cx, cy = self._heading
        u, v, w = self._gust
        gust_world = np.array([cx * u - cy * v, cy * u + cx * v, w])
        return self.mean_wind + gust_world
