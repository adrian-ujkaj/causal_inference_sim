import numpy as np


class DrydenGustModel:
    """
    Turbulence atmospherique, modele de Dryden basse altitude (MIL-F-8785C).

    Parametres du modele (h en ft, W20 = vent a 20 ft)
    ---------------------------------------------------
        sigma_w = 0.1 * W20
        sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h) ** 0.4
        L_w = h
        L_u = L_v = h / (0.177 + 0.000823 h) ** 1.2
    `turbulence_intensity_knots` est W20, en noeuds. Avec W20 = 10 kt pres du sol,
    on obtient sigma_u ~ 1.0 m/s et sigma_w ~ 0.5 m/s.

    Realisation numerique
    ---------------------
    Chaque composante est un processus de Gauss-Markov du premier ordre
        x(k+1) = a x(k) + sigma sqrt(1 - a^2) n(k),    a = exp(-V dt / L)
    qui a EXACTEMENT l'ecart-type sigma et le temps de correlation L / V du
    modele de Dryden. C'est l'approximation usuelle : le spectre longitudinal
    est celui de Dryden ; pour v et w (second ordre chez Dryden), la pente haute
    frequence differe, mais variance et longueur de correlation sont respectees.
    Avantage decisif : l'etat EST la rafale. Quand l'altitude ou la vitesse
    changent, on change a et sigma sans discontinuite de la rafale.

    Repere : u est porte par la vitesse air horizontale par rapport au vent
    MOYEN (l'appelant ne doit pas y inclure la rafale, sinon la direction des
    rafales tourne au hasard), v lui est perpendiculaire dans le plan
    horizontal, w est vertical. La sortie est en repere MONDE.

    Rafale forte sur une fenetre (scenarios de test) : `burst=(t0, t1, W20)`
    porte l'intensite a W20 entre t0 et t1. L'etat de la rafale est remis a
    l'echelle au changement de niveau : meme realisation aleatoire, intensite
    multipliee. Sans fenetre, le comportement est inchange.

    Points d'attention verifies par les tests (tests/test_simulation_fixes.py) :
    bruit mis a l'echelle du pas de temps, sigma_w = 0.1 * W20 converti des
    noeuds, ecart-type et correlation conformes au modele.
    """

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
        # Dryden suppose un ecoulement : en vol stationnaire la vitesse air est
        # nulle et le temps de correlation L / V infini. On borne V par dessous.
        self.min_airspeed = float(min_airspeed)

        # Generateur dedie. Sans graine explicite, il est tire du generateur
        # global : simulation.seed rend donc le vent reproductible.
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(int(seed))

        self._gust = None  # [u, v, w] en m/s, repere du vent
        self._heading = np.array([1.0, 0.0])
        self.last_sigmas = np.zeros(3)  # diagnostic

    # ------------------------------------------------------------------
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
        """
        Avance d'un pas et renvoie le vent total (moyen + rafale) en repere monde.

        h_meters     : altitude [m]
        V_ms         : norme de la vitesse air [m/s]
        airspeed_vec : vitesse air (vitesse sol - vent) en repere monde, pour
                       orienter u ; optionnel, sinon l'orientation precedente.
        t            : temps de simulation [s], utilise par la fenetre `burst`.
        """
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
            # Initialisation dans le regime stationnaire : pas de montee en
            # charge artificielle de la turbulence au debut de la simulation.
            self._gust = self._rng.normal(0.0, 1.0, 3) * sig
        else:
            a_k = np.exp(-V * self.dt / L)
            self._gust = a_k * self._gust + sig * np.sqrt(1.0 - a_k * a_k) * self._rng.normal(0.0, 1.0, 3)

        cx, cy = self._heading
        u, v, w = self._gust
        gust_world = np.array([cx * u - cy * v, cy * u + cx * v, w])
        return self.mean_wind + gust_world
