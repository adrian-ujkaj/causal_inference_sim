import os
import csv
import threading
import pybullet as p
import numpy as np
import zmq
import json
import random

# Utility Imports & Control
from entities.agent import Agent
from entities.sensor import GNSSensor, IMUSensor
from Control.EKF import INSGNSSFilter
from Control.ESKF import ESKF
from utilities import quaternion as Q
from utilities.csv_buffer import CsvBuffer
from environment.wind import DrydenGustModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class UAV(Agent):
    """
    Autonomous Unmanned Aerial Vehicle (UAV) simulator with physics, control, and swarm capabilities.
    Integrates PyBullet physics engine with advanced control systems for realistic quadrotor simulation
    in multi-agent environments. Features include:
    **Physics & Control:**
    - Precise rotor dynamics with thrust and torque coefficients
    - DSL PID controller running at configurable frequency (default 100Hz)
    - Aerodynamic drag modeling with airspeed compensation
    - Quaternion-based orientation tracking
    **Navigation & Planning:**
    - A* path planning with asynchronous execution thread
    - Waypoint management with dynamic replanning
    - Repulsive force computation for obstacle avoidance
    - Collision detection and safety radius enforcement
    **Navigation filter (GNSS/INS):**
    - 15-state error-state Kalman filter (ESKF) by default, 6-state filter as baseline
    - GNSS noise, rate, outages and jamming; IMU noise and biases
    - Optional: inner loop flown on the ESKF attitude (`filter.attitude_source`)
    **Swarm Coordination:**
    - Leader-follower formation control
    - ZMQ-based state broadcasting and message handling
    - Neighbor tracking with network delay simulation
    - Distributed decision-making for autonomous swarms
    **Environmental Simulation:**
    - Dryden Gust Model for realistic turbulence generation
    - Wind-aware flight dynamics and airspeed calculations
    - Support for generated or custom environment layouts
    **Logging & Analysis:**
    - CSV-based state logging at each control cycle
    - Causal inference metrics: ground truth vs. estimated state
    - Tracking error, collision flags, and interaction forces
    - Comprehensive data for post-simulation analysis
    Attributes:
        config (dict): Configuration dictionary with UAV parameters
        dt (float): Physics simulation timestep (typically 1/240)
        physics_client_id (int): PyBullet client identifier
        name (str): UAV identifier
        bodyId (int): PyBullet body ID
        CTRL_FREQ (int): Control loop frequency in Hz
        CTRL_DT (float): Control timestep (1/CTRL_FREQ)
        mass (float): UAV mass in kg
        ekf (ESKF | INSGNSSFilter): navigation filter (attribute name kept for compatibility)
        gnss (GNSSensor): GNSS sensor for position/velocity measurement
        imu (IMUSensor): IMU (specific force and angular rate)
        planner: A* path planner instance
        waypoints (list[np.ndarray]): List of target waypoints
        active_path (list[np.ndarray]): Current planned path segments
        swarm_active (bool): Whether UAV is part of active swarm
        leader (bool): Whether UAV is swarm leader
        other_agent_pos (dict): Positions of neighboring agents
        message_buffer (list): Queue of delayed network messages
        current_wind (np.ndarray): Current wind vector [m/s]
        log_file (str): Path to CSV log file
    """

    def __init__(
        self,
        config: dict,
        physics_client_id: int,
        dt: float,
        known_obstacles_config: dict,
        planner,
        world_type: str,
    ):
        """
        Initialize a UAV entity with physics simulation, control systems, and autonomous capabilities.

        Args:
            config (dict): Configuration dictionary containing:
            - name (str): UAV identifier. Defaults to "UAV".
            - body_id (int): Physics body ID. Defaults to 0000.
            - mass (float): UAV mass in kg. Defaults to 1.5.
            - urdf_path (str): Path to URDF model file. Defaults to "assets/quadrotor.urdf".
            - start_pos (list): Initial position [x, y, z]. Defaults to [0, 0, 1.0].
            - start_orn_euler (list): Initial orientation in Euler angles [roll, pitch, yaw]. Defaults to [0, 0, 0].
            - waypoints (list[list]): List of waypoint coordinates [[x, y, z], ...]. Defaults to [[0, 0, 1]].
            - ctrl_freq (int): Control loop frequency in Hz. Defaults to 100.
            - wind_mean (list): Mean wind vector [x, y, z] in m/s. Defaults to [0, 0, 0].
            - turbulence (float): Dryden gust model turbulence intensity (0-20). Defaults to 15.
            - communication (dict): Network settings with keys:
                - com_period (float): Broadcast interval in seconds. Defaults to 0.1.
                - com_delay_mean (float): Mean communication latency. Defaults to 0.1.
                - com_delay_std (float): Std dev of communication latency. Defaults to 0.02.
            - sensors (dict): Sensor configurations with keys:
                - gnss (dict): GNSS sensor settings (frequency, delay_mean, delay_std).
                - imu (dict): IMU sensor settings (noise parameters).
            - physics (dict): Physics parameters:
                - thrust_coeff (float): Thrust coefficient KF. Defaults to 6.11e-8.
                - torque_coeff (float): Torque coefficient KM. Defaults to 1.5e-9.
                - max_rpm (float): Maximum rotor RPM. Defaults to 22000.
                - max_speed (float): Maximum velocity in m/s. Defaults to 5.
                - max_repulsive_force (float): Max obstacle avoidance force in N. Defaults to 2.0.
                - safety_radius (float): Collision avoidance radius in m. Defaults to 2.0.
            - radar (list, optional): Radar connection configs with ip and port.
            physics_client_id (int): PyBullet physics client identifier.
            dt (float): Physics simulation timestep in seconds (typically 1/240).
            known_obstacles_config (list[dict]): Configuration list for static obstacles with keys:
            - center (list): [x, y, z] center position.
            - height, width, length (float): Obstacle dimensions.
            planner: A* path planner instance with planning and repulsive force computation interface.
            world_type (str): Environment type - "generated" or "custom".

        Initializes:
            - Physics engine: Body properties, dynamics, external force/torque application.
            - Control system: DSL PID controller running at configurable frequency (default 100Hz).
            - Navigation: Waypoint management and A* path planning with async thread support.
            - Navigation: GNSS/IMU filter (ESKF by default).
            - Obstacle avoidance: Repulsive force computation and collision detection.
            - Swarm coordination: Leader-follower formation control and neighbor tracking.
            - Communication: ZMQ-based state broadcasting and message handling.
            - Wind simulation: Dryden Gust Model for realistic turbulence.
            - Logging: CSV state tracking with causal analysis metrics.

        Note:
            Physics loop runs at 240Hz; logic/control loop runs at configurable frequency (100Hz default).
        """
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "UAV")
        self.bodyId = config.get("body_id", 0000)
        self.type = "uav"
        self.mass = config.get("mass", 1.5)  # kg

        urdf_path = config.get("urdf_path", "assets/quadrotor.urdf")
        self.start_pos = list(config.get("start_pos", [0, 0, 1.0]))

        # --- GARDE-FOU 1 : altitude d'apparition ---
        # Un drone place a z=0 apparait en intersection avec le sol : PyBullet
        # genere des forces de contact des le premier pas, le drone peut rester
        # epingle au sol ou partir en collision avec un voisin, et il ne decolle
        # jamais. On remonte l'altitude d'apparition au minimum viable.
        self.spawn_min_alt = float(config.get("spawn_min_alt", 0.15))
        if self.start_pos[2] < self.spawn_min_alt:
            print(
                f"[{config.get('name', 'UAV')}] start_pos z={self.start_pos[2]:.3f} "
                f"< {self.spawn_min_alt:.3f} : remonte a {self.spawn_min_alt:.3f} "
                f"(apparition en contact avec le sol)"
            )
            self.start_pos[2] = self.spawn_min_alt

        self.start_orn = p.getQuaternionFromEuler(config.get("start_orn_euler", [0, 0, 0]))
        super().__init__(urdf_path, self.start_pos, self.start_orn, physics_client_id, self.dt)
        self._sim_time = 0.0
        self._tick = 0

        # --- FREQUENCE DE CONTROLE ---
        # Le controle tourne tous les `ctrl_every` pas physiques. La periode
        # effective est donc un multiple entier de dt, et c'est ELLE (pas
        # 1/ctrl_freq) qu'il faut donner au PID et au filtre. Un test sur le
        # temps flottant (`t - t_dernier >= 1/80` avec dt = 0.00416) echoue de
        # justesse apres 3 pas : le controle tournerait a 60 Hz alors que le
        # PID se croit a 80 Hz.
        self.CTRL_FREQ = float(self.config.get("ctrl_freq", 100))
        self.ctrl_every = max(1, int(round(1.0 / (self.CTRL_FREQ * self.dt))))
        self.CTRL_DT = self.ctrl_every * self.dt
        if abs(1.0 / self.CTRL_DT - self.CTRL_FREQ) > 0.02 * self.CTRL_FREQ:
            print(
                f"[{config.get('name', 'UAV')}] ctrl_freq={self.CTRL_FREQ:g} Hz non "
                f"multiple du pas physique : controle a {1.0 / self.CTRL_DT:.1f} Hz"
            )
        # Le premier controle a lieu au pas 1 (t = dt) : intervalle nominal
        self.last_ctrl_time = self.dt - self.CTRL_DT

        # --- PHYSICS ---
        self.KF = self.config.get("physics", {}).get("thrust_coeff", 6.11e-8)
        self.KM = self.config.get("physics", {}).get("torque_coeff", 1.5e-9)
        self.G = 9.81
        self.MAX_RPM = config.get("physics", {}).get("max_rpm", 22000.0)
        self.max_speed = config.get("physics", {}).get("max_speed", 5)
        self.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])

        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.last_rpms = np.zeros(4)

        # Assiette maximale commandee (deg). Au-dela de 90 deg le drone est
        # irrecuperable avec ce controleur : on garde une marge large.
        self.max_tilt_deg = float(self.config.get("max_tilt_deg", 30.0))
        # Vol pres du sol (garde-fou 6)
        self.ground_min_alt = float(self.config.get("ground_min_alt", 0.15))
        self.ground_clear_alt = float(self.config.get("ground_clear_alt", 0.6))
        self.ground_speed_floor = float(self.config.get("ground_speed_floor", 0.1))
        self.ground_tilt_deg = float(self.config.get("ground_tilt_deg", 10.0))
        self.max_descent_speed = float(self.config.get("max_descent_speed", 1.5))
        self.max_climb_speed = float(self.config.get("max_climb_speed", 1.2))
        # Vitesse maximale de la consigne de lacet [rad/s] (garde-fou 7)
        self.max_yaw_rate = float(self.config.get("max_yaw_rate", 1.0))
        self._yaw_cmd = float(config.get("start_orn_euler", [0, 0, 0])[2])
        self.ground_descent_speed = float(self.config.get("ground_descent_speed", 0.4))
        self._ground_factor = 1.0
        # Part maximale de la demande horizontale laissee au terme integral
        # (anti-emballement, cf. _limit_tilt_demand).
        self.integral_share = float(self.config.get("integral_share", 0.3))
        # Demande verticale admissible, en fraction du poids. Le minimum garantit
        # que l'axe de poussee pointe vers le haut ET que les moteurs gardent une
        # marge pour produire du couple : en vol stationnaire un moteur CF2X tourne
        # a 14 468 tr/min et son minimum est 9 440 tr/min, donc sous 0.43 x poids
        # de poussee collective tous les moteurs sont en butee basse et le
        # controle d'attitude disparait. Le maximum reste sous la poussee
        # disponible du CF2X (environ 2.25 fois le poids) AVEC une marge pour les
        # couples : a 30 deg la poussee totale vaut 1.6 / cos(30) = 1.85 fois le
        # poids ; au-dela, les moteurs saturent en montee et le controleur
        # d'attitude n'a plus de marge differentielle pour corriger.
        self.min_thrust_ratio = float(self.config.get("min_thrust_ratio", 0.6))
        self.max_thrust_ratio = float(self.config.get("max_thrust_ratio", 1.6))
        self._loss_of_control = False

        # --- NAVIGATION ---
        wp_list = config.get("waypoints", [])
        if not wp_list:
            wp_list = [[0, 0, 1]]

        # --- GARDE-FOU 2 : altitude minimale des waypoints ---
        # Un waypoint a z=0 est un piege : (a) il est sous le plancher du
        # domaine A* (world.Astar.world_bounds.z, typiquement 0.01), donc le
        # planificateur repond "cible inaccessible" indefiniment ; (b) le
        # controleur commande une altitude nulle, le drone se pose, reste en
        # contact avec le sol et n'atteint jamais le critere d'arrivee
        # (dist < 0.5 m), car il est bloque par la friction. Resultat : blocage
        # definitif, sans message d'erreur. On releve donc ces waypoints.
        self.min_waypoint_alt = float(config.get("min_waypoint_alt", 0.30))
        clamped = 0
        wp_clean = []
        for w in wp_list:
            w = np.array(w, dtype=float)
            if w[2] < self.min_waypoint_alt:
                w[2] = self.min_waypoint_alt
                clamped += 1
            wp_clean.append(w)
        if clamped:
            print(
                f"[{config.get('name', 'UAV')}] {clamped} waypoint(s) sous "
                f"{self.min_waypoint_alt:.2f} m releve(s) a cette altitude "
                f"(un waypoint au sol bloque le planificateur et le controle)"
            )

        first_wp = np.array(self.start_pos, dtype=float) + np.array([0.0, 0.0, 1.0])
        self.waypoints = [first_wp] + wp_clean
        self.wp_idx = 0

        # Sequencement des waypoints (cf. garde-fou 5)
        self.wp_tol = float(config.get("wp_tolerance", 0.5))  # arrivee franche
        self.wp_capture = float(config.get("wp_capture_radius", 1.5))  # rayon de capture
        self.wp_hyst = float(config.get("wp_hysteresis", 0.3))  # marge "depasse"
        self._wp_track_idx = -1
        self._wp_min_dist = np.inf

        # --- OBSTACLES & PLANNING ---
        self.obs_dic = known_obstacles_config  # Combined list for avoidance
        print(len(self.obs_dic), "known obstacle points loaded.")
        self.environment = world_type

        self.planner = planner

        self.target_yaw_cache = 0.0

        # Parametres d'evitement : lus dans `physics`, puis au niveau de l'agent.
        # (drone_3 les declare au niveau de l'agent : ils etaient ignores et il
        # volait avec les valeurs par defaut 2.0 m / 2.0 N.)
        phys = self.config.get("physics", {}) or {}
        self.max_repulsive_force = float(
            phys.get("max_repulsive_force", self.config.get("max_repulsive_force", 2.0))
        )
        self.safety_radius = float(phys.get("safety_radius", self.config.get("safety_radius", 2.0)))
        self.repulsion_gain_s = float(self.config.get("repulsion_gain_s", 3.0 / 80.0))
        self.lookahead_m = float(self.config.get("lookahead_m", 1.0))
        self.last_repulsive_force_mag = 0.0
        self.dist_to_nearest_neighbor = float("inf")

        # Planning States
        self.active_path = []
        self.is_planning = False
        self.planning_thread = None
        self.replan_timer = 0
        self.calculation_fail_count = 0
        # Relance de A* apres echec : ~1.25 s, quelle que soit la cadence
        self.replan_ticks = max(1, int(round(1.25 / self.CTRL_DT)))
        self.planning_sync = bool(self.config.get("planning_sync", False))
        self.direct_leg_xy = float(self.config.get("direct_leg_xy", 0.75))
        self._plan_wp_idx = -1  # waypoint pour lequel un plan existe
        self._planning_for_wp = -1  # waypoint vise par le calcul en cours
        self._hold_pos = None  # point de maintien (attente / fin de mission)

        self.zmq_ctx = zmq.Context()
        self.sub_socket = None
        self.pub_socket = None
        self.radar_sub_socket = None

        # --- SWARM CONTROL ---
        self.swarm_active = False
        self.leader = False
        self.other_agent_pos = {}
        self.neighbors_data = {}
        self.swarm_name = []

        # --- COMMUNICATION ---
        com = self.config.get("communication")
        self.com_period = com.get("com_period", 0.1)
        self.last_com_time = -self.com_period
        self.message_buffer = []
        self.perception_delay_mean = com.get("com_delay_mean", 0.1)  # 100ms delay
        self.perception_delay_std = com.get("com_delay_std", 0.02)  # +/- 20ms

        # --- SENSORS ---
        sens = self.config.get("sensors", {})

        # --- FILTRE DE NAVIGATION ---
        # filter.type : "eskf" (15 etats, defaut) ou "kf6" (position-vitesse).
        # Les deux sont regles a partir des caracteristiques DECLAREES des
        # capteurs (sensors.imu, sensors.gnss), pas de constantes arbitraires.
        # (Attribut nomme `ekf` pour compatibilite avec le reste du code.)
        fcfg = dict(self.config.get("filter", {}) or {})
        self.filter_type = str(fcfg.get("type", "eskf")).lower()
        start_yaw = float(config.get("start_orn_euler", [0, 0, 0])[2])
        fcfg.setdefault("initial_yaw", start_yaw)
        FilterClass = ESKF if self.filter_type == "eskf" else INSGNSSFilter
        self.ekf = FilterClass(
            self.CTRL_DT, gnss_config=sens.get("gnss", {}), imu_config=sens.get("imu", {}), config=fcfg
        )
        self.ekf.init_state(self.start_pos, np.zeros(3))

        # Source de l'attitude donnee a la boucle interne du controleur :
        #  "truth"  : attitude vraie PyBullet (defaut)
        #  "filter" : attitude estimee par l'ESKF ; le drone vole alors
        #             entierement sur sa propre navigation. Sans magnetometre,
        #             le lacet n'est observable qu'en acceleration horizontale.
        self.attitude_source = str(fcfg.get("attitude_source", "truth")).lower()
        if self.attitude_source == "filter" and self.filter_type != "eskf":
            print(
                f"[{self.name}] attitude_source='filter' exige filter.type='eskf' : attitude vraie utilisee"
            )
            self.attitude_source = "truth"
        self._gnss_updated = False
        self._gnss_err = float("nan")
        self.gnss = GNSSensor(sens.get("gnss", {}))
        # dt nominal = periode de la boucle de controle : c'est a cette cadence
        # que l'IMU est interrogee (cf. _update_control_loop).
        self.imu = IMUSensor(sens.get("imu", {}), dt=self.CTRL_DT)
        self.last_imu_gyro = np.zeros(3)

        self.gnss_freq = sens.get("gnss", {}).get("frequency", 10.0)  # 10 Hz (Realistic Standard)
        self.gnss_dt = 1.0 / self.gnss_freq
        self.last_gnss_update_time = -self.gnss_dt
        self.gnss_delay_mean = sens.get("gnss", {}).get("delay_mean", 0.1)
        self.gnss_delay_std = sens.get("gnss", {}).get("delay_std", 0.01)
        self.next_gnss_trigger = 0.0

        # Nominal GNSS schedule (no long-term drift) + latest raw measurement cache
        self.last_gnss_nominal_time = 0.0
        self.last_gnss_meas_pos = np.array(self.start_pos, dtype=np.float32)
        self.last_gnss_meas_vel = np.zeros(3, dtype=np.float32)

        # --- WIND ---
        # config['wind'] = {'wind_mean': [x, y, z], 'turbulence': W20,
        #                   'burst_start': t0, 'burst_end': t1, 'burst_turbulence': W20}
        # (la rafale forte burst_* est optionnelle ; ancienne syntaxe
        # config['wind_mean'], config['turbulence'] toujours acceptee)
        self.current_wind = np.zeros(3)
        wind_cfg = self.config.get("wind", {}) if isinstance(self.config.get("wind", {}), dict) else {}
        self.mean_wind = wind_cfg.get("wind_mean", self.config.get("wind_mean", [0, 0, 0]))
        self.turbulence = wind_cfg.get("turbulence", self.config.get("turbulence", 15))
        burst = None
        if "burst_turbulence" in wind_cfg:
            burst = (
                float(wind_cfg.get("burst_start", 0.0)),
                float(wind_cfg.get("burst_end", float("inf"))),
                float(wind_cfg["burst_turbulence"]),
            )
        self.wind_module = DrydenGustModel(self.dt, self.turbulence, self.mean_wind, burst=burst)

        # radar
        radar_list = self.config.get("radar", None)
        if radar_list:
            self.radar_com_setup(radar_list)

        # Logs
        self.logging_enabled = True
        # Logs (allow per-run log directory)
        log_dir = self.config.get("log_dir", "logs")
        self.log_file = os.path.join(log_dir, f"{self.name}.csv")
        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        # Complete header for causal analysis
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "gt_x",
                    "gt_y",
                    "gt_z",  # Ground Truth
                    "gt_vx",
                    "gt_vy",
                    "gt_vz",
                    "meas_x",
                    "meas_y",
                    "meas_z",  # Sensors
                    "gnss_error_mag",
                    "ekf_x",
                    "ekf_y",
                    "ekf_z",
                    "ekf_pos_error_mag",
                    "wind_x",
                    "wind_y",
                    "wind_z",  # Environment
                    "wind_mag",
                    "rep_force_mag",  # Interaction
                    "nearest_neighbor_dist",
                    "target_x",
                    "target_y",
                    "target_z",  # Intent
                    "tracking_error_mag",
                    "collision_flag",  # Flags
                ]
            )

        # Journal de validation du filtre, a la cadence de controle.
        # Convention : erreur = ESTIME - VRAI. Colonnes identiques quel que soit
        # le filtre ; celles que le KF6 n'estime pas (attitude, biais) sont vides.
        xyz = ("x", "y", "z")
        self.filter_log = CsvBuffer(
            os.path.join(log_dir, f"{self.name}_filter.csv"),
            ["time"]
            + [f"e_p{a}" for a in xyz]
            + [f"e_v{a}" for a in xyz]
            + [f"e_r{a}" for a in xyz]
            + [f"e_ba{a}" for a in xyz]
            + [f"e_bg{a}" for a in xyz]
            + [f"sig_p{a}" for a in xyz]
            + [f"sig_v{a}" for a in xyz]
            + [f"sig_r{a}" for a in xyz]
            + [f"sig_ba{a}" for a in xyz]
            + [f"sig_bg{a}" for a in xyz]
            + [f"ba{a}" for a in xyz]
            + [f"bg{a}" for a in xyz]
            + [f"ba_true{a}" for a in xyz]
            + [f"bg_true{a}" for a in xyz]
            + ["nees", "nees_full", "nis", "gnss_update", "gnss_available", "gnss_err"],
        )
        # Journal de verite a la cadence de controle : permet de rejouer
        # n'importe quel filtre hors ligne sur la trajectoire reelle du vol
        # (cf. analysis/nav_replay.py et analysis/gnss_outage.py).
        self.truth_log = None
        if self.config.get("log_truth", True):
            self.truth_log = CsvBuffer(
                os.path.join(log_dir, f"{self.name}_truth.csv"),
                ["time", "px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw"],
            )

        if self.pub_socket is not None:
            self.broadcast_state(pos=self.start_pos, vel=[0, 0, 0])
        p.changeDynamics(self.bodyId, -1, linearDamping=0, angularDamping=0)

    # -----------------------------------------------------------------------
    # SWARM API
    # -----------------------------------------------------------------------
    def set_swarm_activate(self):
        """Activate swarm mode for the UAV."""
        self.swarm_active = True
        self.future_state = {"pos": np.array(self.start_pos), "vel": np.zeros(3), "yaw": 0.0}

    # -----------------------------------------------------------------------
    # Obstacle management and path planning
    # -----------------------------------------------------------------------
    def _log_filter_state(self, gt):
        """
        Journalise ce qu'il faut pour valider statistiquement le filtre :
        erreurs (estime - vrai), ecarts-types issus de P, biais estimes et vrais,
        NEES (position-vitesse, et 15 etats pour l'ESKF), NIS.

        Fichier : <log_dir>/<nom>_filter.csv  (separe du log principal, pour ne
        pas casser les scripts d'analyse existants).
        """
        buf = getattr(self, "filter_log", None)
        if buf is None:
            return

        def fmt(v):
            return "" if (v is None or not np.isfinite(v)) else f"{v:.6g}"

        nan3 = [np.nan] * 3
        ba_true = self.imu.accel_error_det
        bg_true = self.imu.gyro_error_det
        if isinstance(self.ekf, ESKF):
            # error_vector renvoie vrai - estime : on change le signe
            e = -self.ekf.error_vector(gt["pos"], gt["vel"], gt["orn_q"], ba_true, bg_true)
            sig = self.ekf.sigmas()
            nees_pv, nees_full = self.ekf.nees(gt["pos"], gt["vel"], gt["orn_q"], ba_true, bg_true)
            ba, bg = list(self.ekf.ba), list(self.ekf.bg)
        else:
            e = np.r_[np.asarray(self.ekf.x[:6]) - np.r_[gt["pos"], gt["vel"]], [np.nan] * 9]
            sig = np.r_[self.ekf.sigmas(), [np.nan] * 9]
            nees_pv, nees_full = self.ekf.nees(gt["pos"], gt["vel"]), np.nan
            ba, bg = nan3, nan3
        nis = self.ekf.last_nis if self._gnss_updated else np.nan

        buf.write(
            [f"{self._sim_time:.6f}"]
            + [fmt(v) for v in e]
            + [fmt(v) for v in sig]
            + [fmt(v) for v in ba]
            + [fmt(v) for v in bg]
            + [fmt(v) for v in ba_true]
            + [fmt(v) for v in bg_true]
            + [
                fmt(nees_pv),
                fmt(nees_full),
                fmt(nis),
                int(self._gnss_updated),
                int(self.gnss.available),
                # erreur GNSS a l'instant de mesure (reference honnete du gain)
                fmt(self._gnss_err) if self._gnss_updated else "",
            ]
        )

    def _log_truth(self, gt):
        if self.truth_log is None:
            return
        q = Q.normalize(gt["orn_q"])
        self.truth_log.write(
            [f"{self._sim_time:.6f}"]
            + [f"{v:.9g}" for v in gt["pos"]]
            + [f"{v:.9g}" for v in gt["vel"]]
            + [f"{v:.12g}" for v in q]
        )

    def close_logs(self):
        """Ecrit les lignes encore en memoire. Appele par SimulationManager.stop()."""
        for buf in (getattr(self, "filter_log", None), getattr(self, "truth_log", None)):
            if buf is not None:
                buf.close()

    def _limit_tilt_demand(self, virtual_target_pos, final_target_vel, pos, vel):
        """
        Reduit la consigne horizontale pour que l'assiette commandee au
        controleur reste sous `self.max_tilt_deg`.

        Le controleur DSL forme sa demande d'effort comme
            target_thrust = P * pos_e + I * integrale + D * vel_e + [0, 0, m*g]
        et en deduit son axe de poussee. L'inclinaison commandee vaut donc
            theta = atan(|target_thrust_xy| / (m*g)).
        On impose |target_thrust_xy| <= tan(max_tilt) * m*g, en reduisant d'un
        meme facteur l'erreur de position et l'erreur de vitesse horizontales :
        la direction de la consigne est preservee, seule son amplitude est
        bornee. La consigne verticale n'est pas touchee.

        Terme integral
        --------------
        Le terme integral n'est pas une marge a provisionner : c'est une partie
        de la demande, qu'on lit directement dans l'etat du controleur. Sans
        precaution il s'emballe, car la cible virtuelle est placee 1 m devant le
        drone pendant toute la croisiere : l'erreur de position reste d'1 m,
        l'integrale sature (+/-2, soit 0.1 N par axe) et continue de pousser
        dans la direction precedente au moment d'un virage. On l'empeche donc de
        depasser une fraction `integral_share` de la demande admissible
        (anti-emballement), puis on borne la demande TOTALE.

        On ne retire PAS la valeur maximale de l'integrale a la limite : P et D
        seraient alors plafonnes a ~3 deg d'inclinaison et le drone, pilote par
        la seule integrale, raterait ses virages.

        Returns:
            (virtual_target_pos, final_target_vel) corriges.
        """
        P = np.asarray(self.ctrl.P_COEFF_FOR, dtype=float)
        D = np.asarray(self.ctrl.D_COEFF_FOR, dtype=float)
        I = np.asarray(self.ctrl.I_COEFF_FOR, dtype=float)
        weight = float(self.ctrl.GRAVITY)  # m*g en N, convention DSL
        integ = getattr(self.ctrl, "integral_pos_e", None)

        pos_e = np.asarray(virtual_target_pos, dtype=float) - np.asarray(pos, dtype=float)
        vel_e = np.asarray(final_target_vel, dtype=float) - np.asarray(vel, dtype=float)
        pos_e = pos_e.copy()
        vel_e = vel_e.copy()

        # --- 1. Demande VERTICALE bornee : la poussee doit toujours pointer vers le haut ---
        # Si la composante verticale de target_thrust devient negative (freinage
        # d'une montee rapide : D_z * (0 - 3 m/s) = -1.5 N contre un poids de
        # 0.26 N), l'axe de poussee commande pointe vers le BAS : le controleur
        # commande un retournement. On garde la demande verticale dans
        # [min_thrust_ratio, max_thrust_ratio] x m*g.
        cz = float(I[2] * integ[2]) if integ is not None else 0.0
        uz = float(P[2] * pos_e[2] + D[2] * vel_e[2])
        tz_min = self.min_thrust_ratio * weight
        tz_max = self.max_thrust_ratio * weight
        tz = uz + cz + weight
        if tz < tz_min and uz < 0.0:
            kz = float(np.clip((tz_min - weight - cz) / uz, 0.0, 1.0))
            pos_e[2] *= kz
            vel_e[2] *= kz
        elif tz > tz_max and uz > 0.0:
            kz = float(np.clip((tz_max - weight - cz) / uz, 0.0, 1.0))
            pos_e[2] *= kz
            vel_e[2] *= kz
        tz = float(P[2] * pos_e[2] + D[2] * vel_e[2]) + cz + weight

        # --- 1b. Vitesse de descente bornee, plus severement pres du sol ---
        # Avec tz_min = 0.3 x poids, le drone pouvait se laisser tomber a 0.7 g
        # quand l'altitude de consigne baissait (formation qui redescend) : un
        # suiveur est tombe de 0.5 m en 0.25 s et a touche le sol en se
        # deplacant, puis s'est retourne. Reduire la demande ne suffit pas a
        # freiner une chute : on IMPOSE une demande minimale en agissant sur la
        # consigne de vitesse verticale (DSL recalcule vel_e = cible - vitesse).
        gf = getattr(self, "_ground_factor", 1.0)
        v_down_max = self.ground_descent_speed + (self.max_descent_speed - self.ground_descent_speed) * gf
        tz_floor = tz_min
        # pres du sol, poussee quasi stationnaire au minimum (pas de chute)
        tz_floor = max(
            tz_floor, weight * (self.min_thrust_ratio + (0.9 - self.min_thrust_ratio) * (1.0 - gf))
        )
        if float(vel[2]) < -v_down_max:
            tz_floor = max(tz_floor, 1.15 * weight)  # descente trop rapide : freiner
        # Montee bornee de meme : une montee rapide se paie au sommet par un
        # freinage a poussee minimale, la ou le controle d'attitude est le plus faible.
        if float(vel[2]) > self.max_climb_speed and tz > weight:
            uz_cap = 0.0 - cz  # plus d'acceleration vers le haut
            if D[2] > 1e-9:
                vel_e[2] = (uz_cap - P[2] * pos_e[2]) / D[2]
            tz = weight
        if tz < tz_floor and D[2] > 1e-9:
            uz_new = tz_floor - weight - cz
            vel_e[2] = (uz_new - P[2] * pos_e[2]) / D[2]
            tz = tz_floor
        tz = max(tz, tz_min)

        # --- 2. Limite horizontale, relative a la demande verticale REELLE ---
        # L'inclinaison vaut atan(|horizontal| / vertical) : a demande verticale
        # reduite, la meme demande horizontale incline davantage.
        # Inclinaison admissible reduite pres du sol (garde-fou 6)
        gf = getattr(self, "_ground_factor", 1.0)
        tilt_deg = self.ground_tilt_deg + (self.max_tilt_deg - self.ground_tilt_deg) * gf
        limit = float(np.tan(np.radians(tilt_deg)) * tz)

        # --- 3. Anti-emballement de l'integrale horizontale du controleur ---
        c = np.zeros(2)
        if integ is not None:
            c = I[:2] * integ[:2]
            c_max = self.integral_share * limit
            nc = float(np.linalg.norm(c))
            if nc > c_max and nc > 1e-12:
                integ[:2] *= c_max / nc  # modifie l'etat du controleur
                c = I[:2] * integ[:2]

        u = P[:2] * pos_e[:2] + D[:2] * vel_e[:2]

        # --- 4. Plus grand k dans [0, 1] tel que |k u + c| <= limit ---
        if float(np.linalg.norm(u + c)) > limit:
            uu, uc, cc = float(u @ u), float(u @ c), float(c @ c)
            if uu < 1e-18:
                k = 0.0
            else:
                disc = uc * uc - uu * (cc - limit * limit)
                k = (-uc + np.sqrt(max(disc, 0.0))) / uu
                k = float(np.clip(k, 0.0, 1.0))
            pos_e[:2] *= k
            vel_e[:2] *= k

        virtual_target_pos = np.asarray(pos, dtype=float) + pos_e
        final_target_vel = np.asarray(vel, dtype=float) + vel_e
        return virtual_target_pos, final_target_vel

    def _trigger_planning(self, start_pos, target_pos):
        """
        Initiates an asynchronous path planning thread using the A* algorithm.

        Creates and starts a daemon thread that runs the async planning routine if one is not already
        in progress. This method prevents concurrent planning operations by checking the `is_planning` flag.

        Args:
            start_pos: The starting position for path planning (coordinates or position object).
            target_pos: The target/goal position for path planning (coordinates or position object).

        Returns:
            None

        Side Effects:
            - Sets `self.is_planning` to True when a new planning thread is started.
            - Creates and starts a daemon thread stored in `self.planning_thread`.
            - Prints a status message indicating planning has started.

        Notes:
            - This method is designed to be non-blocking; actual planning happens in a separate thread.
            - The planning thread is set as a daemon, so it won't prevent program termination.
            - Subsequent calls while `is_planning` is True will be ignored.
        """
        if self.is_planning:
            return
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        # Trajet purement vertical (decollage, montee sur place) : A* travaille
        # sur une grille 2D, depart et arrivee tombent dans la meme cellule et il
        # renvoie None. Ce n'etait pas un echec mais c'etait compte comme tel :
        # apres 6 "echecs" (7.5 s), le waypoint etait saute en pleine montee.
        if float(np.linalg.norm(target_pos[:2] - start_pos[:2])) < self.direct_leg_xy:
            self.active_path = [target_pos.copy()]
            self._plan_wp_idx = self.wp_idx
            self.calculation_fail_count = 0
            return

        self._planning_for_wp = self.wp_idx
        self.is_planning = True
        if self.planning_sync:
            # Mode deterministe (Monte-Carlo) : a graine egale, vol identique.
            # En mode fil d'execution, le moment ou le plan arrive depend du
            # temps de calcul reel, donc de la charge de la machine.
            self._run_async_plan(start_pos, target_pos)
            return
        print(f"[{self.name}] ⏳ Starting A* Thread...")
        self.planning_thread = threading.Thread(target=self._run_async_plan, args=(start_pos, target_pos))
        self.planning_thread.daemon = True
        self.planning_thread.start()

    def _run_async_plan(self, start_pos, target_pos):
        """
        Execute asynchronous path planning from start to target position.
        Attempts to compute a path using the planner. Updates active_path if successful,
        otherwise increments failure counter. Sets is_planning flag to False upon completion.
        :param start_pos: Starting position coordinates
        :param target_pos: Target position coordinates
        """
        try:
            path = self.planner.plan(start_pos, target_pos)
            # Un plan calcule pour un waypoint deja depasse est jete.
            if path and len(path) > 0 and self._planning_for_wp == self.wp_idx:
                self.active_path = [np.asarray(w, dtype=float) for w in path]
                self._plan_wp_idx = self.wp_idx
                self.calculation_fail_count = 0
            elif not path:
                self.calculation_fail_count += 1
        except Exception as e:
            # Une exception est un echec : elle doit compter, sinon un
            # planificateur defaillant est relance indefiniment.
            self.calculation_fail_count += 1
            print(f"[{self.name}] 💥 Error in A* thread: {e}")
        finally:
            self.is_planning = False

    def _compute_repulsive_force(self, current_pos):
        """
        Compute repulsive force from obstacles and other agents.

        Combines repulsive forces from:
        - Other UAVs: Inverse-distance force within safety radius
        - Static obstacles: AABB-based collision avoidance using spatial indexing

        Returns normalized force vector capped at max_repulsive_force magnitude.

        Args:
            current_pos (np.ndarray): Current 3D position [x, y, z]

        Returns:
            np.ndarray: Repulsive force vector [fx, fy, fz] in Newtons
        """
        force_vec = np.array([0.0, 0.0, 0.0])
        min_dist = np.inf

        # Check other agents
        if self.other_agent_pos != {}:
            for _, other_pos in self.other_agent_pos.items():
                diff = current_pos - other_pos
                dist_uav = np.linalg.norm(diff)
                if dist_uav < min_dist:
                    min_dist = dist_uav
                if dist_uav < self.safety_radius:
                    mag = 1.0 - (dist_uav / self.safety_radius)
                    force_vec += ((diff / dist_uav) * mag * self.max_repulsive_force) / 2

        # Check if planner and its index are ready
        if self.planner.building_tree is None:
            total_norm = np.linalg.norm(force_vec)
            if total_norm > self.max_repulsive_force:
                force_vec = (force_vec / total_norm) * self.max_repulsive_force
            return force_vec

        # Find indices of nearby buildings (e.g., 15m radius)
        indices = self.planner.building_tree.query_ball_point(current_pos[:2], r=15.0)

        for idx in indices:
            obs = self.obs_dic[idx]
            center = np.array(obs["center"])
            h, w, l = obs["height"], obs["width"], obs["length"]

            # Skip if UAV is above building
            if current_pos[2] > h + 1.0:
                continue

            # Define axis-aligned bounding box (AABB)
            min_x = center[0] - l / 2
            max_x = center[0] + l / 2
            min_y = center[1] - w / 2
            max_y = center[1] + w / 2

            # Find closest point on or in the rectangle
            # Clamp drone position between building bounds
            closest_x = max(min_x, min(current_pos[0], max_x))
            closest_y = max(min_y, min(current_pos[1], max_y))
            closest_pt = np.array([closest_x, closest_y])

            # Calculate distance vector
            diff = current_pos[:2] - closest_pt
            dist = np.linalg.norm(diff)

            # Special case: if drone is exactly inside (dist ~ 0)
            # create a force to push it out
            if dist < 0.01:
                # Can ignore or push towards nearest edge
                continue

            # Apply repulsive force
            if dist < self.safety_radius:
                mag = 1.0 - (dist / self.safety_radius)
                # diff / dist vector is now perpendicular to wall
                force_vec[:2] += (diff / dist) * mag * self.max_repulsive_force

        # Final normalization
        total_norm = np.linalg.norm(force_vec)
        if total_norm > self.max_repulsive_force:
            force_vec = (force_vec / total_norm) * self.max_repulsive_force

        self.last_repulsive_force_mag = total_norm
        self.dist_to_nearest_neighbor = min_dist

        return force_vec

    # -----------------------------------------------------------------------
    # COMMUNICATION
    # -----------------------------------------------------------------------
    def setup_network_swarm(self, ip, port_pub_swarm, port_sub_swarm):

        # Defensive close if re-running multiple simulations in the same process
        # (e.g., ablation suite). On Windows, stale sockets can keep ports busy.
        for attr in ("sub_socket", "pub_socket"):
            sock = getattr(self, attr, None)
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass

        # IMPORTANT: In this architecture, the Swarm proxy thread binds the ports
        # (XSUB/XPUB). UAVs must CONNECT (not bind), otherwise you'll hit
        # EACCES/"Permission denied" on Windows when ports are already in use.
        self.sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{ip}:{port_sub_swarm}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1)
        try:
            self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error:
            pass

        self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://{ip}:{port_pub_swarm}")
        self.pub_socket.setsockopt(zmq.LINGER, 0)

    def radar_com_setup(self, radars_list):
        """
        Setup ZMQ socket for radar communication.

        Connects to one or more radar sources specified in the configuration.
        Creates a SUB socket with non-blocking mode and optional conflation.

        Args:
            radars_list (list[dict]): List of radar configurations, each with:
            - ip (str): Radar server IP address
            - port (int): Radar server port number

        Returns:
            None
        """
        self.radar_sub_socket = self.zmq_ctx.socket(zmq.SUB)
        self.radar_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.radar_sub_socket.setsockopt(zmq.RCVTIMEO, 1)
        self.radar_message_buffer = []
        try:
            self.radar_sub_socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.Error:
            pass

        for radar_info in radars_list:
            # Get info from config.yaml
            ip = radar_info.get('ip', 'localhost')
            port = radar_info.get('port')

            if port:
                address = f"tcp://{ip}:{port}"
                print(f"[{self.name}] Connecting to radar defined in config: {address}")
                self.radar_sub_socket.connect(address)
            else:
                print(f"[{self.name}] ⚠️ Error: Radar port not specified in config.")

    def broadcast_state(self, pos, vel):
        """
        Broadcast current UAV state to swarm network.
        Sends position, velocity, yaw, and simulation time as JSON via ZMQ pub socket.
        Rounds values to 3 decimal places for network efficiency.
        """
        if getattr(self, "pub_socket", None) is None:
            return
        pos = [round(p, 3) for p in pos]
        vel = [round(v, 3) for v in vel]
        msg = {
            "name": self.name,
            "pos": pos,
            "vel": vel,
            "yaw": round(self.target_yaw_cache, 3),
            "sim_time": round(self._sim_time, 3),
        }
        self.pub_socket.send_string("State " + json.dumps(msg))

    def listen_radar(self):
        """
        Process incoming radar messages with simulated perception delay.
        Buffers messages with random delay to simulate network latency, then processes them
        when their target time is reached. Extracts detected agent positions and updates
        neighbor tracking for collision avoidance and swarm coordination.
        """
        while True:
            try:
                # Non-blocking read
                msg = self.radar_sub_socket.recv_string()
                delay = max(0, random.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self._sim_time + delay
                self.radar_message_buffer.append((visible_time, msg))
            except zmq.Again:
                # No more messages
                break
            except Exception as e:
                print(f"Network error on {self.name}: {e}")
                break

        buffer_remaining = []

        for target_time, msg in self.radar_message_buffer:
            if self._sim_time >= target_time:
                # --- MESSAGE IS READY: PROCESS IT ---
                if " " in msg:
                    _, json_str = msg.split(" ", 1)
                    try:
                        data = json.loads(json_str)
                        radar_data = data.get("data", {})
                        for d_name, d_info in radar_data.items():
                            self.neighbors_data[d_name] = d_info

                            if d_name != self.name:
                                if not self.leader:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                                elif d_name not in self.swarm_name:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                            if d_name == self.name:
                                self.radar_reports = d_info
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
                # --- NOT READY YET: KEEP IT ---

        # Replace buffer with remaining messages
        self.radar_message_buffer = buffer_remaining

    def listen_swarm(self):
        """
        Process incoming swarm messages with simulated perception delay.
        Buffers messages with random delay to simulate latency, then processes them
        when their target time is reached. Handles SWARM (neighbor updates) and
        FUTURE_POS (state predictions) message types.
        """
        while True:
            try:
                # Non-blocking read
                msg = self.sub_socket.recv_string()
                delay = max(0, random.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self._sim_time + delay
                self.message_buffer.append((visible_time, msg))
            except zmq.Again:
                # No more messages
                break
            except Exception as e:
                print(f"Network error on {self.name}: {e}")
                break

        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self._sim_time >= target_time:
                # --- MESSAGE IS READY: PROCESS IT ---
                if " " in msg:
                    topic, json_str = msg.split(" ", 1)
                    try:
                        if topic == "SWARM":
                            data = json.loads(json_str)
                            for d_name, d_info in data.items():
                                self.neighbors_data[d_name] = d_info
                                if d_name != self.name and not self.leader:
                                    pos = d_info["pos"]
                                    self.other_agent_pos[d_name] = np.array(pos)
                        elif topic == "FUTURE_POS" and self.swarm_active and not self.leader:
                            state = json.loads(json_str)
                            mine = state.get(self.name)
                            if mine is not None:  # jamais ecraser par None
                                self.future_state = mine
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
                # --- NOT READY YET: KEEP IT ---

        # Replace buffer with remaining messages
        self.message_buffer = buffer_remaining

    # -----------------------------------------------------------------------
    # MAIN LOOP & LOGIC
    # -----------------------------------------------------------------------
    def think_and_act(self):
        """
        Perform a single simulation tick: update time, wind, control logic (100Hz), physics (240Hz), and optionally log state.
        """
        if not p.isConnected(self.physics_client_id):
            return

        # 1. Horloge : compteur entier de pas physiques. Un temps flottant
        # cumule (t += dt) puis compare a un seuil produit des cadences
        # irregulieres ; le compteur garantit un controle tous les
        # `ctrl_every` pas exactement.
        self._tick += 1
        self._sim_time = self._tick * self.dt

        # Vent : la vitesse air oriente la composante longitudinale des rafales
        gt = self.get_ground_truth_state()
        h = gt["pos"][2]
        # Vitesse air par rapport au vent MOYEN (convection de la turbulence
        # figee). Avec le vent instantane, la direction de la rafale suivrait la
        # rafale elle-meme et tournerait au hasard a chaque pas : le vent
        # deviendrait un bruit blanc sans effet sur le vol.
        v_air = np.asarray(gt["vel"], dtype=float) - self.wind_module.mean_wind
        self.current_wind = self.wind_module.step(h, float(np.linalg.norm(v_air)), v_air, t=self._sim_time)

        # 2. Boucle de controle, tous les ctrl_every pas
        if (self._tick - 1) % self.ctrl_every == 0:  # pas 1, 1+n, 1+2n...
            self._update_control_loop(gt)
            self.last_ctrl_time = self._sim_time

        # 3. Physique, a chaque pas
        self._apply_lib_physics(self.last_rpms, gt)

        # Journal principal tous les 10 pas physiques (cadence reguliere)
        if self._tick % 10 == 0:
            self._log_full_state(gt)

    def _update_control_loop(self, gt):
        """
        High-Level Control Loop (100 Hz).
        Orchestrates sensor fusion, state estimation, communication, planning, and motor control.

        **Sensor Fusion:**
        - Processes noisy GNSS measurements with jittered update intervals
        - EKF prediction using IMU acceleration and orientation
        - Maintains corrected position and velocity estimates

        **Communication:**
        - Broadcasts state to swarm at regular intervals
        - Listens for swarm and radar messages

        **Target Logic:**
        - Swarm Mode: Follows leader's predicted state
        - Planning Mode: Decelerates during path calculation
        - Autonomous Mode: Navigates waypoints via A* with replanning and failsafe

        **Collision Avoidance:**
        - Computes repulsive forces from obstacles and neighbors

        **Control:**
        - Generates motor RPMs via PID controller with target position, velocity, and yaw

        Args:
            gt (dict): Ground truth state with pos, vel, orn_q, ang_vel

        Returns:
            None (updates self.last_rpms)
        """

        true_orn_q = np.array(gt["orn_q"])  # attitude PHYSIQUE (surveillance)
        ang_vel = np.array(gt["ang_vel"])

        # ==================== ADVANCED SENSOR FUSION ====================

        # ORDRE DU FILTRE : predict PUIS update, et on publie l'etat CORRIGE
        # (corriger avant de propager appliquerait la mesure a une prediction
        # perimee).

        # 1. Lecture de l'IMU (acceleration + vitesse angulaire)
        # Le pas reel entre deux appels est passe explicitement : la boucle est
        # cadencee par `_sim_time`, qui avance par multiples du dt physique,
        # donc l'intervalle effectif n'est pas exactement CTRL_DT. Comme
        # acc = dv/dt, utiliser un dt errone biaise directement l'acceleration.
        imu_dt = self._sim_time - self.last_ctrl_time
        self._ctrl_elapsed = imu_dt
        imu_acc, imu_gyro = self.imu.measure(gt["vel"], gt["orn_q"], ang_vel=gt["ang_vel"], dt=imu_dt)
        self.last_imu_gyro = imu_gyro

        # 2. Propagation inertielle
        #  - ESKF : propage sa PROPRE attitude a partir du gyrometre ;
        #  - KF6  : n'estime pas l'attitude, on lui fournit l'attitude vraie a
        #           mi-intervalle (celle avec laquelle l'IMU a projete).
        if self.filter_type == "eskf":
            self.ekf.predict(imu_acc, imu_gyro, dt=imu_dt)
        else:
            self.ekf.predict(imu_acc, self.imu.q_mid, dt=imu_dt)

        # 3. Correction GNSS (cadence propre, avec gigue)
        # NB : meas_* dans les logs sont les sorties GNSS brutes, l'estimee du
        # filtre est journalisee separement. Pendant une coupure, le recepteur
        # ne delivre rien : on garde la derniere mesure pour les logs et le
        # filtre poursuit en inertie pure.
        self._gnss_updated = False
        if self._sim_time >= self.next_gnss_trigger:
            meas_pos, meas_vel = self.gnss.measure(gt["pos"], gt["vel"], t=self._sim_time)
            if meas_pos is not None:
                self.last_gnss_meas_pos = np.array(meas_pos, dtype=np.float32)
                self.last_gnss_meas_vel = np.array(meas_vel, dtype=np.float32)
                # Precision rapportee par le recepteur (brouillage compris)
                self.ekf.update(
                    meas_pos, meas_vel, pos_std=self.gnss.last_pos_std, vel_std=self.gnss.last_vel_std
                )
                self._gnss_updated = True
                self._gnss_err = float(
                    np.linalg.norm(np.asarray(meas_pos, dtype=float) - np.asarray(gt["pos"], dtype=float))
                )

            # Prochaine echeance : periode nominale + gigue positive (sans derive)
            jitter = max(0.0, random.gauss(self.gnss_delay_mean, self.gnss_delay_std))
            # Echeancier nominal strictement periodique : la gigue decale CHAQUE
            # mesure sans s'accumuler. (Le max(..., t) rattrapait l'instant de
            # mesure, gigue comprise : le recepteur tournait a 9.1 Hz au lieu de 10.)
            self.last_gnss_nominal_time += self.gnss_dt
            self.next_gnss_trigger = self.last_gnss_nominal_time + jitter

        # 4. Etat estime publie (utilise par le controle et la navigation)
        pos = np.array(self.ekf.position, dtype=float)
        vel = np.array(self.ekf.velocity, dtype=float)

        # Attitude fournie a la boucle interne
        if self.attitude_source == "filter":
            orn_q = np.array(self.ekf.attitude, dtype=float)
        else:
            orn_q = true_orn_q
        rpy = np.array(p.getEulerFromQuaternion(orn_q))

        # 5. Journaux de validation (filtre, et verite pour le rejeu hors ligne)
        self._log_filter_state(gt)
        self._log_truth(gt)

        # --- COMMUNICATION ---
        if self.swarm_active or self.leader:
            if (self._sim_time - self.last_com_time) >= self.com_period:
                self.last_com_time = self._sim_time
                self.broadcast_state(pos, vel)

        # Receive Messages
        if self.sub_socket is not None:
            self.listen_swarm()
        if self.radar_sub_socket is not None:
            self.listen_radar()

        # --- TARGET LOGIC ---
        target_pos = pos
        target_vel = np.zeros(3)
        if not self.is_planning and self.wp_idx < len(self.waypoints):
            self._hold_pos = None  # point de maintien reinitialise en route

        # 1. Suiveur d'essaim : position ET vitesse de consigne de la formation.
        # La vitesse sert d'anticipation : sans elle le suiveur poursuit une
        # cible qui avance, avec un retard permanent.
        if self.swarm_active and not self.leader:
            fs = self.future_state or {}
            if fs.get("pos") is not None:
                target_pos = np.asarray(fs["pos"], dtype=float)
            if fs.get("vel") is not None:
                target_vel = np.asarray(fs["vel"], dtype=float)
                # Le message arrive avec ~0.1 s de retard : on extrapole la cible.
                if fs.get("t") is not None:
                    age = float(np.clip(self._sim_time - float(fs["t"]), 0.0, 0.5))
                    target_pos = target_pos + target_vel * age

        # 2. Planning (Wait)
        elif self.is_planning:
            # Attente du plan : on freine vers le point ou la planification a
            # commence (point fixe, pas `pos` qui annulerait le terme P).
            if self._hold_pos is None:
                self._hold_pos = np.array(pos, dtype=float)
            target_pos = self._hold_pos

        # 3. Autonomous Navigation
        else:
            # --- FAILSAFE CHECK ---
            if self.calculation_fail_count > 5:
                print(f"[{self.name}] ⚠️ Too many A* failures ({self.calculation_fail_count}). Skipping WP.")
                self.wp_idx += 1
                self.calculation_fail_count = 0
                self.replan_timer = 0
                return  # Skip this cycle to reset logic
            # ----------------------

            else:
                # Detect arrival at Waypoint
                if self.wp_idx < len(self.waypoints):
                    dist_wp = np.linalg.norm(self.waypoints[self.wp_idx] - pos)

                    # --- GARDE-FOU 5 : critere d'arrivee avec capture ---------
                    # Le seul test `dist < 0.5 m` est fragile : un drone qui
                    # arrive trop vite, ou dont l'assiette est bornee, decrit une
                    # orbite autour du waypoint sans jamais entrer dans la
                    # tolerance. Il reste alors bloque sur ce waypoint pour toute
                    # la simulation, sans message. On accepte donc l'arrivee aussi
                    # lorsque le waypoint a ete approche puis depasse : on memorise
                    # la distance minimale atteinte et on valide des qu'on s'en
                    # eloigne de nouveau. C'est la logique de sequencement usuelle
                    # en guidage.
                    if self.wp_idx != self._wp_track_idx:
                        self._wp_track_idx = self.wp_idx
                        self._wp_min_dist = np.inf
                    self._wp_min_dist = min(self._wp_min_dist, dist_wp)

                    reached = dist_wp < self.wp_tol
                    passed = (
                        self._wp_min_dist < self.wp_capture and dist_wp > self._wp_min_dist + self.wp_hyst
                    )

                    if (reached or passed) and not self.is_planning:
                        why = "reached" if reached else f"passed (closest {self._wp_min_dist:.2f} m)"
                        print(f"[{self.name}] Waypoint {self.wp_idx} {why}.")
                        self.wp_idx += 1
                        self._wp_min_dist = np.inf
                        self.active_path = []  # Force a new calculation
                        self.replan_timer = 0
                        self.calculation_fail_count = 0  # compteur propre a chaque WP

                # Planification : une fois par waypoint. Quand le chemin A* a ete
                # entierement parcouru, il reste moins de 0.7 m jusqu'au waypoint :
                # on y va en ligne droite (relancer A* si pres du but ferait
                # freiner le drone a chaque waypoint).
                if (
                    self.wp_idx < len(self.waypoints)
                    and len(self.active_path) == 0
                    and self._plan_wp_idx != self.wp_idx
                    and self.replan_timer <= 0
                ):
                    self._trigger_planning(pos, self.waypoints[self.wp_idx])
                    self.replan_timer = self.replan_ticks

            # Follow Path
            if len(self.active_path) > 0:
                local_target = self.active_path[0]
                if np.linalg.norm(local_target - pos) < 0.7:
                    self.active_path.pop(0)
                    if len(self.active_path) > 0:
                        local_target = self.active_path[0]
                    elif self.wp_idx < len(self.waypoints):
                        local_target = self.waypoints[self.wp_idx]
                target_pos = local_target

            elif self.wp_idx < len(self.waypoints):
                # Pas (ou plus) de chemin : ligne droite vers le waypoint. Le
                # sequencement est gere en un seul endroit (garde-fou 5) ; le
                # second test d'arrivee qui existait ici pouvait faire sauter
                # deux waypoints proches dans le meme pas.
                target_pos = self.waypoints[self.wp_idx]
            else:
                # Mission terminee : maintien sur un point FIXE. Viser `pos` a
                # chaque pas annule le terme proportionnel : le drone ne tenait
                # plus que sur l'amortissement en vitesse et derivait au vent.
                if self._hold_pos is None:
                    self._hold_pos = np.array(pos, dtype=float)
                target_pos = self._hold_pos

        if self.replan_timer > 0:
            self.replan_timer -= 1

        # --- CONTROL COMMANDS ---
        # (voir _limit_tilt_demand plus bas pour la limitation d'assiette)
        self.current_target_pos = target_pos

        # Repulsive Force
        if self.environment == "generated":
            f_rep = self._compute_repulsive_force(pos)
        elif self.environment == "custom":
            f_rep = self.planner.compute_repulsive_force(
                pos,
                self.safety_radius,
                self.max_repulsive_force,
                self.swarm_active,
                self.leader,
                self.other_agent_pos,
            )
            self.last_repulsive_force_mag = float(np.linalg.norm(f_rep))
        else:
            f_rep = np.zeros(3)

        # La "force" repulsive est convertie en increment de vitesse de consigne
        # par un gain explicite [s], independant de la frequence de controle.
        acc_rep = f_rep / self.mass
        final_target_vel = np.asarray(target_vel, dtype=float) + acc_rep * self.repulsion_gain_s

        # --- GARDE-FOU 6 : vol pres du sol ---------------------------------
        # Au decollage, les suiveurs recoivent une cible qui part deja a la
        # vitesse du leader. Ils acceleraient a l'horizontale (3 a 5 m/s) a
        # 0.2-0.9 m d'altitude, inclines a 30 deg : un rotor touchait le sol et
        # le drone basculait. C'etait la cause de TOUS les retournements
        # observes (8 sur 72 vols, toujours un suiveur, entre 1.4 et 1.8 s).
        # Sous `ground_clear_alt`, la vitesse horizontale et l'inclinaison
        # admissibles sont reduites progressivement.
        self._ground_factor = float(
            np.clip(
                (float(pos[2]) - self.ground_min_alt)
                / max(self.ground_clear_alt - self.ground_min_alt, 1e-3),
                0.0,
                1.0,
            )
        )
        vmax_xy = self.max_speed * (
            self.ground_speed_floor + (1.0 - self.ground_speed_floor) * self._ground_factor
        )

        # Clamp Speed
        speed_xy = np.linalg.norm(final_target_vel[:2])
        if speed_xy > vmax_xy:
            ratio = vmax_xy / speed_xy
            final_target_vel[:2] *= ratio

        # PID Target Helper
        final_target_pos = target_pos + (final_target_vel * self.CTRL_DT)
        vector_to_target = final_target_pos - pos
        dist_to_target = np.linalg.norm(vector_to_target)
        if dist_to_target > self.lookahead_m:
            # Cible virtuelle a distance bornee, continue en fonction de la distance.
            virtual_target_pos = pos + (vector_to_target / dist_to_target) * self.lookahead_m
        else:
            virtual_target_pos = final_target_pos

        # --- GARDE-FOU 3 : limitation de l'assiette commandee -----------------
        # DSLPIDControl construit son axe de poussee a partir de
        #     target_thrust = P*pos_e + I*integrale + D*vel_e + [0, 0, m*g]
        # L'inclinaison commandee vaut donc atan(|composante horizontale| / m*g).
        # Avec les gains CF2X (P_xy=0.4, D_xy=0.2) et m*g = 0.265 N, une erreur
        # de 1 m plus une consigne de 5 m/s demandent 1.40 N d'effort horizontal,
        # soit 79 deg d'inclinaison : une rafale ou une force repulsive suffit
        # alors a franchir 90 deg. Or au-dela de 90 deg le drone est irrecuperable :
        # DSLPIDControl calcule scalar_thrust = max(0, target_thrust . z_corps),
        # qui devient nul des que l'axe corps pointe vers le bas. La poussee tombe
        # au minimum, le drone reste colle au sol jusqu'a la fin de la simulation
        # et aucun message n'est emis. C'etait la cause des drones bloques au sol.
        # On borne donc la demande horizontale pour rester sous max_tilt_deg.
        virtual_target_pos, final_target_vel = self._limit_tilt_demand(
            virtual_target_pos, final_target_vel, pos, vel
        )

        # --- GARDE-FOU 4 : detection de perte de controle ---------------------
        # Si malgre tout le drone s'est retourne, on le signale une fois : sans
        # cela l'anomalie est totalement silencieuse dans les logs.
        # Attitude PHYSIQUE (verite), pas l'estimee : on surveille le drone reel.
        tilt = float(
            np.degrees(
                np.arccos(
                    np.clip(np.array(p.getMatrixFromQuaternion(true_orn_q)).reshape(3, 3)[2, 2], -1.0, 1.0)
                )
            )
        )
        if tilt > 90.0 and not self._loss_of_control:
            self._loss_of_control = True
            print(
                f"[{self.name}] PERTE DE CONTROLE a t={self._sim_time:.2f}s : "
                f"inclinaison {tilt:.0f} deg (> 90). La poussee commandee "
                f"s'annule, le drone ne peut plus se redresser."
            )

        # Yaw
        direction_vec = final_target_pos - pos
        if (
            self.swarm_active
            and not self.leader
            and self.future_state
            and self.future_state.get("yaw") is not None
        ):
            self.target_yaw_cache = float(self.future_state["yaw"])
        elif np.linalg.norm(direction_vec[:2]) > 0.5:
            self.target_yaw_cache = np.arctan2(direction_vec[1], direction_vec[0])

        # --- GARDE-FOU 7 : consigne de lacet a vitesse bornee ----------------
        # La consigne de lacet sautait d'un coup (vers le waypoint suivant, ou au
        # lacet de la formation) : 176 deg d'un pas sur un cas mesure. Le PID de
        # DSL convertit une telle marche en couple de lacet maximal ; les quatre
        # moteurs saturent (21 666 tr/min), il ne reste plus de marge pour le
        # roulis et le tangage, et l'inclinaison derive librement (~100 deg/s)
        # jusqu'au retournement. On fait donc tourner la consigne vers le lacet
        # desire a vitesse limitee.
        dyaw = float(
            np.arctan2(
                np.sin(self.target_yaw_cache - self._yaw_cmd), np.cos(self.target_yaw_cache - self._yaw_cmd)
            )
        )
        max_step = self.max_yaw_rate * self._ctrl_elapsed
        self._yaw_cmd = float(
            np.arctan2(
                np.sin(self._yaw_cmd + np.clip(dyaw, -max_step, max_step)),
                np.cos(self._yaw_cmd + np.clip(dyaw, -max_step, max_step)),
            )
        )

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])

        # Compute RPMs (PID)
        rpms, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self._ctrl_elapsed,  # pas REEL depuis le dernier appel
            state=state_vec,
            target_pos=virtual_target_pos,
            target_vel=final_target_vel,
            target_rpy=np.array([0, 0, self._yaw_cmd]),
        )

        self.last_rpms = rpms

    def _apply_lib_physics(self, rpms, gt):
        """
        Apply physics simulation to the UAV using rotor RPM values.
        Converts RPM to thrust forces and torques, applies them to the quadrotor,
        and simulates aerodynamic drag accounting for wind effects.

        Parameters
        ----------
        rpms : array-like
            Rotational speeds (RPM) of the four rotors, shape (4,)
        gt : dict
            Ground truth state with 'orn_q' (quaternion) and 'vel' (velocity)
        """
        rpms = np.clip(rpms, 0, self.MAX_RPM)
        forces = np.array(rpms**2) * self.KF
        torques = np.array(rpms**2) * self.KM
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        for i in range(4):
            p.applyExternalForce(
                self.bodyId,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.physics_client_id,
            )

        p.applyExternalTorque(
            self.bodyId, 4, [0, 0, z_torque], p.LINK_FRAME, physicsClientId=self.physics_client_id
        )

        # Trainee aerodynamique (modele gym-pybullet-drones), calculee sur la
        # vitesse AIR : c'est par elle que le vent agit sur le drone.
        # La force est exprimee en repere CORPS et appliquee avec LINK_FRAME
        # (la convertir en repere monde la ferait tourner deux fois).
        rot = np.array(p.getMatrixFromQuaternion(gt["orn_q"])).reshape(3, 3)
        v_air_body = rot.T @ (np.asarray(gt["vel"], dtype=float) - self.current_wind)
        prop_wash_factor = np.sum(2 * np.pi * rpms / 60)
        drag_force_body = -self.DRAG_COEFF * prop_wash_factor * v_air_body
        p.applyExternalForce(
            self.bodyId,
            -1,
            forceObj=drag_force_body.tolist(),
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.physics_client_id,
        )

    def _log_full_state(self, gt):
        """
        Log the complete state of the UAV to a CSV file.
        Logs ground truth position and velocity, measured position with GNSS error,
        current wind conditions, repulsive forces, nearest neighbor distance, target
        tracking information, and collision status.
        :param self: The UAV instance
        :param gt: Dictionary containing ground truth data with keys 'pos' (position)
                   and 'vel' (velocity)
        :return: None
        """
        # Detect collision (simple proximity check for log flag)
        collision_flag = 0
        if len(p.getContactPoints(self.bodyId)) > 0:
            collision_flag = 1

        meas_pos = np.array(self.last_gnss_meas_pos, dtype=np.float32)
        ekf_pos = np.array(self.ekf.x[:3], dtype=np.float32)

        # Derived metrics
        gnss_error = np.linalg.norm(np.array(meas_pos) - np.array(gt["pos"]))
        ekf_error = np.linalg.norm(np.array(ekf_pos) - np.array(gt["pos"]))
        wind_mag = np.linalg.norm(self.current_wind)
        tracking_error = np.linalg.norm(np.array(gt["pos"]) - np.array(self.current_target_pos))

        with open(self.log_file, "a", newline="") as f:
            row = [
                round(self._sim_time, 3),
                # Ground Truth
                *gt["pos"],
                *gt["vel"],
                # Sensors
                *meas_pos,
                gnss_error,
                *ekf_pos,
                ekf_error,
                # Environment
                *self.current_wind,
                wind_mag,
                # Interaction
                round(self.last_repulsive_force_mag, 3),
                round(self.dist_to_nearest_neighbor, 3),
                # Intent
                *self.current_target_pos,
                tracking_error,
                collision_flag,
            ]
            # Clean float formatting
            row = [x if isinstance(x, (int, str)) else round(float(x), 4) for x in row]
            csv.writer(f).writerow(row)
