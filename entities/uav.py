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
    """Quadrirotor simule : physique PyBullet, controleur PID (DSLPIDControl),
    planification A*, filtre de navigation GNSS/IMU, communication avec l'essaim
    (ZMQ) et journaux CSV pour l'analyse."""

    def __init__(
        self,
        config: dict,
        physics_client_id: int,
        dt: float,
        known_obstacles_config: dict,
        planner,
        world_type: str,
    ):
        """config : section du drone dans config.yaml ; dt : pas physique (s)."""
        self.config = config
        self.dt = float(dt)
        self.physics_client_id = physics_client_id
        self.name = config.get("name", "UAV")
        self.bodyId = config.get("body_id", 0000)
        self.type = "uav"
        self.mass = config.get("mass", 1.5)  # kg

        urdf_path = config.get("urdf_path", "assets/quadrotor.urdf")
        self.start_pos = list(config.get("start_pos", [0, 0, 1.0]))

        # Altitude d'apparition minimale : un drone pose a z = 0 reste colle au sol
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

        # Controle tous les ctrl_every pas physiques : CTRL_DT est un multiple de dt
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

        # Inclinaison maximale commandee (deg)
        self.max_tilt_deg = float(self.config.get("max_tilt_deg", 30.0))
        # Vol pres du sol
        self.ground_min_alt = float(self.config.get("ground_min_alt", 0.15))
        self.ground_clear_alt = float(self.config.get("ground_clear_alt", 0.6))
        self.ground_speed_floor = float(self.config.get("ground_speed_floor", 0.1))
        self.ground_tilt_deg = float(self.config.get("ground_tilt_deg", 10.0))
        self.max_descent_speed = float(self.config.get("max_descent_speed", 1.5))
        self.max_climb_speed = float(self.config.get("max_climb_speed", 1.2))
        # Vitesse maximale de la consigne de lacet (rad/s)
        self.max_yaw_rate = float(self.config.get("max_yaw_rate", 1.0))
        self._yaw_cmd = float(config.get("start_orn_euler", [0, 0, 0])[2])
        self.ground_descent_speed = float(self.config.get("ground_descent_speed", 0.4))
        self._ground_factor = 1.0
        # Part maximale de la demande horizontale laissee au terme integral
        self.integral_share = float(self.config.get("integral_share", 0.3))
        # Poussee verticale admissible, en fraction du poids
        self.min_thrust_ratio = float(self.config.get("min_thrust_ratio", 0.6))
        self.max_thrust_ratio = float(self.config.get("max_thrust_ratio", 1.6))
        self._loss_of_control = False

        # --- NAVIGATION ---
        wp_list = config.get("waypoints", [])
        if not wp_list:
            wp_list = [[0, 0, 1]]

        # Waypoints remontes a une altitude minimale (un waypoint au sol bloque A*)
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

        # Sequencement des waypoints
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

        # Parametres d'evitement, dans physics ou au niveau du drone
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
        # Relance de A* apres un echec (~1.25 s)
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

        # Filtre de navigation : "eskf" (15 etats, defaut) ou "kf6" (position-vitesse)
        fcfg = dict(self.config.get("filter", {}) or {})
        self.filter_type = str(fcfg.get("type", "eskf")).lower()
        start_yaw = float(config.get("start_orn_euler", [0, 0, 0])[2])
        fcfg.setdefault("initial_yaw", start_yaw)
        FilterClass = ESKF if self.filter_type == "eskf" else INSGNSSFilter
        self.ekf = FilterClass(
            self.CTRL_DT, gnss_config=sens.get("gnss", {}), imu_config=sens.get("imu", {}), config=fcfg
        )
        self.ekf.init_state(self.start_pos, np.zeros(3))

        # Attitude utilisee par la boucle interne : "truth" (defaut) ou "filter" (ESKF)
        self.attitude_source = str(fcfg.get("attitude_source", "truth")).lower()
        if self.attitude_source == "filter" and self.filter_type != "eskf":
            print(
                f"[{self.name}] attitude_source='filter' exige filter.type='eskf' : attitude vraie utilisee"
            )
            self.attitude_source = "truth"
        self._gnss_updated = False
        self._gnss_err = float("nan")
        self.gnss = GNSSensor(sens.get("gnss", {}))
        # IMU lue a la cadence de controle
        self.imu = IMUSensor(sens.get("imu", {}), dt=self.CTRL_DT)
        self.last_imu_gyro = np.zeros(3)

        self.gnss_freq = sens.get("gnss", {}).get("frequency", 10.0)  # 10 Hz (Realistic Standard)
        self.gnss_dt = 1.0 / self.gnss_freq
        self.last_gnss_update_time = -self.gnss_dt
        self.gnss_delay_mean = sens.get("gnss", {}).get("delay_mean", 0.1)
        self.gnss_delay_std = sens.get("gnss", {}).get("delay_std", 0.01)
        self.next_gnss_trigger = 0.0

        # Echeancier GNSS et derniere mesure recue
        self.last_gnss_nominal_time = 0.0
        self.last_gnss_meas_pos = np.array(self.start_pos, dtype=np.float32)
        self.last_gnss_meas_vel = np.zeros(3, dtype=np.float32)

        # --- WIND ---
        # wind : wind_mean, turbulence, et en option une rafale (burst_start, burst_end, burst_turbulence)
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

        # Journal de validation du filtre (erreur = estime - vrai)
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
        # Trajectoire vraie, pour rejouer les filtres hors ligne
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
        """Erreurs, ecarts-types, biais, NEES et NIS du filtre dans <nom>_filter.csv."""
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
                # erreur GNSS a l'instant de mesure
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
        """Borne la consigne envoyee au PID : poussee verticale, vitesse verticale et
        inclinaison commandee (sous max_tilt_deg). L'erreur horizontale est reduite
        sans changer de direction, et le terme integral est limite."""
        P = np.asarray(self.ctrl.P_COEFF_FOR, dtype=float)
        D = np.asarray(self.ctrl.D_COEFF_FOR, dtype=float)
        I = np.asarray(self.ctrl.I_COEFF_FOR, dtype=float)
        weight = float(self.ctrl.GRAVITY)  # m*g en N, convention DSL
        integ = getattr(self.ctrl, "integral_pos_e", None)

        pos_e = np.asarray(virtual_target_pos, dtype=float) - np.asarray(pos, dtype=float)
        vel_e = np.asarray(final_target_vel, dtype=float) - np.asarray(vel, dtype=float)
        pos_e = pos_e.copy()
        vel_e = vel_e.copy()

        # 1. Poussee verticale bornee : l'axe de poussee doit rester vers le haut
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

        # 1b. Vitesse verticale bornee, descente plus lente pres du sol
        gf = getattr(self, "_ground_factor", 1.0)
        v_down_max = self.ground_descent_speed + (self.max_descent_speed - self.ground_descent_speed) * gf
        tz_floor = tz_min
        # pres du sol, pas de chute libre
        tz_floor = max(
            tz_floor, weight * (self.min_thrust_ratio + (0.9 - self.min_thrust_ratio) * (1.0 - gf))
        )
        if float(vel[2]) < -v_down_max:
            tz_floor = max(tz_floor, 1.15 * weight)  # descente trop rapide : freiner
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

        # 2. Inclinaison admissible, plus faible pres du sol
        gf = getattr(self, "_ground_factor", 1.0)
        tilt_deg = self.ground_tilt_deg + (self.max_tilt_deg - self.ground_tilt_deg) * gf
        limit = float(np.tan(np.radians(tilt_deg)) * tz)

        # 3. Limite du terme integral horizontal
        c = np.zeros(2)
        if integ is not None:
            c = I[:2] * integ[:2]
            c_max = self.integral_share * limit
            nc = float(np.linalg.norm(c))
            if nc > c_max and nc > 1e-12:
                integ[:2] *= c_max / nc  # modifie l'etat du controleur
                c = I[:2] * integ[:2]

        u = P[:2] * pos_e[:2] + D[:2] * vel_e[:2]

        # 4. Plus grand k dans [0, 1] tel que |k u + c| <= limit
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
        """Lance le calcul A* dans un thread (ou directement si planning_sync)."""
        if self.is_planning:
            return
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        # Trajet vertical : rien a calculer pour A* (grille 2D)
        if float(np.linalg.norm(target_pos[:2] - start_pos[:2])) < self.direct_leg_xy:
            self.active_path = [target_pos.copy()]
            self._plan_wp_idx = self.wp_idx
            self.calculation_fail_count = 0
            return

        self._planning_for_wp = self.wp_idx
        self.is_planning = True
        if self.planning_sync:
            # Mode deterministe pour les campagnes Monte-Carlo
            self._run_async_plan(start_pos, target_pos)
            return
        print(f"[{self.name}] Starting A* thread...")
        self.planning_thread = threading.Thread(target=self._run_async_plan, args=(start_pos, target_pos))
        self.planning_thread.daemon = True
        self.planning_thread.start()

    def _run_async_plan(self, start_pos, target_pos):
        """Calcule le chemin A* et met a jour active_path."""
        try:
            path = self.planner.plan(start_pos, target_pos)
            # Plan ignore si le waypoint a change pendant le calcul
            if path and len(path) > 0 and self._planning_for_wp == self.wp_idx:
                self.active_path = [np.asarray(w, dtype=float) for w in path]
                self._plan_wp_idx = self.wp_idx
                self.calculation_fail_count = 0
            elif not path:
                self.calculation_fail_count += 1
        except Exception as e:
            # Une exception compte comme un echec
            self.calculation_fail_count += 1
            print(f"[{self.name}] Error in A* thread: {e}")
        finally:
            self.is_planning = False

    def _compute_repulsive_force(self, current_pos):
        """Repulsion des autres drones et des batiments proches, bornee a max_repulsive_force."""
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

        # Ferme d'anciens sockets (plusieurs simulations dans le meme processus)
        for attr in ("sub_socket", "pub_socket"):
            sock = getattr(self, attr, None)
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass

        # Le proxy de l'essaim fait le bind, les drones se connectent seulement
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
        """Abonnement ZMQ aux radars de la configuration."""
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
                print(f"[{self.name}] Error: radar port not specified in config.")

    def broadcast_state(self, pos, vel):
        """Diffuse l'etat du drone a l'essaim (JSON via ZMQ)."""
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
        """Lit les messages radar, avec un retard de perception simule."""
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

        # Replace buffer with remaining messages
        self.radar_message_buffer = buffer_remaining

    def listen_swarm(self):
        """Lit les messages de l'essaim, avec un retard de perception simule."""
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

        # Replace buffer with remaining messages
        self.message_buffer = buffer_remaining

    # -----------------------------------------------------------------------
    # MAIN LOOP & LOGIC
    # -----------------------------------------------------------------------
    def think_and_act(self):
        """Un pas de simulation : vent, controle (tous les ctrl_every pas), physique, journal."""
        if not p.isConnected(self.physics_client_id):
            return

        # 1. Horloge : compteur entier de pas physiques
        self._tick += 1
        self._sim_time = self._tick * self.dt

        gt = self.get_ground_truth_state()
        h = gt["pos"][2]
        # Vent : rafales orientees selon la vitesse air (par rapport au vent moyen)
        v_air = np.asarray(gt["vel"], dtype=float) - self.wind_module.mean_wind
        self.current_wind = self.wind_module.step(h, float(np.linalg.norm(v_air)), v_air, t=self._sim_time)

        # 2. Boucle de controle, tous les ctrl_every pas
        if (self._tick - 1) % self.ctrl_every == 0:  # pas 1, 1+n, 1+2n...
            self._update_control_loop(gt)
            self.last_ctrl_time = self._sim_time

        # 3. Physique, a chaque pas
        self._apply_lib_physics(self.last_rpms, gt)

        # Journal principal tous les 10 pas
        if self._tick % 10 == 0:
            self._log_full_state(gt)

    def _update_control_loop(self, gt):
        """Boucle de controle : IMU et GNSS, filtre, communication, choix de la cible, commande PID."""

        true_orn_q = np.array(gt["orn_q"])  # attitude vraie
        ang_vel = np.array(gt["ang_vel"])

        # --- NAVIGATION ---

        # Predict puis update ; on publie l'etat corrige

        # 1. IMU, sur le pas reel depuis le dernier appel
        imu_dt = self._sim_time - self.last_ctrl_time
        self._ctrl_elapsed = imu_dt
        imu_acc, imu_gyro = self.imu.measure(gt["vel"], gt["orn_q"], ang_vel=gt["ang_vel"], dt=imu_dt)
        self.last_imu_gyro = imu_gyro

        # 2. Propagation (l'ESKF propage sa propre attitude, le KF6 recoit l'attitude vraie)
        if self.filter_type == "eskf":
            self.ekf.predict(imu_acc, imu_gyro, dt=imu_dt)
        else:
            self.ekf.predict(imu_acc, self.imu.q_mid, dt=imu_dt)

        # 3. Correction GNSS ; pendant une coupure le filtre continue en inertie pure
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

            # Prochaine mesure : echeancier periodique + gigue, sans accumulation
            jitter = max(0.0, random.gauss(self.gnss_delay_mean, self.gnss_delay_std))
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

        # 1. Suiveur : position et vitesse de consigne de la formation
        if self.swarm_active and not self.leader:
            fs = self.future_state or {}
            if fs.get("pos") is not None:
                target_pos = np.asarray(fs["pos"], dtype=float)
            if fs.get("vel") is not None:
                target_vel = np.asarray(fs["vel"], dtype=float)
                # Message recu avec retard : on extrapole la cible
                if fs.get("t") is not None:
                    age = float(np.clip(self._sim_time - float(fs["t"]), 0.0, 0.5))
                    target_pos = target_pos + target_vel * age

        # 2. Planning (Wait)
        elif self.is_planning:
            # En attente du plan : maintien sur place
            if self._hold_pos is None:
                self._hold_pos = np.array(pos, dtype=float)
            target_pos = self._hold_pos

        # 3. Autonomous Navigation
        else:
            # --- FAILSAFE CHECK ---
            if self.calculation_fail_count > 5:
                print(f"[{self.name}] Too many A* failures ({self.calculation_fail_count}). Skipping WP.")
                self.wp_idx += 1
                self.calculation_fail_count = 0
                self.replan_timer = 0
                return  # Skip this cycle to reset logic

            else:
                # Detect arrival at Waypoint
                if self.wp_idx < len(self.waypoints):
                    dist_wp = np.linalg.norm(self.waypoints[self.wp_idx] - pos)

                    # Arrivee : dans la tolerance, ou waypoint approche puis depasse
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

                # Un seul calcul A* par waypoint
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
                # Pas de chemin : ligne droite vers le waypoint
                target_pos = self.waypoints[self.wp_idx]
            else:
                # Mission terminee : maintien sur un point fixe
                if self._hold_pos is None:
                    self._hold_pos = np.array(pos, dtype=float)
                target_pos = self._hold_pos

        if self.replan_timer > 0:
            self.replan_timer -= 1

        # --- CONTROL COMMANDS ---
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

        # Force repulsive convertie en vitesse de consigne (gain en s)
        acc_rep = f_rep / self.mass
        final_target_vel = np.asarray(target_vel, dtype=float) + acc_rep * self.repulsion_gain_s

        # Pres du sol : vitesse horizontale et inclinaison reduites
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
            # Cible virtuelle a lookahead_m au plus
            virtual_target_pos = pos + (vector_to_target / dist_to_target) * self.lookahead_m
        else:
            virtual_target_pos = final_target_pos

        # Limites de poussee et d'inclinaison
        virtual_target_pos, final_target_vel = self._limit_tilt_demand(
            virtual_target_pos, final_target_vel, pos, vel
        )

        # Detection de retournement (attitude vraie)
        tilt = float(
            np.degrees(
                np.arccos(
                    np.clip(np.array(p.getMatrixFromQuaternion(true_orn_q)).reshape(3, 3)[2, 2], -1.0, 1.0)
                )
            )
        )
        if tilt > 90.0 and not self._loss_of_control:
            self._loss_of_control = True
            print(f"[{self.name}] Perte de controle a t={self._sim_time:.2f}s (inclinaison {tilt:.0f} deg)")

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

        # Consigne de lacet a vitesse bornee (un saut de lacet sature les moteurs)
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
            control_timestep=self._ctrl_elapsed,  # pas reel depuis le dernier appel
            state=state_vec,
            target_pos=virtual_target_pos,
            target_vel=final_target_vel,
            target_rpy=np.array([0, 0, self._yaw_cmd]),
        )

        self.last_rpms = rpms

    def _apply_lib_physics(self, rpms, gt):
        """Poussee et couple des rotors, trainee aerodynamique sur la vitesse air."""
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

        # Trainee (modele gym-pybullet-drones), en repere corps, sur la vitesse air
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
        """Ajoute une ligne au journal principal <nom>.csv."""
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
