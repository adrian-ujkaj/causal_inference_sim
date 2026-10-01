import os
import csv
import threading
import pybullet as p
import numpy as np
import zmq 
import json
import random

# Utility Imports & Control
from utilities.utilities import point_in_cube
from entities.agent import Agent
from entities.sensor import GNSSensor, IMUSensor, LidarSensor
from Control.EKF import EKF
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
    **Sensor Fusion:**
    - Extended Kalman Filter (EKF) combining GNSS, IMU, and Lidar
    - Realistic GNSS latency and noise simulation
    - IMU measurement integration with body-frame acceleration
    - Adaptive sensor scheduling based on configurable frequencies
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
        ekf (EKF): Extended Kalman Filter for state estimation
        gnss (GNSSensor): GNSS sensor for position/velocity measurement
        imu (IMUSensor): IMU sensor for acceleration measurement
        lidar (LidarSensor): Lidar sensor for obstacle detection
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
    def __init__(self, config: dict, physics_client_id: int, dt: float, known_obstacles_config: dict, planner, world_type: str):
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
                - lidar (dict): Lidar sensor settings (frequency, range, noise).
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
            - Sensor fusion: EKF with GNSS, IMU, and Lidar integration.
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
        self.start_pos = config.get("start_pos", [0, 0, 1.0])
        self.start_orn = p.getQuaternionFromEuler(config.get("start_orn_euler", [0,0,0]))
        super().__init__(urdf_path, self.start_pos, self.start_orn, physics_client_id, self.dt)
        self._sim_time = 0.0
        
        # --- CONTROL FREQUENCY OPTIMIZATION (100Hz) ---
        self.CTRL_FREQ = self.config.get("ctrl_freq", 100)
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.last_ctrl_time = -self.CTRL_DT  # Force update at t=0

        # --- PHYSICS ---
        self.KF = self.config.get("physics", {}).get("thrust_coeff", 6.11e-8)
        self.KM = self.config.get("physics", {}).get("torque_coeff", 1.5e-9)
        self.G = 9.81
        self.MAX_RPM = config.get("physics", {}).get("max_rpm", 22000.0)
        self.max_speed = config.get("physics", {}).get("max_speed", 5)
        self.DRAG_COEFF = np.array([9.17e-7, 9.17e-7, 10.31e-7])
        
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.last_rpms = np.zeros(4)
        
        # --- NAVIGATION ---
        wp_list = config.get("waypoints", [])
        if not wp_list:
            wp_list = [[0, 0, 1]]
        first_wp = np.array(self.start_pos) + np.array([0, 0, 1])
        self.waypoints = [first_wp] + [np.array(w) for w in wp_list]
        self.wp_idx = 0
        
        # --- OBSTACLES & PLANNING ---
        self.obs_dic = known_obstacles_config  # Combined list for avoidance
        print(len(self.obs_dic), "known obstacle points loaded.")
        self.environment = world_type
        
        self.planner = planner
        
        self.target_yaw_cache = 0.0
        
        self.max_repulsive_force = self.config.get("physics", {}).get("max_repulsive_force", 2.0)
        self.safety_radius = self.config.get("physics", {}).get("safety_radius", 2.0)

        # Planning States
        self.active_path = []
        self.is_planning = False      
        self.planning_thread = None   
        self.replan_timer = 0         
        self.calculation_fail_count = 0

        self.zmq_ctx = zmq.Context()
        self.sub_socket = None
        self.pub_socket = None
        self.radar_sub_socket = None 

        # --- SWARM CONTROL ---
        self.swarm_active = False
        self.swarm_target_pos = None
        self.swarm_target_vel = None
        self.swarm_target_yaw = None 
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
        self.ekf = EKF(self.CTRL_DT)
        self.ekf.x[:3] = self.start_pos
        self.gnss = GNSSensor(sens.get("gnss", {}))
        self.imu = IMUSensor(sens.get("imu", {}))
        self.lidar = LidarSensor(sens.get("lidar", {}))
        
        lidar_freq = sens.get("lidar", {}).get("frequency", 10.0)
        self.lidar_period = 1.0 / lidar_freq
        self.last_lidar_time = -self.lidar_period
        
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
        # Backward compatible parsing:
        # - preferred: config['wind'] = {'wind_mean': [x,y,z], 'turbulence': val}
        # - legacy:    config['wind_mean'], config['turbulence']
        self.current_wind = np.zeros(3)
        wind_cfg = self.config.get("wind", {}) if isinstance(self.config.get("wind", {}), dict) else {}
        self.mean_wind = wind_cfg.get("wind_mean", self.config.get("wind_mean", [0, 0, 0]))
        self.turbulence = wind_cfg.get("turbulence", self.config.get("turbulence", 15))
        self.wind_module = DrydenGustModel(self.dt, self.turbulence, self.mean_wind)
        
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
        if os.path.exists(self.log_file): os.remove(self.log_file)
        
        
        # Complete header for causal analysis
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", 
                "gt_x", "gt_y", "gt_z",         # Ground Truth
                "gt_vx", "gt_vy", "gt_vz",      
                "meas_x", "meas_y", "meas_z",  # Sensors
                "gnss_error_mag",
                "ekf_x", "ekf_y", "ekf_z",
                "ekf_pos_error_mag",
                "wind_x", "wind_y", "wind_z",  # Environment
                "wind_mag",
                "rep_force_mag",                # Interaction
                "nearest_neighbor_dist",
                "target_x", "target_y", "target_z",  # Intent
                "tracking_error_mag",
                "collision_flag"                # Flags
            ])
        
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
        if not self.is_planning:
            self.is_planning = True
            print(f"[{self.name}] ⏳ Starting A* Thread...")
            self.planning_thread = threading.Thread(
                target=self._run_async_plan, 
                args=(start_pos, target_pos)
            )
            self.planning_thread.daemon = True 
            self.planning_thread.start()
        if not self.is_planning:
            self.is_planning = True
            print(f"[{self.name}] ⏳ Starting A* Thread...")
            self.planning_thread = threading.Thread(
                target=self._run_async_plan, 
                args=(start_pos, target_pos)
            )
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
            if path and len(path) > 0:
                self.active_path = path 
                self.calculation_fail_count = 0
            else:
                self.calculation_fail_count += 1
        except Exception as e:
            print(f"[{self.name}] 💥 Error in A* thread: {e}")
        finally:
            self.is_planning = False 

    def _analyze_target_accessibility(self, start_pos, target_pos):
        """
        Analyze target accessibility by checking obstacles and line-of-sight.
        
        Validates whether a target position can be reached from the start position by:
        1. Checking if target is inside any known obstacle
        2. Checking if direct line-of-sight is blocked by collisions
        
        Args:
            start_pos (np.ndarray): Starting position [x, y, z]
            target_pos (np.ndarray): Target position [x, y, z]
        
        Returns:
            str: One of "INVALID" (target in obstacle), "BLOCKED" (line-of-sight obstructed), or "CLEAR"
        """
        for obstacle in self.obs_dic:
            if point_in_cube(target_pos, obstacle):
                return "INVALID"

        if self.planner.building_tree is not None:
            indices = self.planner.building_tree.query_ball_point(target_pos[:2], r=5.0)
            for idx in indices:
                obs = self.obs_dic[idx]
                center = np.array(obs["center"])
                h = obs["height"]
                if target_pos[2] < h + 1.0 and point_in_cube(target_pos, obs):
                    return "INVALID"

        return "CLEAR"
        for obstacle in self.obs_dic:
            if point_in_cube(target_pos, obstacle):
                return "INVALID"
        if p.raycast(start_pos, target_pos)[1][0] >= 0:
            return "BLOCKED"
        return "CLEAR"
    
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
                    mag = (1.0 - (dist_uav / self.safety_radius))
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
                mag = (1.0 - (dist / self.safety_radius))
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
        if not hasattr(self, "pub_socket"):
            return 
        pos = [round(p, 3) for p in pos]
        vel = [round(v, 3) for v in vel]
        msg = {
            "name": self.name,
            "pos": pos,
            "vel": vel,
            "yaw": round(self.target_yaw_cache, 3),
            "sim_time": round(self._sim_time, 3)
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
                            self.future_state = state.get(self.name, None)
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
        
        # 1. Update Sim Time
        self._sim_time += self.dt

        # Wind simulation
        gt = self.get_ground_truth_state()
        h = gt["pos"][2]
        v_airspeed = np.linalg.norm(gt["vel"] - self.current_wind)

        self.current_wind = self.wind_module.step(h, v_airspeed)
        
        # 2. Logic Schedule (100 Hz)
        if (self._sim_time - self.last_ctrl_time) >= self.CTRL_DT:
            self._update_control_loop(gt)
            self.last_ctrl_time = self._sim_time
        
        # 3. Physics Application (Always 240Hz)
        # Use the last calculated RPMs to maintain stability
        self._apply_lib_physics(self.last_rpms, gt)
        
        # Logging (Optional: Log at physics freq or control freq)
        if int(self._sim_time / self.dt) % 10 == 0:
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
        
        orn_q = np.array(gt["orn_q"])
        ang_vel = np.array(gt["ang_vel"])
        rpy = np.array(p.getEulerFromQuaternion(orn_q))

        # ==================== ADVANCED SENSOR FUSION ====================
        
        # 1. GNSS measurement (jittered schedule) + EKF update
        # IMPORTANT: meas_* in logs are the raw GNSS outputs. EKF is logged separately.
        if self._sim_time >= self.next_gnss_trigger:
            meas_pos, meas_vel = self.gnss.measure(gt["pos"], gt["vel"], t=self._sim_time)
            self.last_gnss_meas_pos = np.array(meas_pos, dtype=np.float32)
            self.last_gnss_meas_vel = np.array(meas_vel, dtype=np.float32)
            self.ekf.update(meas_pos, meas_vel)

            # Next trigger: nominal period + random positive jitter (no drift)
            jitter = max(0.0, random.gauss(self.gnss_delay_mean, self.gnss_delay_std))
            self.last_gnss_nominal_time = max(self.last_gnss_nominal_time + self.gnss_dt, self._sim_time)
            self.next_gnss_trigger = self.last_gnss_nominal_time + jitter

        # Read the IMU (Acceleration + Orientation)
        # Note: In simple PyBullet, you can cheat and take gt['orn_q']
        # or use self.imu.measure(...) if your IMU sensor is complete.
        # Here I assume self.imu returns the raw accel.
        imu_acc, _ = self.imu.measure(gt["vel"], gt["orn_q"]) 
        # Note: Make sure your IMUSensor returns a np.array for imu_acc
        
        # 2. EKF Prediction (Based on IMU)
        self.ekf.predict(imu_acc_body=imu_acc, orientation_quat=gt["orn_q"])
        
        # 3. EKF Correction (GPS)
        # GPS corrects for IMU drift
        pos = self.ekf.x[:3]
        vel = self.ekf.x[3:6]

        # --- SENSORS (Lidar) ---
        if (self._sim_time - self.last_lidar_time) >= self.lidar_period:
            self.last_lidar_time = self._sim_time
            # new_pts = self.lidar.measure(pos, rpy[0], rpy[2], rpy[1])
            # if len(new_pts) > 0: ...

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
        
        # 1. Swarm Follower
        if self.swarm_active and not self.leader:          
            target_pos = self.future_state["pos"]
            if self.swarm_target_vel is not None:
                target_vel = self.future_state["vel"]

        # 2. Planning (Wait)
        elif self.is_planning:
            target_pos = pos 
            target_vel = -1 * vel  # Brake
            
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
                    if dist_wp < 0.5 and not self.is_planning:
                        print(f"[{self.name}] Waypoint {self.wp_idx} reached.")
                        self.wp_idx += 1
                        self.active_path = []  # Force a new calculation
                        self.replan_timer = 0

                while self.wp_idx < len(self.waypoints):
                    global_target = self.waypoints[self.wp_idx]
                
                    if len(self.active_path) == 0:
                        # Force A* planning to trigger for each new WP
                        if self.replan_timer <= 0:
                            self._trigger_planning(pos, global_target)
                            self.replan_timer = 100
                        break 
                    break

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
                target_pos = self.waypoints[self.wp_idx]
                if np.linalg.norm(target_pos - pos) < 0.3:
                    print(f"[{self.name}] WP {self.wp_idx} Reached.")
                    self.wp_idx += 1

        if self.replan_timer > 0:
            self.replan_timer -= 1

        # --- CONTROL COMMANDS ---
        self.current_target_pos = target_pos
        
        # Repulsive Force
        if self.environment == "generated":
            f_rep = self._compute_repulsive_force(pos)    
        if self.environment == "custom":
            f_rep = self.planner.compute_repulsive_force(pos, self.safety_radius, self.max_repulsive_force, self.swarm_active, self.leader, self.other_agent_pos)
        
        acc_rep = f_rep / self.mass

        final_target_vel = target_vel + (acc_rep * 3 * self.CTRL_DT)  # Use CTRL_DT here

        # Clamp Speed
        speed_xy = np.linalg.norm(final_target_vel[:2])
        if speed_xy > self.max_speed:
            ratio = self.max_speed / speed_xy
            final_target_vel[:2] *= ratio
        
        # PID Target Helper
        final_target_pos = target_pos + (final_target_vel * self.CTRL_DT)
        vector_to_target = final_target_pos - pos
        dist_to_target = np.linalg.norm(vector_to_target)
        if dist_to_target > 2.0:
            virtual_target_pos = pos + (vector_to_target / dist_to_target) * 1.0
        else:
            virtual_target_pos = final_target_pos
        
        # Yaw
        direction_vec = final_target_pos - pos
        if self.swarm_active and self.swarm_target_yaw is not None:
            self.target_yaw_cache = self.future_state["yaw"]
        elif np.linalg.norm(direction_vec[:2]) > 0.5:
            self.target_yaw_cache = np.arctan2(direction_vec[1], direction_vec[0])

        state_vec = np.hstack([pos, orn_q, rpy, vel, ang_vel, self.last_rpms])
        
        # Compute RPMs (PID)
        rpms, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.CTRL_DT,  # Important: 0.01s
            state=state_vec, 
            target_pos=virtual_target_pos, 
            target_vel=final_target_vel, 
            target_rpy=np.array([0, 0, self.target_yaw_cache]) 
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
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        for i in range(4):
            p.applyExternalForce(self.bodyId, i, forceObj=[0, 0, forces[i]], posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        
        try:
            p.applyExternalTorque(self.bodyId, 4, [0, 0, z_torque], p.LINK_FRAME)
            rot = np.array(p.getMatrixFromQuaternion(gt["orn_q"])).reshape(3, 3)
            v_air_world = gt["vel"] - self.current_wind
            v_air_body = rot.T @ v_air_world
            prop_wash_factor = np.sum(2 * np.pi * rpms / 60)
            drag_force_body = -1 * self.DRAG_COEFF * prop_wash_factor * v_air_body
            f_drag_for_bullet = rot @ drag_force_body 
            p.applyExternalForce(self.bodyId, -1, forceObj=f_drag_for_bullet, posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.physics_client_id)
        except:
            pass 

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
                *gt["pos"], *gt["vel"],
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
                collision_flag
            ]
            # Clean float formatting
            row = [x if isinstance(x, (int, str)) else round(float(x), 4) for x in row]
            csv.writer(f).writerow(row)
