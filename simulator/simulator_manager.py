# simulator/simulator_manager.py
import time
import random
import numpy as np
import pybullet as p
import pybullet_data

from environment.world import World
from entities.uav import UAV
from swarm.swarm import Swarm
from entities.static_sensor import RadarStation
from Control.Path_planning import HeightmapAStar


class SimulationManager:
    """Connexion PyBullet, monde, drones, radars et essaims, puis boucle de simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.dt = float(self.config["simulation"]["dt"])

        # Reproducibility (set simulation.seed in config.yaml)
        sim_cfg = self.config.get("simulation", {})
        simulation_seed = sim_cfg.get("seed", None)
        if simulation_seed is not None:
            try:
                simulation_seed = int(simulation_seed)
                random.seed(simulation_seed)
                np.random.seed(simulation_seed)
            except Exception:
                pass
        # 1. Connexion PyBullet
        mode_str = str(self.config["simulation"]["connect_mode"]).strip().lower()
        self.realtime = bool(self.config.get("simulation", {}).get("realtime", mode_str == "gui"))
        mode = p.GUI if mode_str == "gui" else p.DIRECT
        self.physics_client_id = p.connect(mode)
        if self.physics_client_id < 0:
            raise ConnectionError("Impossible de se connecter à PyBullet.")

        print(f"Connecté à PyBullet, client_id={self.physics_client_id}")

        # Pas d'integration physique = simulation.dt (PyBullet utilise 1/240 s par defaut)
        p.setTimeStep(self.dt, physicsClientId=self.physics_client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(
            *self.config["physics"]["gravity"],
            physicsClientId=self.physics_client_id,
        )

        # 2. Monde (sol + obstacles)
        self.world = World(self.physics_client_id)

        if mode == p.GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=6.0,
                cameraYaw=45.0,
                cameraPitch=-35.0,
                cameraTargetPosition=[0.0, 1.0, 1.0],
                physicsClientId=self.physics_client_id,
            )

        # Liste de tous les agents (UAV + radars)
        self.agents: list[UAV | RadarStation] = []

        # Liste des essaims (on n'en crée qu'un, mais on garde une liste)
        self.swarms: list[Swarm] = []

        # Liste des radars
        self.radars: list[RadarStation] = []

        # 3. Charger scénario (obstacles + drones + objectifs éventuels)
        self.load_scenario()

        # 4. Créer un essaim si demandé dans la config
        self._create_swarm_from_config()

    # ------------------------------------------------------------------
    def load_scenario(self):
        print("Chargement du scénario...")

        self.obstacles_config = self.config.get("world", {})
        print(self.obstacles_config)
        # Obstacles
        res = self.obstacles_config.get("res", 0.25)
        world_type = self.obstacles_config.get("type", "generated")
        if world_type not in ("generated", "custom"):
            raise ValueError(f"world.type inconnu : {world_type!r} (attendu 'generated' ou 'custom')")
        obstacles = []
        self.planner = None
        if world_type == "generated":
            """Charge le sol + règle la physique."""
            obstacles = self.world.generate_city_urdf(self.obstacles_config.get("city", {}))

            p.loadURDF(
                self.world.city_urdf_path,  # fichier propre a ce processus
                basePosition=[0, 0, 0],
                useFixedBase=1,
                physicsClientId=self.physics_client_id,
            )
            self.planner = HeightmapAStar(
                self.obstacles_config.get("Astar", {}),
                resolution=res,
            )
            self.planner.build_from_buildings(obstacles)

        if world_type == "custom":
            world_file = self.obstacles_config.get("filename")
            p.loadURDF(
                world_file,
                basePosition=[0, 0, 0],
                useFixedBase=1,
                physicsClientId=self.physics_client_id,
            )
            self.planner = HeightmapAStar(
                self.obstacles_config.get("Astar", {}),
                resolution=res,
            )
            self.planner.custom_heightmap()
        # Drones
        # Optional per-run log directory (propagated to UAV configs)
        sim_log_dir = None
        if isinstance(self.config, dict):
            sim_log_dir = self.config.get("simulation", {}).get("log_dir", None)
        for agent_cfg in self.config.get("agents", []):
            # Propagate global log directory to each UAV config (if provided)
            if sim_log_dir and isinstance(agent_cfg, dict) and agent_cfg.get("type") == "uav":
                agent_cfg["log_dir"] = sim_log_dir

            if agent_cfg.get("type") == "radar":
                radar = RadarStation(config=agent_cfg, physics_client_id=self.physics_client_id, dt=self.dt)
                self.agents.append(radar)  # On l'ajoute à la boucle principale pour le think_and_act
                self.radars.append(radar)

            elif agent_cfg.get("type") == "uav":
                uav = UAV(
                    config=agent_cfg,
                    physics_client_id=self.physics_client_id,
                    dt=self.dt,
                    known_obstacles_config=obstacles,
                    planner=self.planner,
                    world_type=world_type,
                )
                self.agents.append(uav)

        for radar in self.radars:
            radar.targets = [a for a in self.agents if isinstance(a, UAV)]
        # Objectifs optionnels du YAML (section `objectives`)
        for objective in self.config.get("objectives", []):
            agent_id = objective.get("agent")
            if objective.get("type") == "reach_position":
                if agent_id is None:
                    raise ValueError("Objective of type 'reach_position' is missing 'agent'.")
                setpoint = np.array(
                    objective.get("target_pos", [0.0, 0.0, 0.0]),
                    dtype=float,
                )

                # assign target to the matching agent (by index or by name)
                if isinstance(agent_id, int):
                    if 0 <= agent_id < len(self.agents):
                        agent = self.agents[agent_id]
                        if hasattr(agent, "set_target_pos"):
                            agent.set_target_pos(setpoint)
                        elif hasattr(agent, "set_target_position"):
                            agent.set_target_position(setpoint)
                        elif hasattr(agent, "set_target"):
                            agent.set_target(setpoint)
                        else:
                            agent.target_pos = setpoint
                    else:
                        raise ValueError(f"Agent index {agent_id} out of range for objective.")
                else:
                    # agent_id est un nom
                    found = False
                    for agent in self.agents:
                        if getattr(agent, "name", None) == agent_id:
                            if hasattr(agent, "set_target_pos"):
                                agent.set_target_pos(setpoint)
                            elif hasattr(agent, "set_target_position"):
                                agent.set_target_position(setpoint)
                            elif hasattr(agent, "set_target"):
                                agent.set_target(setpoint)
                            else:
                                agent.target_pos = setpoint
                            found = True
                            break
                    if not found:
                        raise ValueError(f"No agent with name '{agent_id}' found for objective.")

        print(f"Scénario chargé : {len(self.agents)} drones, {len(self.world.obstacle_ids)} obstacles.")

    # ------------------------------------------------------------------
    def _create_swarm_from_config(self):
        """
        Crée les essaims en associant les drones par 'swarm_id'
        et en appliquant la config spécifique définie dans le YAML.
        """
        # 1. Chargement des configs d'essaims (YAML)
        # Le YAML est une liste : [{id: "A", ...}, {id: "B", ...}]
        raw_swarm_configs = self.config.get("swarm", [])

        # On convertit en dictionnaire pour accès rapide : { "A": {config}, "B": {config} }
        swarm_configs_map = {}
        if isinstance(raw_swarm_configs, list):
            for cfg in raw_swarm_configs:
                sid = str(cfg.get("id"))
                swarm_configs_map[sid] = cfg
        elif isinstance(raw_swarm_configs, dict):
            # Cas où il n'y a qu'un seul essaim défini sans tiret
            sid = str(raw_swarm_configs.get("id", "default"))
            swarm_configs_map[sid] = raw_swarm_configs

        # 2. Regroupement des drones
        swarms_groups = {}
        uavs = [a for a in self.agents if isinstance(a, UAV)]

        for uav in uavs:
            s_id = uav.config.get("swarm_id", None)
            if s_id is not None:
                s_id = str(s_id)
                if s_id not in swarms_groups:
                    swarms_groups[s_id] = []
                swarms_groups[s_id].append(uav)

        # 3. Création des objets Swarm
        if not swarms_groups:
            print("[Swarm] Aucun 'swarm_id' trouvé sur les drones.")
            return

        for s_id, members in swarms_groups.items():
            if len(members) < 2:
                print(f"[Swarm] Groupe '{s_id}' : Trop petit (<2). Ignoré.")
                continue

            # Config de cet essaim dans 'swarm:' (valeurs par défaut sinon)
            specific_cfg = swarm_configs_map.get(s_id, {})

            print(f"[Swarm] Création groupe '{s_id}' avec config : {specific_cfg}")

            # Extraction des paramètres spécifiques
            leader_name = specific_cfg.get("leader", None)  # Le nom du drone leader
            min_sep = float(specific_cfg.get("min_sep", 0.6))
            avoid_gain = float(specific_cfg.get("avoid_gain", 0.5))

            # Paramètres Réseau
            port_in = int(specific_cfg.get("port_in", 5556))
            port_out = int(specific_cfg.get("port_out", 5557))
            ip = specific_cfg.get("ip", "localhost")

            # Création de l'instance
            new_swarm = Swarm(
                agents=members,
                leader_name=leader_name,
                formation_body_offsets=None,
                min_sep=min_sep,
                avoid_gain=avoid_gain,
                port_in=port_in,
                port_out=port_out,
                ip=ip,
            )
            self.swarms.append(new_swarm)

    # ------------------------------------------------------------------
    def run(self):
        """
        Boucle principale de simulation.
        """
        sim_time = 0.0
        max_time = float(self.config["simulation"]["max_sim_time"])

        while sim_time < max_time and p.isConnected(self.physics_client_id):
            # 1. Mise à jour des essaims (leader/followers) si activés
            for swarm in self.swarms:
                swarm.update()

            # 2. Contrôle de chaque drone
            for agent in self.agents:
                if agent.type == "uav":
                    agent.think_and_act()

                elif agent.type == "radar":
                    if sim_time > agent.radar_period + agent.radar_last_time:
                        agent.radar_last_time = sim_time
                        agent.think_and_act(sim_time)
            # 3. Avancer la physique
            p.stepSimulation(physicsClientId=self.physics_client_id)

            # 4. Real time
            if self.realtime:
                time.sleep(self.dt)
            sim_time += self.dt

    # ------------------------------------------------------------------
    def stop(self):
        # Ecrire les journaux en memoire, meme si la simulation a ete interrompue
        for agent in self.agents:
            if hasattr(agent, "close_logs"):
                try:
                    agent.close_logs()
                except Exception as exc:
                    print(f"[stop] journaux de {getattr(agent, 'name', '?')} : {exc}")
        if p.isConnected(self.physics_client_id):
            print("Déconnexion de PyBullet.")
            p.disconnect(self.physics_client_id)
        # Nettoyage de l'essaim (proxy et sockets ZMQ), meme si la fenetre est fermee
        for swarm in self.swarms:
            swarm.cleanup()
        # Sockets ZMQ des drones et des radars
        for agent in self.agents:
            for attr in ("sub_socket", "pub_socket", "radar_sub_socket"):
                sock = getattr(agent, attr, None)
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception:
                        pass
            ctx = getattr(agent, "zmq_ctx", None)
            if ctx is not None:
                try:
                    ctx.term()
                except Exception:
                    pass
        # Fichier URDF temporaire de la ville
        path = getattr(self.world, "city_urdf_path", None)
        if path:
            try:
                import os

                os.remove(path)
            except OSError:
                pass
