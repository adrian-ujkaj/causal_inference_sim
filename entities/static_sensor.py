import json
import pybullet as p
import numpy as np
from entities.agent import Agent
import zmq


class RadarStation(Agent):
    def __init__(self, config: dict, physics_client_id: int, dt: float):
        self.config = config
        self.name = self.config.get("name", "Radar")
        self.physics_client_id = physics_client_id

        # Bruit de mesure
        self.pos_noise_std = float(self.config.get("position_noise_std", 0.1))
        self.range_noise_std = float(self.config.get("range_noise_std", 0.05))
        # Moyenne du bruit (0 par defaut)
        self.pos_noise_mean = float(self.config.get("position_noise_mean", 0.0))
        self.range_noise_mean = float(self.config.get("range_noise_mean", 0.0))

        self.bodyId = self.config.get("bodyId", 1000)

        self.radar_period = float(self.config.get("period", 0.1))  # s
        self.radar_last_time = 0.0

        self.type = self.config.get("type", "radar")

        self.pos = self.config.get("pos", [0, 0, 0])
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        urdf_path = self.config.get("urdf_path", "assets/cube.urdf")

        super().__init__(urdf_path, self.pos, start_orn, physics_client_id, dt)

        # Objet statique
        p.changeDynamics(
            self.bodyId, -1, mass=0, localInertiaDiagonal=[0, 0, 0], physicsClientId=self.physics_client_id
        )
        p.changeVisualShape(
            self.bodyId, -1, rgbaColor=[0.8, 0, 0, 0.6], physicsClientId=self.physics_client_id
        )

        self.detection_range = float(self.config.get("range", 15.0))  # m
        self.targets = []  # rempli par SimulationManager

        ip = self.config.get("ip", "localhost")
        port_out = self.config.get("port_out", 5557)
        self.setup_network(ip, port_out)

    def setup_network(self, ip, port_pub):
        """Socket ZMQ de publication du radar."""
        try:
            self.zmq_ctx = zmq.Context()
            self.pub_socket = self.zmq_ctx.socket(zmq.PUB)
            # Le radar est une station fixe : il fait le bind
            self.pub_socket.bind(f"tcp://{ip}:{port_pub}")
            print(f"[{self.name}] Radio active on tcp://{ip}:{port_pub}")
        except Exception as e:
            print(f"[{self.name}] ZMQ Error: {e}")

    def publish_detection(self, report, sim_time):
        """Publie les detections (JSON)."""
        if not report:
            return

        wrapper = {"radar_name": self.name, "data": report, "timestamp": sim_time}
        try:
            self.pub_socket.send_string("RADAR " + json.dumps(wrapper))
        except Exception as e:
            print(f"[{self.name}] Send Error: {e}")

    def think_and_act(self, sim_time):
        """Detecte les drones a portee, ajoute le bruit de mesure et publie."""
        detected_report = {}

        for agent in self.targets:
            if agent.bodyId == self.bodyId:
                continue

            target_pos, _ = p.getBasePositionAndOrientation(
                agent.bodyId, physicsClientId=self.physics_client_id
            )
            dist = np.linalg.norm(np.array(target_pos) - np.array(self.pos))

            if dist <= self.detection_range:
                meas_dist = dist + np.random.normal(self.range_noise_mean, self.range_noise_std)
                est_pos = np.array(target_pos) + np.random.normal(self.pos_noise_mean, self.pos_noise_std, 3)

                detected_report[agent.name] = {
                    "type": "radar",
                    "anchor_pos": self.pos,
                    "measured_dist": meas_dist,
                    "pos": est_pos.tolist(),
                }

        if detected_report:
            self.publish_detection(detected_report, sim_time)

        return detected_report
