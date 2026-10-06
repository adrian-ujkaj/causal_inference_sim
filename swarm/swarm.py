import json
import numpy as np
import pybullet as p
from entities.uav import UAV
import zmq
import threading
import random


class Swarm:
    """Essaim leader-suiveurs. Les suiveurs tiennent une formation en V derriere le leader
    (offsets dans le repere du leader), avec une correction de separation entre drones."""

    def __init__(
        self,
        agents: list[UAV],
        leader_name: str | None = None,
        min_sep: float = 0.6,
        avoid_gain: float = 0.5,
        formation_body_offsets: dict[str, np.ndarray] | None = None,
        port_in: int = 5556,
        port_out: int = 5557,
        ip: str = "localhost",
    ):
        if len(agents) == 0:
            raise ValueError("Swarm nécessite au moins un agent UAV.")
        self.sim_time = 0.0
        self.dt = agents[0].dt
        self.agents = agents
        self.name = "swarm 1"
        # broadcast a 50 Hz
        self.broadcast_interval = 0.02
        self.last_broadcast = -self.broadcast_interval
        self.prev_targets = {}
        self.max_formation_yaw_rate = 2.0  # rad/s, rotation maximale de la formation
        # Com latency
        self.perception_delay_mean = 0.1  # 100ms de retard
        self.perception_delay_std = 0.02  # +/- 20ms
        self.message_buffer = []
        agents_names = [a.name for a in agents if a.type == "uav"]
        # Choix du leader
        if leader_name is not None:
            leader_list = [a for a in agents if getattr(a, "name", "") == leader_name]
            if len(leader_list) == 0:
                raise ValueError(f"Aucun UAV avec name='{leader_name}' trouvé pour le leader.")
            self.leader = leader_list[0]
            self.leader.leader = True
        else:
            # par défaut, le premier UAV de la liste est le leader
            self.leader = agents[0]
            self.leader.leader = True
        self.agents_data = {}
        for agent in self.agents:
            # start_orn est un quaternion
            yaw0 = float(p.getEulerFromQuaternion(agent.start_orn)[2])
            self.agents_data[agent.name] = {
                "name": agent.name,
                "pos": list(agent.start_pos),
                "vel": [0, 0, 0],
                "yaw": yaw0,
                "sim_time": 0.0,
            }
            agent.set_swarm_activate()  # Indique que l'agent fait partie d'un essaim
            agent.swarm_name = agents_names
        self.followers_future_state = self.agents_data.copy()
        # Followers = tous les autres
        self.followers: list[UAV] = [a for a in agents if a is not self.leader and a.type == "uav"]
        self.physics_client_id = self.leader.physics_client_id

        self.radar = [a for a in agents if a.type == "radar"]

        # Separation
        self.min_sep = float(min_sep)
        self.avoid_gain = float(avoid_gain)

        # Reseau
        self.port_in = port_in
        self.port_out = port_out
        self.ip = ip
        self.init_proxy()
        self.setup_swarm_com()
        # Les drones se connectent au proxy (seul le proxy fait le bind)
        for a in self.agents:
            a.setup_network_swarm(self.ip, self.port_in, self.port_out)
        # Offsets de formation
        self.formation_body_offsets: dict[str, np.ndarray] = {}

        if formation_body_offsets is not None:
            for f in self.followers:
                if f.name not in formation_body_offsets:
                    raise ValueError(
                        f"formation_body_offsets ne contient pas d'offset pour follower '{f.name}'."
                    )
                off = np.array(formation_body_offsets[f.name], dtype=float)
                if off.shape != (3,):
                    raise ValueError("Chaque offset doit être un vecteur 3D [x, y, z].")
                self.formation_body_offsets[f.name] = off
        else:
            self._assign_default_triangular_offsets()

        print(
            f"[Swarm] Essaim créé avec leader='{self.leader.name}', "
            f"{len(self.followers)} follower(s). "
            f"(min_sep={self.min_sep:.2f}, avoid_gain={self.avoid_gain:.2f})"
        )

    def _assign_default_triangular_offsets(self):
        """Formation en V derriere le leader. Les suiveurs sont places selon leur
        position laterale initiale pour ne pas se croiser au decollage."""
        sx, sy = 1.0, 1.0
        followers = list(self.followers)
        n = len(followers)
        if n == 0:
            return
        slots = []
        for k in range(n):
            r = k // 2 + 1
            side = -1.0 if k % 2 == 0 else 1.0
            if n % 2 == 1 and k == n - 1:
                side = 0.0  # dernier suiveur seul : dans l'axe
            slots.append(np.array([-r * sx, side * r * sy, 0.0]))
        yaw0 = float(p.getEulerFromQuaternion(self.leader.start_orn)[2])
        c, s_ = np.cos(yaw0), np.sin(yaw0)
        lead0 = np.asarray(self.leader.start_pos, dtype=float)

        def lateral(f):
            d = np.asarray(f.start_pos, dtype=float) - lead0
            return -s_ * d[0] + c * d[1]  # coordonnee y dans le repere du leader

        followers.sort(key=lateral)
        slots.sort(key=lambda o: (o[1], o[0]))
        for f, off in zip(followers, slots):
            self.formation_body_offsets[f.name] = off
        return

    def init_proxy(self):
        self.proxy_ready = threading.Event()
        self.proxy_error = None
        self.proxy_ctx = zmq.Context()

        def run_proxy():
            ctx = self.proxy_ctx
            frontend = backend = None
            try:
                # Entree XSUB, sortie XPUB
                frontend = ctx.socket(zmq.XSUB)
                frontend.bind(f"tcp://*:{self.port_in}")

                backend = ctx.socket(zmq.XPUB)
                backend.bind(f"tcp://*:{self.port_out}")

                print(f"[Swarm Network] Proxy démarré (In: {self.port_in} -> Out: {self.port_out})")
                self.proxy_ready.set()

                # Bloquant ; les sockets restent dans ce thread
                zmq.proxy(frontend, backend)

            except zmq.ContextTerminated:
                pass  # arret normal (cleanup)
            except Exception as e:
                self.proxy_error = e
                print(f"[Swarm Network] Erreur dans le proxy : {e}")
            finally:
                for sock in (frontend, backend):
                    if sock is not None:
                        sock.close(linger=0)
                self.proxy_ready.set()

        self.proxy_thread = threading.Thread(target=run_proxy, daemon=True)
        self.proxy_thread.start()
        # Erreur explicite si le proxy n'a pas demarre (port deja pris)
        self.proxy_ready.wait(timeout=2.0)
        if self.proxy_error is not None:
            raise RuntimeError(
                f"Proxy de l'essaim indisponible (ports {self.port_in}/{self.port_out}) : {self.proxy_error}"
            )

    def setup_swarm_com(self):
        """
        Configure le Swarm pour qu'il écoute aussi son propre réseau
        (comme un drone client).
        """
        self.client_ctx = zmq.Context()
        self.sub_socket = self.client_ctx.socket(zmq.SUB)
        self.pub_socket = self.client_ctx.socket(zmq.PUB)
        # Proxy local
        self.sub_socket.connect(f"tcp://localhost:{self.port_out}")

        # On s'abonne à tout (ou au topic 'SWARM')
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 1)  # Timeout 1ms pour ne pas bloquer

        self.pub_socket.connect(f"tcp://localhost:{self.port_in}")
        self.pub_socket.setsockopt(zmq.LINGER, 1)  # Fermeture immédiate

    def broadcast_state(self):
        """Envoie la position et vitesse actuelle au réseau"""
        # Envoi sur le topic 'SWARM'
        # Format: "TOPIC JSON"
        self.pub_socket.send_string("SWARM " + json.dumps(self.agents_data))

    def broadcast_future_pos(self):
        """Envoie la position future calculée des followers au réseau"""
        # Envoi sur le topic 'FUTURE_POS'
        self.pub_socket.send_string("FUTURE_POS " + json.dumps(self.followers_future_state))
        pass

    def listen_swarm(self):
        """
        Vérifie la boite aux lettres et met à jour la liste des voisins.
        À appeler à chaque step.
        """
        while True:
            try:
                # Lecture non-bloquante
                msg = self.sub_socket.recv_string()
                delay = max(0, random.gauss(self.perception_delay_mean, self.perception_delay_std))
                visible_time = self.sim_time + delay
                self.message_buffer.append((visible_time, msg))
            except zmq.Again:
                # Plus de messages
                break
            except Exception as e:
                print(f"Erreur réseau sur {self.name}: {e}")
                break

        buffer_remaining = []

        for target_time, msg in self.message_buffer:
            if self.sim_time >= target_time:
                if " " in msg:
                    topic, json_str = msg.split(" ", 1)
                    try:
                        if topic == "State":
                            data = json.loads(json_str)
                            if "name" in data:
                                self.agents_data[data["name"]] = data
                    except ValueError:
                        pass
            else:
                buffer_remaining.append((target_time, msg))
        # On ne garde que les messages pas encore delivres
        self.message_buffer = buffer_remaining

    def cleanup(self):
        """Ferme les sockets et arrete le proxy."""
        self.sub_socket.close(linger=0)
        self.pub_socket.close(linger=0)
        self.client_ctx.term()
        try:
            self.proxy_ctx.term()  # zmq.proxy se termine sur ContextTerminated
        except Exception:
            pass

    def update(self):
        """Calcule et diffuse les cibles de formation des suiveurs (50 Hz).
        La position du leader arrive avec retard, on l'extrapole avec sa vitesse."""
        self.sim_time += self.dt
        if (self.sim_time - self.last_broadcast) < self.broadcast_interval:
            return
        if len(self.followers) == 0:
            self.last_broadcast = self.sim_time
            return
        self.listen_swarm()

        state_leader = self.agents_data.get(self.leader.name)
        if state_leader is None:
            return

        # Pas reel depuis la diffusion precedente
        dt_swarm = self.sim_time - self.last_broadcast
        if not np.isfinite(dt_swarm) or dt_swarm <= 0 or dt_swarm > 1.0:
            dt_swarm = self.broadcast_interval

        pos_leader = np.asarray(state_leader["pos"], dtype=float)
        vel_leader = np.asarray(state_leader.get("vel", [0, 0, 0]), dtype=float)
        age = self.sim_time - float(state_leader.get("sim_time", self.sim_time))
        age = float(np.clip(age, 0.0, 0.5))
        pos_leader_pred = pos_leader + vel_leader * age
        target_yaw_leader = float(state_leader["yaw"])

        # Lissage du lacet de formation (passe-bas, vitesse de rotation bornee)
        if not hasattr(self, "smooth_swarm_yaw"):
            self.smooth_swarm_yaw = target_yaw_leader
        diff_yaw = np.arctan2(
            np.sin(target_yaw_leader - self.smooth_swarm_yaw),
            np.cos(target_yaw_leader - self.smooth_swarm_yaw),
        )
        alpha_yaw = 0.3
        max_step = self.max_formation_yaw_rate * dt_swarm
        step_yaw = float(np.clip(diff_yaw * alpha_yaw, -max_step, max_step))
        self.smooth_swarm_yaw = float(
            np.arctan2(np.sin(self.smooth_swarm_yaw + step_yaw), np.cos(self.smooth_swarm_yaw + step_yaw))
        )
        omega_z = step_yaw / dt_swarm

        cy, sy = np.cos(self.smooth_swarm_yaw), np.sin(self.smooth_swarm_yaw)
        R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

        # Positions vraies des drones pour la separation (coordinateur centralise)
        positions = {}
        for a in [self.leader] + self.followers:
            st = a.get_ground_truth_state()
            if st:
                positions[a.name] = np.asarray(st["pos"], dtype=float)

        for follower in self.followers:
            off_body = self.formation_body_offsets.get(follower.name)
            if off_body is None:
                continue
            off_world = R_yaw @ off_body
            base_target = pos_leader_pred + off_world

            correction = np.zeros(3)
            for other in [self.leader] + self.followers:
                if other is follower:
                    continue
                pos_o = positions.get(other.name)
                if pos_o is None:
                    continue
                diff = base_target - pos_o
                diff[2] = 0.0
                dist = float(np.linalg.norm(diff))
                if dist < self.min_sep:
                    # Cible confondue avec un voisin : direction arbitraire
                    u = diff / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                    correction += (self.min_sep - dist) * u

            final_target = base_target + self.avoid_gain * correction
            final_target[2] = base_target[2]
            target_vel = vel_leader + np.cross([0.0, 0.0, omega_z], off_world)

            self.followers_future_state[follower.name] = {
                "pos": final_target.tolist(),
                "vel": target_vel.tolist(),
                "yaw": self.smooth_swarm_yaw,
                "t": self.sim_time,  # horodatage pour l'extrapolation
            }

        self.broadcast_state()
        self.broadcast_future_pos()
        self.last_broadcast = self.sim_time
