import pybullet as p
import numpy as np
import os


class Agent:
    def __init__(self, urdf_path, start_pos, start_orn_q, physics_client_id, dt: float):
        self.p = p
        self.dt = dt
        self.physics_client_id = physics_client_id

        # Chemin relatif -> absolu pour éviter les erreurs
        if not os.path.exists(urdf_path):
            base = os.path.dirname(os.path.abspath(__file__))
            # On remonte de 'entities' vers 'Stage'
            urdf_path = os.path.join(os.path.dirname(base), urdf_path)

        # Flag indispensable pour la physique des drones
        self.bodyId = self.p.loadURDF(
            fileName=urdf_path,
            basePosition=start_pos,
            baseOrientation=start_orn_q,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client_id,
        )

    def get_ground_truth_state(self):
        if not self.p.isConnected(self.physics_client_id):
            return {}
        pos, orn = self.p.getBasePositionAndOrientation(self.bodyId, physicsClientId=self.physics_client_id)
        vel, ang_vel = self.p.getBaseVelocity(self.bodyId, physicsClientId=self.physics_client_id)
        return {
            "pos": np.array(pos),
            "orn_q": np.array(orn),
            "vel": np.array(vel),
            "ang_vel": np.array(ang_vel),
        }
