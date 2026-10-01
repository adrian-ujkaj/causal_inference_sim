import math
import numpy as np
import pybullet as p


class Sensor:
    """Classe de base pour les capteurs."""
    def __init__(self):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("La méthode 'measure' doit être implémentée.")


class GNSSensor(Sensor):
    """
    Capteur GPS simple :
    - position_noise_std : écart-type du bruit sur la position (m)
    - velocity_noise_std : écart-type du bruit sur la vitesse (m/s)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.pos_noise_std = float(config.get("position_noise_std", 0.0))
        self.vel_noise_std = float(config.get("velocity_noise_std", 0.0))

        # Optional GNSS degradation / jamming schedule (for causal case studies)
        self.pos_noise_std_base = self.pos_noise_std
        self.vel_noise_std_base = self.vel_noise_std
        self.jam_start = config.get("jam_start", None)
        self.jam_end = config.get("jam_end", None)
        self.jam_pos_noise_std = config.get("jam_position_noise_std", config.get("jam_pos_noise_std", None))
        self.jam_vel_noise_std = config.get("jam_velocity_noise_std", config.get("jam_vel_noise_std", None))
        self.jam_multiplier = float(config.get("jam_multiplier", 1.0))

        if self.pos_noise_std < 0.0:
            self.pos_noise_std = 0.0
        if self.vel_noise_std < 0.0:
            self.vel_noise_std = 0.0

    def measure(
        self,
        ground_truth_position: np.ndarray,
        ground_truth_velocity: np.ndarray,
        t: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Renvoie (position_mesurée, vitesse_mesurée) avec bruit gaussien.
        Optionnel: t (temps sim) pour appliquer une fenêtre de brouillage (jam).
        """
        pos_std = self.pos_noise_std
        vel_std = self.vel_noise_std

        # Apply jamming window if configured
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

        pos_noise = np.random.normal(0.0, pos_std, 3)
        vel_noise = np.random.normal(0.0, vel_std, 3)

        meas_pos = ground_truth_position + pos_noise
        meas_vel = ground_truth_velocity + vel_noise

        return meas_pos, meas_vel


# Dans entities/sensor.py

class IMUSensor:
    def __init__(self, config_dict=None):
        self.last_vel = np.zeros(3)
        self.dt = 1/100 # Supposé
        self.g_vector = np.array([0, 0, 9.81])

    def measure(self, vel, orn_q):
        """
        Simule un accéléromètre + gyroscope
        Retourne : acc_body, gyro_body
        """
        # 1. Calcul Accélération Monde (a = dv/dt)
        # (C'est une approximation discrete)
        acc_world = (np.array(vel) - self.last_vel) / self.dt
        self.last_vel = np.array(vel)
        
        # 2. Ajout de la "pesanteur ressentie" (Proper Acceleration)
        # Un accéléromètre mesure (a - g). Comme g pointe vers le bas (-9.81), 
        # a_mesure = a_monde - (-9.81) = a_monde + 9.81
        acc_proper_world = acc_world + self.g_vector
        
        # 3. Rotation vers Body Frame (Monde -> Drone)
        # On utilise la matrice inverse de rotation
        R_world_to_body = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3,3).T
        acc_body = R_world_to_body @ acc_proper_world
        
        # Ajout de bruit (facultatif)
        acc_body += np.random.normal(0, 0.1, 3) # Bruit blanc
        
        # Gyro (Vitesse angulaire, ici on met 0 ou la vraie pour simplifier)
        gyro_body = np.zeros(3) 
        
        return acc_body, gyro_body

class LidarSensor:
    def __init__(self, config: dict):
        """
        Lidar paramétrable avec un champ de vision (FOV) conique.
        """
        self.max_distance = float(config.get("max_distance", 5.0))
        self.angle_resolution = float(config.get("angle_resolution", 2.0))
        
        # FOV en degrés, convertis en demi-angles
        fov_h = float(config.get("fov_horizontal", 90.0))
        fov_v = float(config.get("fov_vertical", 30.0))
        
        self.half_fov_h = fov_h / 2.0
        self.half_fov_v = fov_v / 2.0
        
        # Pré-génération des vecteurs de rayons dans le repère LOCAL du drone
        # Axe X = Devant, Y = Gauche, Z = Haut
        self.local_rays = self._generate_local_rays()
        print(f"[LidarSensor] Initialisé : {len(self.local_rays)} rayons (FOV H:{fov_h}°, V:{fov_v}°)")

    def _generate_local_rays(self):
        rays = []
        # On balaie de gauche à droite (-fov_h/2 à +fov_h/2)
        for az in np.arange(-self.half_fov_h, self.half_fov_h, self.angle_resolution):
            # On balaie de bas en haut (-fov_v/2 à +fov_v/2)
            for el in np.arange(-self.half_fov_v, self.half_fov_v, self.angle_resolution):
                
                # Conversion degrés -> radians
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Coordonnées sphériques vers Cartésiennes (X est devant)
                # x = cos(el) * cos(az)
                # y = cos(el) * sin(az)
                # z = sin(el)
                x = np.cos(el_rad) * np.cos(az_rad)
                y = np.cos(el_rad) * np.sin(az_rad)
                z = np.sin(el_rad)
                
                # Normalisation (juste par sécurité)
                v = np.array([x, y, z])
                v = v / np.linalg.norm(v)
                rays.append(v)
        
        return np.array(rays)

    def measure(self, position, roll, yaw, pitch):
        """
        Effectue un raycast par lot (batch) dans PyBullet.
        Args:
            position: [x, y, z] du drone
            roll, yaw, pitch: orientation actuelle en radians
        Returns:
            points: Liste de np.array [x, y, z] des impacts détectés
        """
        # 1. Calcul de la matrice de rotation du drone
        # PyBullet utilise l'ordre [roll, pitch, yaw] pour les quaternions Euler
        orn_q = p.getQuaternionFromEuler([roll, pitch, yaw])
        rot_matrix = p.getMatrixFromQuaternion(orn_q)
        
        # Transformation de la matrice plate (9,) en (3,3)
        R = np.array(rot_matrix).reshape(3, 3)
        
        # 2. Rotation de tous les rayons locaux vers le monde
        # Formule: Ray_Monde = R * Ray_Local
        # Optimisation vectorielle : (N,3) dot (3,3) -> (N,3)
        # Note: on utilise transpose pour aligner les dimensions correctement
        world_rays_dir = self.local_rays @ R.T 
        
        # 3. Préparation des positions de départ et d'arrivée
        num_rays = len(world_rays_dir)
        ray_froms = np.tile(position, (num_rays, 1))
        ray_tos = ray_froms + world_rays_dir * self.max_distance
        
        # 4. Raycast PyBullet (Batch = très rapide)
        results = p.rayTestBatch(ray_froms, ray_tos)
        
        # 5. Filtrage des impacts
        detected_points = []
        for _, res in enumerate(results):
            # res structure: (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
            hit_id = res[0]
            if hit_id >= 0: # Si on a touché un objet (id >= 0)
                hit_pos = np.array(res[3])
                if hit_pos[2]>0.01:  # Filtre pour éviter les points trop proches
                    hit_id = [round(coord, 3) for coord in hit_pos]
                    detected_points.append(hit_pos)
                
        return detected_points
    
class HeightSensor:
    def __init__(self, physics_client_id, noise_std=0.01, max_range=4.0):
        """
        Simule un capteur de distance orienté vers le bas (Down-facing Lidar/Sonar).
        """
        self.client_id = physics_client_id
        self.noise_std = noise_std
        self.max_range = max_range

    def measure(self, pos, orn_q):
        """
        Retourne la distance mesurée vers le sol (ou None si hors de portée).
        """
        # 1. Calcul du vecteur direction (Le capteur pointe vers le "Bas" du drone)
        # En repère monde, le bas du drone change si le drone penche (Roll/Pitch)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn_q)).reshape(3, 3)
        # Le vecteur "Bas" dans le repère du drone est [0, 0, -1]
        # On le tourne dans le repère monde
        down_vec_world = rot_mat @ np.array([0, 0, -1])

        # 2. Raycast (Tir du rayon)
        start = np.array(pos)
        end = start + (down_vec_world * self.max_range)
        
        results = p.rayTest(start, end, physicsClientId=self.client_id)
        # results[0] contient [objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal]
        
        hit_fraction = results[0][2]
        
        # 3. Traitement
        if hit_fraction == 1.0: # Rien touché
            return None # Trop haut pour le capteur
        
        # Distance réelle = hit_fraction * max_range
        dist = hit_fraction * self.max_range
        
        # Ajout du bruit
        dist += np.random.normal(0, self.noise_std)
        
        # Protection valeurs négatives
        return max(0.0, dist)