import math
import numpy as np
from scipy.spatial import KDTree
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import pybullet as p


class HeightmapAStar:
    def __init__(self, config, resolution=0.25):
        self.res = resolution
        self.bounds = config.get("world_bounds", {'x': [-500, 500], 'y': [-500, 500], 'z': [0, 100]})
        self.min_x, self.min_y = self.bounds['x'][0], self.bounds['y'][0]

        # Dimensions de la grille
        self.width = int((self.bounds['x'][1] - self.min_x) / self.res)
        self.height = int((self.bounds['y'][1] - self.min_y) / self.res)

        # Heightmap en float32
        self.h_map = np.zeros((self.width, self.height), dtype=np.float32)

        # Mouvement : Interdit de frôler les coins (Only when no obstacle)
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)

        # Récupération de la marge en MÈTRES (ex: 1.0m)
        self.safety_margin = config.get("safety_margin", 1.0)

    def build_from_buildings(self, buildings):
        """
        Construit la carte en appliquant l'inflation GÉOMÉTRIQUE.
        On augmente la taille physique des bâtiments avant de les mettre sur la grille.
        """
        source_data = buildings.values() if isinstance(buildings, dict) else buildings
        centers = []

        for data in source_data:
            center = data["center"]
            centers.append(center[:2])

            # Dimensions réelles
            h = data["height"]
            real_w = data["width"]
            real_l = data["length"]

            # --- INFLATION GÉOMÉTRIQUE (La Solution) ---
            # On ajoute la marge de sécurité des deux côtés.
            # Si marge = 1m, l'immeuble devient 2m plus large et plus long.
            effective_w = real_w + (2.0 * self.safety_margin)
            effective_l = real_l + (2.0 * self.safety_margin)

            # Calcul des indices avec les dimensions gonflées
            # Axe X (Width)
            # floor / ceil : une cellule partiellement couverte est bloquee. Avec
            # int() (troncature) la derniere cellule etait perdue et la marge de
            # 0.25 m se reduisait a 8 cm d'un mur, moins que la demi-envergure
            # du drone (7.5 cm) plus l'erreur de suivi.
            x_min = int(math.floor((center[0] - effective_w / 2 - self.min_x) / self.res))
            x_max = int(math.ceil((center[0] + effective_w / 2 - self.min_x) / self.res))

            # Axe Y (Length)
            y_min = int(math.floor((center[1] - effective_l / 2 - self.min_y) / self.res))
            y_max = int(math.ceil((center[1] + effective_l / 2 - self.min_y) / self.res))

            # Clamping pour ne pas sortir de la carte
            x0 = max(0, x_min)
            x1 = min(self.width, x_max)
            y0 = max(0, y_min)
            y1 = min(self.height, y_max)

            # Remplissage
            if x0 < x1 and y0 < y1:
                self.h_map[x0:x1, y0:y1] = np.maximum(self.h_map[x0:x1, y0:y1], h)

        if centers:
            self.building_tree = KDTree(np.array(centers))

    def custom_heightmap(self):
        z_start = self.bounds['z'][1] + 100
        z_end = self.bounds['z'][0] - 1.0  # Un peu en dessous du sol

        # On parcourt la grille par ligne (X) pour envoyer des paquets de rayons (Y)
        for i in range(self.width):
            ray_starts = []
            ray_ends = []

            # Calcul du X monde actuel
            curr_x = self.min_x + i * self.res

            for j in range(self.height):
                # Calcul du Y monde actuel
                curr_y = self.min_y + j * self.res

                ray_starts.append([curr_x, curr_y, z_start])
                ray_ends.append([curr_x, curr_y, z_end])

            # Envoi de la ligne complète à PyBullet
            results = p.rayTestBatch(ray_starts, ray_ends)

            # Extraction des altitudes
            for j, res in enumerate(results):
                hit_fraction = res[2]
                hit_pos = res[3]

                if hit_fraction < 1.0:
                    # On stocke l'altitude Z de l'impact
                    self.h_map[i, j] = hit_pos[2]
                else:
                    # Si rien n'est touché, on considère le sol (Z=0)
                    self.h_map[i, j] = 0.0

            if i % 400 == 0:  # Barre de progression simple
                print(f"Progression : {int(i / self.width * 100)}%")

        print("Heightmap générée avec succès.")

    def compute_repulsive_force(self, current_pos, safety_radius, max_force, swarm_active, leader, swarm_pos):
        force_vec = np.array([0.0, 0.0, 0.0])
        k_obs = 0.5
        rows, cols = self.h_map.shape

        # Conversion position monde -> index grille (avec (0,0) au centre)
        # On utilise // pour obtenir un entier (floor division)
        ix = int(current_pos[0] / self.res + rows / 2)
        iy = int(current_pos[1] / self.res + cols / 2)
        window_px = int(safety_radius / self.res)

        # Bornes de la fenêtre (sécurisées pour ne pas sortir de la matrice)
        x_min, x_max = max(0, ix - window_px), min(rows, ix + window_px + 1)
        y_min, y_max = max(0, iy - window_px), min(cols, iy + window_px + 1)

        # Extraction de la zone locale
        local_h_map = self.h_map[x_min:x_max, y_min:y_max]

        # Calcul des coordonnées réelles de chaque pixel de la zone extraite
        # (On fait l'opération inverse pour retrouver le mètre depuis l'index)
        x_range = (np.arange(x_min, x_max) - rows / 2) * self.res
        y_range = (np.arange(y_min, y_max) - cols / 2) * self.res
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')

        DX = current_pos[0] - X
        DY = current_pos[1] - Y
        Dist_horizontale = np.sqrt(DX**2 + DY**2)

        # --- LOGIQUE DE DÉCISION ---

        # Filtre de base : points dans le rayon de sécurité
        mask_near = (Dist_horizontale < safety_radius) & (Dist_horizontale > 0.1)

        # CAS A : Le point est AU-DESSUS du drone (Mur/Obstacle haut) -> On pousse sur le côté (XY)
        mask_wall = mask_near & (local_h_map >= current_pos[2])
        n_wall_pts = np.sum(mask_wall)

        if n_wall_pts > 0:
            # On calcule la force moyenne pour ne pas exploser les compteurs
            # Utilisation d'une décroissance quadratique pour plus de douceur
            mags = (1.0 - (Dist_horizontale[mask_wall] / safety_radius)) ** 2
            weights = mags / (Dist_horizontale[mask_wall] + 0.01)
            sum_weights = np.sum(weights)

            # Vecteur de répulsion pur (pousse vers l'arrière)
            fx_rep = np.sum((DX[mask_wall] / Dist_horizontale[mask_wall]) * weights * max_force) / sum_weights
            fy_rep = np.sum((DY[mask_wall] / Dist_horizontale[mask_wall]) * weights * max_force) / sum_weights

            k_glide = 0.5  # Ajustez entre 0.2 et 0.8
            fx_glide = -fy_rep * k_glide
            fy_glide = fx_rep * k_glide

            # 3. Application de la force combinée
            force_vec[0] += (fx_rep + fx_glide) * k_obs
            force_vec[1] += (fy_rep + fy_glide) * k_obs

        # CAS B : Le point est EN-DESSOUS du drone (Sol/Toit) -> On pousse vers le haut (Z)
        # On ne considère que si on est proche verticalement (ex: marge de 2.0m)
        v_margin = 2.5  # Zone d'influence verticale (mètres)
        ground_threshold = 1  # En dessous de 2m, on considère que c'est le sol

        # Filtre de base : points sous le drone et dans la zone d'influence
        mask_below = mask_near & (local_h_map < current_pos[2]) & (local_h_map > current_pos[2] - v_margin)

        if np.any(mask_below):
            # CAS B1 : C'est le SOL (Altitude basse)
            mask_is_ground = mask_below & (local_h_map < ground_threshold)

            # CAS B2 : C'est un IMMEUBLE / OBSTACLE (Altitude haute mais sous le drone)
            mask_is_roof = mask_below & (local_h_map >= ground_threshold)

            # Comportement pour le SOL : Force douce pour le maintien d'altitude
            if np.any(mask_is_ground):
                dist_v_ground = current_pos[2] - local_h_map[mask_is_ground]
                mag_ground = (1.0 - (dist_v_ground / v_margin)) ** 2
                # On applique un gain plus faible (k_ground) pour éviter que le drone ne "saute"
                force_vec[2] += np.mean(mag_ground * max_force) * 0.3

            # Comportement pour un TOIT : Force plus ferme pour éviter la collision
            if np.any(mask_is_roof):
                dist_v_roof = current_pos[2] - local_h_map[mask_is_roof]
                mag_roof = (1.0 - (dist_v_roof / v_margin)) ** 2
                # On applique une force plus importante (k_roof) car l'impact est plus dangereux
                force_vec[2] += np.mean(mag_roof * max_force) * 1.2

                # OPTIONNEL : Si c'est un immeuble, on peut aussi ajouter une petite
                # force horizontale (XY) pour que le drone s'écarte des bords du toit
                force_vec[0] += (
                    np.mean((DX[mask_is_roof] / Dist_horizontale[mask_is_roof]) * mag_roof * max_force) * 0.2
                )
                force_vec[1] += (
                    np.mean((DY[mask_is_roof] / Dist_horizontale[mask_is_roof]) * mag_roof * max_force) * 0.2
                )
        if swarm_active and not leader:
            for _, other_pos in swarm_pos.items():
                diff = current_pos - other_pos
                dist_uav = np.linalg.norm(diff)
                if dist_uav < safety_radius:
                    mag = 1.0 - (dist_uav / safety_radius)
                    force_vec += ((diff / dist_uav) * mag * max_force) / 2

        # Normalisation finale
        total_norm = np.linalg.norm(force_vec)
        if total_norm > max_force:
            force_vec = (force_vec / total_norm) * max_force

        return force_vec

    def plan(self, start_pos, goal_pos):
        # 1. Conversion positions -> indices
        sx, sy = self._pos_to_idx(start_pos)
        gx, gy = self._pos_to_idx(goal_pos)

        # 2. Création de la Grille Binaire
        # Altitude de planification : la plus BASSE du trajet (le chemin est vole
        # vers l'altitude du but), avec un plancher. Avec l'altitude de depart
        # seule, un drone au sol (z <= 0) voyait toutes les cellules bloquees, et
        # un drone haut ignorait des toits qu'il croiserait en descendant.
        drone_z = max(min(float(start_pos[2]), float(goal_pos[2])), 0.3)
        # True = Libre (Walkable), False = Mur gonflé
        walkable_grid = (self.h_map < drone_z).astype(int)

        # --- CORRECTION CRASH : Transposition (.T) ---
        # La librairie pathfinding lit [y][x], numpy est [x][y]
        grid = Grid(matrix=walkable_grid.T)

        # 3. Vérification des limites
        if not grid.inside(sx, sy) or not grid.inside(gx, gy):
            print("[A*] Erreur : Départ ou Arrivée hors de la carte.")
            return None

        # 4. Gestion des Nœuds Départ/Arrivée
        node_start = grid.node(sx, sy)
        node_end = grid.node(gx, gy)

        # Force le départ walkable (pour décoller même si on est dans la zone de sécurité)
        node_start.walkable = True

        if not node_end.walkable:
            print("[A*] Cible inaccessible ")
            return None

        # 5. Calcul du chemin
        path, runs = self.finder.find_path(node_start, node_end, grid)

        if not path or len(path) < 2:
            return None

        # 6. Lissage et Conversion
        smoothed_nodes = self._smooth_path(path, grid)

        # --- CORRECTION "SUBSCRIPTABLE" : Utilisation de p.x et p.y ---
        world_path = [self._idx_to_pos(p.x, p.y, goal_pos[2]) for p in smoothed_nodes]
        return world_path

    def _smooth_path(self, path_nodes, grid):
        """Simplification du chemin (String Pulling)"""
        if len(path_nodes) < 3:
            return path_nodes
        smoothed = [path_nodes[0]]
        curr_idx = 0

        while curr_idx < len(path_nodes) - 1:
            best_next = curr_idx + 1
            for i in range(len(path_nodes) - 1, curr_idx, -1):
                if self._line_of_sight(grid, path_nodes[curr_idx], path_nodes[i]):
                    best_next = i
                    break
            curr_idx = best_next
            smoothed.append(path_nodes[curr_idx])

        return smoothed

    def _line_of_sight(self, grid, node_a, node_b):
        x0, y0 = node_a.x, node_a.y
        x1, y1 = node_b.x, node_b.y
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if not grid.node(x, y).walkable:
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

    def _pos_to_idx(self, pos):
        nx = int((pos[0] - self.min_x) / self.res)
        ny = int((pos[1] - self.min_y) / self.res)
        return max(0, min(nx, self.width - 1)), max(0, min(ny, self.height - 1))

    def _idx_to_pos(self, x, y, z):
        # Retourne le CENTRE de la case pour éloigner des murs
        return np.array(
            [x * self.res + self.min_x + (self.res / 2.0), y * self.res + self.min_y + (self.res / 2.0), z]
        )
