"""
Environnement Gymnasium SIMPLIFIÉ pour robot 4 roues dans corridor.
Version nettoyée avec grille 0.05m et CNN unique.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


class CorridorEnv(gym.Env):
    """
    Robot 4 roues naviguant un corridor avec trous et rampes.
    
    Observation:
        - Position robot (x, y, z): 3
        - Vitesse robot (vx, vy, vz): 3  
        - Bounding box coins (4 coins × 2 coords): 8
        - Grille environnement 120×80: 9600 (0=sol, 0.5=rampe, 1=trou)
    
    Action (4 valeurs):
        - Couple roue FL, FR, RL, RR
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, corridor_xml="corridor_100.xml", max_steps=3000):
        super().__init__()
        
        # Charger modèle MuJoCo
        self.model = self._build_model("four_wheels_robot.xml", corridor_xml)
        self.data = mujoco.MjData(self.model)
        
        # Paramètres SIMPLIFIÉS
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_size = 0.05  # 5cm par cellule
        
        # Grille vision: 6m (4m devant + 2m derrière) × 3m largeur
        self.vision_length = 6.0  # 4m devant + 2m derrière
        self.vision_width = 3.0   # 3m largeur (exactement la largeur du couloir)
        self.grid_rows = int(self.vision_length / self.cell_size)  # 120 lignes
        self.grid_cols = int(self.vision_width / self.cell_size)   # 60 colonnes
        
        # Dimensions robot (bounding box) - ALIGNÉES sur la grille
        self.robot_length = 0.60  # 12 cellules exactement (12 × 0.05)
        self.robot_width = 0.40   # 8 cellules exactement (8 × 0.05)
        
        # Historique des positions pour anticipation (AVANT les espaces)
        self.history_interval = 20  # Sauvegarder position tous les 20 steps
        self.history_length = 5     # Garder les 5 dernières positions
        self.position_history = []  # Buffer des positions (x, y, angle)
        
        # Période de stabilisation
        self.stabilization_steps = 20  # Pas d'actions pendant 20 steps
        
        # Construire carte
        self.cell_map = self._build_cell_map()
        
        # Espaces - UN SEUL CNN + historique
        history_size = self.history_length * 8  # 5 positions × 8 coords (4 coins × 2) = 40
        obs_size = 6 + 8 + history_size + (self.grid_rows * self.grid_cols)  # 6 + 8 + 40 + 7200 = 7254
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        
        # État
        self.step_count = 0
        self.prev_x = 0.0
        
        # Détection blocage
        self.stuck_check_interval = 50
        self.stuck_min_advance = 0.3
        self.stuck_x_checkpoint = 0.0
        self.stuck_counter = 0
        self.stuck_max_count = 2
        
        # IDs corps MuJoCo
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # SPAWN FIXE à x=1
        spawn_x = 1.0  # Toujours au début
        spawn_y = np.random.uniform(-1.0, 1.0)  # Y aléatoire entre -1 et 1
        spawn_angle = np.random.uniform(-np.pi/6, np.pi/6)  # Angle aléatoire
        
        # Position
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = 0.45
        
        # Orientation (quaternion Z-axis)
        self.data.qpos[3] = np.cos(spawn_angle / 2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(spawn_angle / 2)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.prev_x = self.data.qpos[0]
        self.stuck_counter = 0
        self.stuck_x_checkpoint = self.data.qpos[0]  # Reset checkpoint aussi
        
        # Reset historique des positions
        self.position_history = []
        self._update_position_history()  # Ajouter position initiale
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Période de stabilisation : pas d'actions pendant les premiers steps
        if self.step_count < self.stabilization_steps:
            # Actions nulles pendant la stabilisation
            self.data.ctrl[:] = 0.0
        else:
            # Appliquer couple roues normalement
            action = np.clip(action, -1.0, 1.0) * 20.0
            self.data.ctrl[:] = action
        
        # Simuler (4 substeps)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Mettre à jour historique des positions
        if self.step_count % self.history_interval == 0:
            self._update_position_history()
        
        # Récompense et terminaison
        reward, terminated, info = self._compute_reward()
        truncated = self.step_count >= self.max_steps
        
        info.update(self._get_info())
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation avec historique des positions pour anticipation."""
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        
        # Bounding box du robot (4 coins dans repère grille)
        bbox_corners = self._get_robot_bbox_corners(pos[0], pos[1])
        
        # Historique des positions (5 positions × 3 coords = 15 valeurs)
        position_history = self._get_position_history_obs()
        
        # Grille environnement
        grid = self._get_grid_obs(pos[0], pos[1])
        
        return np.concatenate([
            pos,                    # 3 valeurs (position actuelle)
            vel,                    # 3 valeurs (vitesse actuelle)
            bbox_corners,           # 8 valeurs (4 coins × 2 coords actuels)
            position_history,       # 40 valeurs (5 positions × 8 coords relatives)
            grid.flatten()          # 7200 valeurs (120×60)
        ]).astype(np.float32)       # Total: 6 + 8 + 40 + 7200 = 7254 valeurs
    
    def _get_robot_bbox_corners(self, robot_x, robot_y):
        """Position des 4 coins de la bounding box dans le repère relatif de la grille.
        
        La bounding box est un rectangle FIXE de 0.6m × 0.4m (12×8 cellules) qui pivote
        avec l'angle du robot. Les dimensions restent constantes quelle que soit l'orientation.
        """
        # Récupérer orientation robot (quaternion → angle autour de Z)
        quat = self.data.qpos[3:7]
        # Formule correcte pour extraire l'angle yaw d'un quaternion
        angle = 2 * np.arctan2(quat[3], quat[0])
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 4 coins de la bounding box dans le repère LOCAL du robot
        # X = avant/arrière (longueur), Y = gauche/droite (largeur)
        half_length = self.robot_length / 2  # 0.3m = 6 cellules
        half_width = self.robot_width / 2    # 0.2m = 4 cellules
        
        # Coins dans le repère local (X avant, Y gauche)
        corners_local = [
            ( half_length,  half_width),  # avant-gauche
            ( half_length, -half_width),  # avant-droite
            (-half_length,  half_width),  # arrière-gauche
            (-half_length, -half_width),  # arrière-droite
        ]
        
        corners_grid = []
        
        # Position du robot dans la grille (vision fixe)
        robot_grid_row = 40  # 2m derrière = 40 cellules (fixe en X)
        
        # Position Y du robot dans la grille
        robot_grid_col = int((robot_y + self.corridor_width/2) / self.cell_size)
        
        for local_x, local_y in corners_local:
            # Rotation 2D autour du centre du robot
            # world_offset_x = déplacement en X (direction du couloir = rows)
            # world_offset_y = déplacement en Y (perpendiculaire = cols)
            world_offset_x = cos_a * local_x - sin_a * local_y
            world_offset_y = sin_a * local_x + cos_a * local_y
            
            # Convertir en cellules de grille
            # row augmente vers l'avant (X positif)
            # col augmente vers la gauche (Y positif)
            delta_row = round(world_offset_x / self.cell_size)
            delta_col = round(world_offset_y / self.cell_size)
            
            grid_row = robot_grid_row + delta_row
            grid_col = robot_grid_col + delta_col
            
            corners_grid.extend([grid_row, grid_col])
        
        return np.array(corners_grid, dtype=np.float32)
    
    def _update_position_history(self):
        """Mettre à jour l'historique des 4 coins de la bounding box."""
        # Calculer les 4 coins actuels dans le repère grille
        current_corners = self._get_robot_bbox_corners(self.data.qpos[0], self.data.qpos[1])
        
        # Stocker les 4 coins (8 valeurs : 4 coins × 2 coords)
        self.position_history.append(current_corners.copy())
        
        # Garder seulement les N dernières positions
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
    
    def _get_position_history_obs(self):
        """Obtenir l'historique des 4 coins en coordonnées RELATIVES."""
        # Position actuelle des 4 coins
        current_corners = self._get_robot_bbox_corners(self.data.qpos[0], self.data.qpos[1])
        
        history_obs = []
        
        for i in range(self.history_length):
            if i < len(self.position_history):
                # Coins passés
                past_corners = self.position_history[i]
                
                # Convertir en coordonnées RELATIVES par rapport à la position actuelle
                # (différence entre position passée et position actuelle)
                relative_corners = past_corners - current_corners
                history_obs.extend(relative_corners)
            else:
                # Remplir avec position actuelle (différence = 0) si pas assez d'historique
                history_obs.extend([0.0] * 8)  # 8 valeurs (4 coins × 2 coords)
        
        return np.array(history_obs, dtype=np.float32)
    
    def _get_grid_obs(self, robot_x, robot_y):
        """Grille unique 120×80 avec environnement ET robot intégré."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Position du robot dans la grille monde
        robot_row_world = int(robot_x / self.cell_size)
        robot_col_world = int((robot_y + self.corridor_width/2) / self.cell_size)
        
        # Limites de la vision (4m devant, 2m derrière)
        vision_start_row = robot_row_world - 40  # 2m derrière = 40 cellules
        vision_end_row = robot_row_world + 80    # 4m devant = 80 cellules
        
        # Vision Y FIXE : toujours toute la largeur du couloir (3m = 60 cellules)
        # Centrée sur y=0 (centre du couloir en coordonnées monde)
        corridor_center_world_col = int((0 + self.corridor_width/2) / self.cell_size)  # y=0 → col=30
        vision_start_col = corridor_center_world_col - 30  # 30 - 30 = 0 (1.5m à gauche)
        vision_end_col = corridor_center_world_col + 30    # 30 + 30 = 60 (1.5m à droite)
        
        # Remplir la grille avec l'environnement
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Position dans le monde
                world_row = vision_start_row + i
                world_col = vision_start_col + j
                
                # Vérifier si dans les limites du couloir
                world_y = (world_col * self.cell_size) - self.corridor_width/2
                
                if world_col < 0 or world_y < -self.corridor_width/2 or world_y > self.corridor_width/2:
                    # En dehors du couloir = trou
                    grid[i, j] = 1.0
                else:
                    # Chercher dans la carte des cellules
                    cell_type = self.cell_map.get((world_row, world_col), 2)  # Défaut trou
                    
                    # Normaliser: 0=sol, 0.5=rampe, 1=trou
                    if cell_type == 0:
                        grid[i, j] = 0.0  # Sol
                    elif cell_type == 1:
                        grid[i, j] = 0.5  # Rampe
                    else:  # cell_type == 2
                        grid[i, j] = 1.0  # Trou
        
        # PAS de robot dans la grille - les 4 coins sont donnés séparément dans l'observation
        
        return grid
    
    def _compute_reward(self):
        """Récompense shaped pour apprentissage efficace."""
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        z = self.data.qpos[2]
        vx = self.data.qvel[0]
        vy = self.data.qvel[1]
        
        terminated = False
        info = {}
        
        # Succès
        if x >= self.corridor_length:
            info['reason'] = 'success'
            return 500.0, True, info
        
        # Échec: tombé
        if z < 0.15:
            info['reason'] = 'fell'
            return -100.0, True, info
        
        # Échec: retourné
        quat = self.data.qpos[3:7]
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0:
            info['reason'] = 'flipped'
            return -50.0, True, info
        
        # Échec: sorti du couloir (pénalité plus forte)
        if abs(y) > self.corridor_width / 2:
            info['reason'] = 'out_of_bounds'
            return -50.0, True, info
        
        # Récompense de progression
        delta_x = x - self.prev_x
        self.prev_x = x
        

        # Récompense progression TRÈS généreuse
        reward = delta_x * 20.0
        
        # BONUS DENSE pour encourager apprentissage
        
        # Gros bonus pour avancer
        if vx > 0.05:
            reward += 2.0
        
        # Bonus distance parcourue (récompense cumulative)
        if x > 5.0:
            reward += 1.0
        if x > 10.0:
            reward += 2.0
        
        # Malus modéré pour instabilité
        if abs(vy) > 1.0:
            reward -= 0.5  # Réduit de 0.5 à 0.3
        
        # Petit malus temps (encourage vitesse)
        reward -= 0.01
        
        info['reason'] = None
        return reward, terminated, info
    
    def _get_info(self):
        return {
            'x': float(self.data.qpos[0]),
            'y': float(self.data.qpos[1]),
            'z': float(self.data.qpos[2]),
            'step': self.step_count
        }
    
    def _build_cell_map(self):
        """Construire carte des cellules depuis géométries MuJoCo - GRILLE 0.05m."""
        cell_map = {}
        n_rows = int(self.corridor_length / self.cell_size) + 100  # Grille 0.05m = 2000+ lignes
        n_cols = int(self.corridor_width / self.cell_size)         # 60 colonnes
        
        # Tout est trou par défaut
        for r in range(n_rows):
            for c in range(n_cols):
                cell_map[(r, c)] = 2  # Trou par défaut
        
        # Parcourir géométries MuJoCo
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            
            pos = self.model.geom_pos[geom_id]
            size = self.model.geom_size[geom_id]
            
            # Déterminer le type de cellule
            name_lower = name.lower()
            if 'ramp' in name_lower:
                cell_type = 1  # Rampe
            elif 'flat' in name_lower or 'floor' in name_lower or 'cell' in name_lower:
                cell_type = 0  # Sol
            else:
                continue  # Ignorer autres géométries
            
            # Marquer toutes les cellules couvertes par cette géométrie
            min_x = pos[0] - size[0]
            max_x = pos[0] + size[0]
            min_y = pos[1] - size[1]
            max_y = pos[1] + size[1]
            
            for r in range(n_rows):
                cx = (r + 0.5) * self.cell_size  # Centre de la cellule en X
                if min_x <= cx <= max_x:
                    for c in range(n_cols):
                        cy = (c + 0.5) * self.cell_size - self.corridor_width/2  # Centre en Y
                        if min_y <= cy <= max_y:
                            cell_map[(r, c)] = cell_type
        
        return cell_map

    def _build_model(self, robot_xml, corridor_xml):
        """Combiner robot et corridor."""
        robot_tree = ET.parse(robot_xml)
        corridor_tree = ET.parse(corridor_xml)
        
        robot_root = robot_tree.getroot()
        corridor_root = corridor_tree.getroot()
        
        root = ET.Element('mujoco')
        root.set('model', 'robot_in_corridor')
        
        # Compiler
        for child in robot_root:
            if child.tag == 'compiler':
                root.append(child)
                break
        
        # Options
        option = ET.SubElement(root, 'option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        
        # Size
        size = ET.SubElement(root, 'size')
        size.set('njmax', '4000')
        size.set('nconmax', '1000')
        
        # Default
        for child in robot_root:
            if child.tag == 'default':
                root.append(child)
                break
        
        # Visual
        for child in robot_root:
            if child.tag == 'visual':
                root.append(child)
                break
        
        # Assets
        asset = ET.SubElement(root, 'asset')
        added = set()
        for src in [robot_root, corridor_root]:
            asset_elem = src.find('asset')
            if asset_elem is not None:
                for mat in asset_elem:
                    name = mat.get('name', '')
                    if name not in added:
                        asset.append(mat)
                        added.add(name)
        
        # Worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Corridor
        corridor_wb = corridor_root.find('worldbody')
        if corridor_wb is not None:
            for elem in corridor_wb:
                worldbody.append(elem)
        
        # Robot (position sera overridée dans reset)
        robot_wb = robot_root.find('worldbody')
        if robot_wb is not None:
            for body in robot_wb:
                if body.get('name') == 'robot':
                    body.set('pos', '2 0 0.45')
                    worldbody.append(body)
        
        # Actuateurs
        robot_act = robot_root.find('actuator')
        if robot_act is not None:
            root.append(robot_act)
        
        xml_str = ET.tostring(root, encoding='unicode')
        return mujoco.MjModel.from_xml_string(xml_str)
    
    def render(self):
        pass
    
    def set_max_steps(self, new_max_steps):
        """Ajuster dynamiquement la durée max des épisodes."""
        self.max_steps = new_max_steps
    
    def close(self):
        pass