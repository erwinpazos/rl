"""
Environnement Gymnasium SIMPLIFIÉ pour robot 4 roues dans corridor.
Version nettoyée avec grille 0.1m et CNN unique.
Génération aléatoire de corridors à chaque reset.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET
from corridor_generator import generate_corridor_grid, grid_to_cell_map, grid_to_xml_string


class CorridorEnv(gym.Env):
    """
    Robot 4 roues naviguant un corridor avec trous et bumps.
    
    Observation:
        - Position robot (x, y, z): 3
        - Vitesse robot (vx, vy, vz): 3  
        - Bounding box coins (4 coins × 2 coords): 8
        - Grille environnement 60×30: 1800 (0=sol, 0.5=bump, 1=trou)
    
    Action (4 valeurs):
        - Couple roue FL, FR, RL, RR
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, max_steps=3000, corridor_xml=None):
        super().__init__()
        
        self.robot_xml = "four_wheels_robot.xml"
        self.corridor_xml = corridor_xml
        self.use_random_corridor = corridor_xml is None
        
        if self.use_random_corridor:
            # Générer premier corridor aléatoire
            self.current_grid = generate_corridor_grid()
            self.model = self._build_model_from_grid(self.current_grid)
        else:
            # Utiliser corridor XML fixe
            self.model = self._build_model_from_xml(corridor_xml)
        
        self.data = mujoco.MjData(self.model)
        
        # Paramètres SIMPLIFIÉS
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_size = 0.1  # 10cm par cellule (plus gros = plus facile à apprendre)
        
        # Grille vision: 5.2m devant + 0.8m derrière = 6m × 3m largeur
        self.vision_behind = 0.8  # 0.8m derrière (8 cellules) - assez pour voir coins arrière même inclinés
        self.vision_front = 5.2   # 5.2m devant (52 cellules)
        self.vision_length = self.vision_behind + self.vision_front  # 6m total
        self.vision_width = 3.0   # 3m largeur (exactement la largeur du couloir)
        self.grid_rows = int(self.vision_length / self.cell_size)  # 60 lignes
        self.grid_cols = int(self.vision_width / self.cell_size)   # 30 colonnes
        self.robot_row_in_grid = round(self.vision_behind / self.cell_size)  # 8 (0.8m derrière)
        
        # Dimensions robot (bounding box) - ENGLOBENT TOUT LE ROBOT + ROUES
        self.robot_length = 1.10  # 11 cellules (empattement 0.7m + diamètre roues 0.4m = 1.1m)
        self.robot_width = 0.70   # 7 cellules (voie 0.6m + dépassement roues ~0.1m = 0.7m)
        
        # Historique des positions pour anticipation (AVANT les espaces)
        self.history_interval = 10  # Sauvegarder position tous les 10 steps (plus fréquent)
        self.history_length = 8     # Garder les 8 dernières positions (plus d'historique)
        self.position_history = []  # Buffer des positions + vitesses
        
        # Période de stabilisation
        self.stabilization_steps = 20  # Pas d'actions pendant 20 steps
        
        # Construire carte (sera regénérée à chaque reset si random_corridor=True)
        self.cell_map = self._build_cell_map()
        
        # Espaces - UN SEUL CNN + historique étendu
        # Historique: 8 positions × (8 coords + 3 vitesses) = 8 × 11 = 88 valeurs
        history_size = self.history_length * 11  # positions (8) + vitesses (3) par frame
        obs_size = 6 + 8 + history_size + (self.grid_rows * self.grid_cols)  # 6 + 8 + 88 + 1800 = 1902
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
        
        if self.use_random_corridor:
            # Générer un nouveau corridor aléatoire à chaque reset
            self.current_grid = generate_corridor_grid()
            self.model = self._build_model_from_grid(self.current_grid)
            self.data = mujoco.MjData(self.model)
            self.cell_map = grid_to_cell_map(self.current_grid, self.cell_size)
            
            # Mettre à jour robot_body_id pour le nouveau modèle
            self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        else:
            # Corridor fixe, juste reset les données
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
            position_history,       # 88 valeurs (8 frames × 11 valeurs: 8 coins + 3 vitesses)
            grid.flatten()          # 1800 valeurs (60×30)
        ]).astype(np.float32)       # Total: 6 + 8 + 88 + 1800 = 1902 valeurs
    
    def _get_robot_bbox_corners(self, robot_x, robot_y):
        """Position des 4 coins de la bounding box dans le repère de la grille EGO-CENTRIQUE.
        
        Les coins sont toujours dans le repère LOCAL du robot (pas de rotation nécessaire
        car la grille tourne avec le robot).
        """
        # Dans le repère ego-centrique, le robot est toujours orienté "vers le haut"
        # donc les coins sont fixes dans la grille
        
        # 4 coins de la bounding box dans le repère LOCAL du robot
        # X = avant/arrière (longueur), Y = gauche/droite (largeur)
        half_length = self.robot_length / 2  # 0.55m = 5.5 cellules
        half_width = self.robot_width / 2    # 0.35m = 3.5 cellules
        
        # Position du robot dans la grille (toujours au centre)
        robot_grid_row = self.robot_row_in_grid  # 8 (0.8m derrière, fixe en X)
        robot_grid_col = self.grid_cols // 2      # 15 (centre de la grille, fixe en Y)
        
        # Coins dans le repère LOCAL (pas de rotation car grille ego-centrique)
        # X local = lignes de la grille, Y local = colonnes de la grille
        corners_grid = []
        
        # Avant-gauche
        corners_grid.extend([
            robot_grid_row + round(half_length / self.cell_size),  # row
            robot_grid_col + round(half_width / self.cell_size)    # col
        ])
        
        # Avant-droite
        corners_grid.extend([
            robot_grid_row + round(half_length / self.cell_size),  # row
            robot_grid_col - round(half_width / self.cell_size)    # col
        ])
        
        # Arrière-gauche
        corners_grid.extend([
            robot_grid_row - round(half_length / self.cell_size),  # row
            robot_grid_col + round(half_width / self.cell_size)    # col
        ])
        
        # Arrière-droite
        corners_grid.extend([
            robot_grid_row - round(half_length / self.cell_size),  # row
            robot_grid_col - round(half_width / self.cell_size)    # col
        ])
        
        return np.array(corners_grid, dtype=np.float32)
    
    def _update_position_history(self):
        """Mettre à jour l'historique des 4 coins + vitesses."""
        # Calculer les 4 coins actuels dans le repère grille
        current_corners = self._get_robot_bbox_corners(self.data.qpos[0], self.data.qpos[1])
        
        # Vitesses actuelles
        current_velocities = self.data.qvel[:3].copy()  # vx, vy, vz
        
        # Stocker coins + vitesses (8 + 3 = 11 valeurs)
        frame_data = np.concatenate([current_corners, current_velocities])
        self.position_history.append(frame_data)
        
        # Garder seulement les N dernières positions
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
    
    def _get_position_history_obs(self):
        """Obtenir l'historique des 4 coins + vitesses en coordonnées RELATIVES."""
        # État actuel
        current_corners = self._get_robot_bbox_corners(self.data.qpos[0], self.data.qpos[1])
        current_velocities = self.data.qvel[:3]
        
        history_obs = []
        
        for i in range(self.history_length):
            if i < len(self.position_history):
                # Frame passée (8 coins + 3 vitesses)
                past_frame = self.position_history[i]
                past_corners = past_frame[:8]  # 8 premiers = coins
                past_velocities = past_frame[8:]  # 3 derniers = vitesses
                
                # Coins relatifs (différence par rapport à position actuelle)
                relative_corners = past_corners - current_corners
                
                # Vitesses relatives (différence par rapport à vitesse actuelle)
                relative_velocities = past_velocities - current_velocities
                
                # Combiner coins + vitesses (8 + 3 = 11 valeurs)
                history_obs.extend(relative_corners)
                history_obs.extend(relative_velocities)
            else:
                # Remplir avec zéros si pas assez d'historique
                history_obs.extend([0.0] * 11)  # 8 coins + 3 vitesses
        
        return np.array(history_obs, dtype=np.float32)
    
    def _get_grid_obs(self, robot_x, robot_y):
        """Grille unique 60×30 avec environnement, CENTRÉE ET ORIENTÉE selon le robot."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        # Récupérer l'angle du robot
        quat = self.data.qpos[3:7]
        robot_angle = 2 * np.arctan2(quat[3], quat[0])
        cos_a = np.cos(robot_angle)  # Rotation directe (pas inverse)
        sin_a = np.sin(robot_angle)
        
        # Position du robot dans la grille monde
        robot_row_world = int(robot_x / self.cell_size)
        robot_col_world = int((robot_y + self.corridor_width/2) / self.cell_size)
        
        # Pour chaque cellule de la grille de vision
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Position relative dans le repère de la grille (robot au centre)
                # i=0 → 0.8m derrière, i=8 → robot, i=60 → 5.2m devant
                relative_x = (i - self.robot_row_in_grid) * self.cell_size  # Distance devant/derrière
                relative_y = (j - self.grid_cols // 2) * self.cell_size     # Distance gauche/droite
                
                # Rotation pour obtenir position dans le repère monde
                world_offset_x = cos_a * relative_x - sin_a * relative_y
                world_offset_y = sin_a * relative_x + cos_a * relative_y
                
                # Position absolue dans le monde
                world_x = robot_x + world_offset_x
                world_y = robot_y + world_offset_y
                
                # Convertir en indices de grille monde
                world_row = int(world_x / self.cell_size)
                world_col = int((world_y + self.corridor_width/2) / self.cell_size)
                
                # Vérifier si en dehors du couloir
                if world_y < -self.corridor_width/2 or world_y > self.corridor_width/2:
                    # En dehors du couloir = valeur distincte -1.0
                    grid[i, j] = -1.0
                else:
                    # Chercher dans la carte des cellules
                    cell_type = self.cell_map.get((world_row, world_col), 2)  # Défaut trou
                    
                    # Normaliser: 0=sol, 0.5=bump, 1=trou
                    if cell_type == 0:
                        grid[i, j] = 0.0  # Sol
                    elif cell_type == 1:
                        grid[i, j] = 0.5  # Bump
                    else:  # cell_type == 2
                        grid[i, j] = 1.0  # Trou
        
        return grid
    
    def _compute_reward(self):
        """Récompense SIMPLE: avancer = bien, tomber = mal."""
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        z = self.data.qpos[2]
        
        terminated = False
        info = {}
        
        # Succès: fin du corridor
        if x >= self.corridor_length:
            info['reason'] = 'success'
            return 100.0, True, info
        
        # Échec: tombé dans un trou
        if z < 0.15:
            info['reason'] = 'fell'
            return -10.0, True, info
        
        # Échec: retourné
        quat = self.data.qpos[3:7]
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0:
            info['reason'] = 'flipped'
            return -10.0, True, info
        
        # Pas de vérification out_of_bounds : le robot tombera naturellement dans le vide
        
        # Récompense SIMPLE: juste la progression en X
        delta_x = x - self.prev_x
        self.prev_x = x
        reward = delta_x * 10.0  # +10 par mètre avancé
        
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
        """Construire carte initiale."""
        if self.use_random_corridor:
            return grid_to_cell_map(self.current_grid, self.cell_size)
        else:
            return self._build_cell_map_from_xml()

    def _build_model_from_grid(self, grid):
        """Construire modèle MuJoCo avec robot + corridor généré."""
        # Générer XML du corridor
        corridor_xml_str = grid_to_xml_string(grid)
        corridor_root = ET.fromstring(corridor_xml_str)
        
        # Charger robot
        robot_tree = ET.parse(self.robot_xml)
        robot_root = robot_tree.getroot()
        
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
        
        # Assets (combiner robot + corridor)
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
        
        # Corridor (géométries générées)
        corridor_wb = corridor_root.find('worldbody')
        if corridor_wb is not None:
            for elem in corridor_wb:
                worldbody.append(elem)
        
        # Robot
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
    
    def _build_model_from_xml(self, corridor_xml):
        """Construire modèle MuJoCo avec robot + corridor XML fixe."""
        robot_tree = ET.parse(self.robot_xml)
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
        
        # Assets (combiner robot + corridor)
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
        
        # Robot
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
    
    def _build_cell_map_from_xml(self):
        """Construire carte des cellules depuis géométries MuJoCo XML."""
        cell_map = {}
        n_rows = int(self.corridor_length / self.cell_size) + 100
        n_cols = int(self.corridor_width / self.cell_size)
        
        # Tout est trou par défaut
        for r in range(n_rows):
            for c in range(n_cols):
                cell_map[(r, c)] = 2
        
        # Parcourir géométries MuJoCo
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            
            pos = self.model.geom_pos[geom_id]
            size = self.model.geom_size[geom_id]
            
            # Déterminer le type de cellule
            name_lower = name.lower()
            if 'bump' in name_lower:
                cell_type = 1  # Bump
            elif 'flat' in name_lower or 'floor' in name_lower or 'cell' in name_lower:
                cell_type = 0  # Sol
            else:
                continue
            
            # Marquer toutes les cellules couvertes par cette géométrie
            min_x = pos[0] - size[0]
            max_x = pos[0] + size[0]
            min_y = pos[1] - size[1]
            max_y = pos[1] + size[1]
            
            for r in range(n_rows):
                cx = (r + 0.5) * self.cell_size
                if min_x <= cx <= max_x:
                    for c in range(n_cols):
                        cy = (c + 0.5) * self.cell_size - self.corridor_width/2
                        if min_y <= cy <= max_y:
                            cell_map[(r, c)] = cell_type
        
        return cell_map
    
    def render(self):
        pass
    
    def set_max_steps(self, new_max_steps):
        """Ajuster dynamiquement la durée max des épisodes."""
        self.max_steps = new_max_steps
    
    def close(self):
        pass