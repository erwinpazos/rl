"""
Environnement Gymnasium pour robot 4 roues dans corridor avec obstacles.
Optimisé pour apprentissage PPO avec spawn aléatoire.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


class CorridorEnv(gym.Env):
    """
    Robot 4 roues naviguant un corridor avec trous et rampes.
    
    Observation (198 valeurs):
        - Position robot (x, y, z): 3
        - Vitesse robot (vx, vy, vz): 3
        - Position 4 roues dans grille (row, col): 8
        - Grille sol 16x12 devant robot: 192 (0=sol, 1=rampe, 2=trou)
    
    Action (4 valeurs):
        - Couple roue FL, FR, RL, RR
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, corridor_xml="corridor_3x100.xml", max_steps=1000):
        super().__init__()
        
        # Charger modèle MuJoCo
        self.model = self._build_model("four_wheels_robot.xml", corridor_xml)
        self.data = mujoco.MjData(self.model)
        
        # Paramètres
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_size = 0.0625  # 4× plus petit (0.25 → 0.0625) = 6.25cm
        
        # Grille: 32 lignes (2m devant) × 64 colonnes (4m largeur = tout le couloir + zones)
        self.grid_rows = 32  # 2m ÷ 0.0625m = 32 lignes
        self.grid_cols = 64  # 4m ÷ 0.0625m = 64 colonnes
        
        # Distance roues depuis centre (correspond au XML)
        self.wheel_offset_x = 0.35  # avant/arrière (±0.35 dans XML)
        self.wheel_offset_y = 0.30  # gauche/droite (±0.30 dans XML)
        
        # Construire carte
        self.cell_map = self._build_cell_map()
        
        # Espaces - DEUX COUCHES
        obs_size = 6 + 8 + 2 * (self.grid_rows * self.grid_cols)  # 6 + 8 + 2×2048 = 4110
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        
        # État
        self.step_count = 0
        self.prev_x = 0.0
        
        # Détection blocage (basé sur position, pas delta)
        self.stuck_check_interval = 50   # Vérifier tous les 50 steps
        self.stuck_min_advance = 0.3     # Doit avancer d'au moins 30cm en 50 steps
        self.stuck_x_checkpoint = 0.0    # Position X au dernier checkpoint
        self.stuck_counter = 0           # Nombre de fois bloqué consécutives
        self.stuck_max_count = 2         # 2 × 50 = 100 steps bloqué = terminaison
        
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
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Appliquer couple roues
        action = np.clip(action, -1.0, 1.0) * 20.0
        self.data.ctrl[:] = action
        
        # Simuler (4 substeps)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Récompense et terminaison
        reward, terminated, info = self._compute_reward()
        truncated = self.step_count >= self.max_steps
        
        info.update(self._get_info())
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation complète avec DEUX COUCHES : environnement + robot."""
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        
        # Position des 4 roues dans la grille
        wheel_positions = self._get_wheel_positions(pos[0], pos[1])
        
        # COUCHE 1: Grille environnement (sol/trou/rampe)
        env_grid = self._get_grid_obs_environment(pos[0], pos[1])
        
        # COUCHE 2: Grille robot (position du corps)
        robot_grid = self._get_grid_obs_robot(pos[0], pos[1])
        
        return np.concatenate([
            pos, 
            vel, 
            wheel_positions,
            env_grid.flatten(),    # Couche 1: 32×64 = 2048 valeurs
            robot_grid.flatten()   # Couche 2: 32×64 = 2048 valeurs
        ]).astype(np.float32)  # Total: 6 + 8 + 2048 + 2048 = 4110 valeurs
    
    def _get_wheel_positions(self, robot_x, robot_y):
        """Position (row, col) de chaque roue dans la grille."""
        # Récupérer orientation robot
        quat = self.data.qpos[3:7]
        # Angle autour de Z (approximation pour rotation 2D)
        angle = 2 * np.arctan2(quat[3], quat[0])
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        wheels = []
        offsets = [
            (self.wheel_offset_x, self.wheel_offset_y),   # FL
            (self.wheel_offset_x, -self.wheel_offset_y),  # FR
            (-self.wheel_offset_x, self.wheel_offset_y),  # RL
            (-self.wheel_offset_x, -self.wheel_offset_y), # RR
        ]
        
        for offset_x, offset_y in offsets:
            # Position mondiale roue
            wx = robot_x + cos_a * offset_x - sin_a * offset_y
            wy = robot_y + sin_a * offset_x + cos_a * offset_y
            
            # Convertir en indices grille
            row = int(wx / self.cell_size)
            col = int((wy + self.corridor_width/2) / self.cell_size)
            
            wheels.extend([row, col])
        
        return np.array(wheels, dtype=np.float32)
    
    def _get_grid_obs(self, robot_x, robot_y):
        """Grille 16×16 AUTOUR du robot (pas seulement devant)."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        robot_row = int(robot_x / self.cell_size)
        robot_col = int((robot_y + self.corridor_width/2) / self.cell_size)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Vision centrée: 16 lignes derrière, 16 lignes devant
                row = robot_row - 16 + i  # De -16 à +15 par rapport au robot
                col = robot_col - 32 + j  # Centré sur robot (±32 colonnes = 4m largeur)
                
                # Si en dehors du couloir (col < 0 ou col >= 12), c'est un trou
                if col < 0 or col >= 12:
                    grid[i, j] = 2  # Trou autour du couloir
                else:
                    grid[i, j] = self.cell_map.get((row, col), 2)  # Défaut trou
        
        return grid
    
    def _get_grid_obs_environment(self, robot_x, robot_y):
        """COUCHE 1: Grille environnement NORMALISÉE (0-1)."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        robot_row = int(robot_x / self.cell_size)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Vision X: centrée sur robot (1m derrière, 1m devant)
                row = robot_row - 16 + i  # De -16 à +15 par rapport au robot
                
                # Vision Y: FIXE de -2m à +2m (couvre tout le couloir)
                world_y = -2.0 + (j * self.cell_size)  # De -2m à +2m selon j
                
                # Convertir vers cell_map (grille 0.0625m)
                if row >= 0 and row < len(self.cell_map) // 48:  # Vérifier limites
                    world_col = int((world_y + self.corridor_width/2) / self.cell_size)
                    
                    # Si en dehors du couloir (48 colonnes), c'est un trou
                    if world_col < 0 or world_col >= 48:
                        cell_type = 2  # Trou autour du couloir
                    else:
                        cell_type = self.cell_map.get((row, world_col), 2)  # Défaut trou
                else:
                    cell_type = 2  # Trou si hors limites
                
                # NORMALISER: 0=sol, 0.5=rampe, 1=trou
                if cell_type == 0:
                    grid[i, j] = 0.0  # Sol = 0
                elif cell_type == 1:
                    grid[i, j] = 0.5  # Rampe = 0.5
                else:  # cell_type == 2
                    grid[i, j] = 1.0  # Trou = 1
        
        return grid
    
    def _get_grid_obs_robot(self, robot_x, robot_y):
        """COUCHE 2: Grille robot NORMALISÉE (0-1)."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)  # Vide = 0
        
        # Récupérer orientation robot
        quat = self.data.qpos[3:7]
        angle = 2 * np.arctan2(quat[3], quat[0])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Dimensions robot
        robot_length = 0.7  # wheelbase
        robot_width = 0.6   # track width
        
        # Position du robot dans la grille
        center_row = 16  # Robot au centre des 32 lignes
        robot_y_in_grid = int((robot_y + 2.0) / self.cell_size)  # Position Y dans la grille
        center_col = robot_y_in_grid
        
        # Remplissage ULTRA DENSE du corps du robot
        step = self.cell_size / 4  # Sous-échantillonnage dense
        
        x_range = np.arange(-robot_length/2, robot_length/2 + step, step)
        y_range = np.arange(-robot_width/2, robot_width/2 + step, step)
        
        for local_x in x_range:
            for local_y in y_range:
                # Appliquer la rotation du robot
                rotated_x = cos_a * local_x - sin_a * local_y
                rotated_y = sin_a * local_x + cos_a * local_y
                
                # Convertir en position grille
                grid_row = center_row + int(rotated_x / self.cell_size)
                grid_col = center_col + int(rotated_y / self.cell_size)
                
                if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
                    grid[grid_row, grid_col] = 1.0  # Robot présent = 1
        
        return grid  # Vide = 0, Robot = 1
    
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
        
        # DÉTECTION BLOCAGE: vérifier progression sur fenêtre de temps
        if self.step_count % self.stuck_check_interval == 0 and self.step_count > 0:
            advance_since_checkpoint = x - self.stuck_x_checkpoint
            
            if advance_since_checkpoint < self.stuck_min_advance:
                # Pas assez avancé depuis le dernier checkpoint
                self.stuck_counter += 1
            else:
                # Bonne progression, reset
                self.stuck_counter = 0
            
            # Mettre à jour checkpoint
            self.stuck_x_checkpoint = x
        
        # Échec: bloqué trop longtemps (2 × 50 = 100 steps sans avancer de 30cm)
        if self.stuck_counter >= self.stuck_max_count:
            info['reason'] = 'stuck'
            return -50.0, True, info
        
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
        """Construire carte des cellules depuis géométries MuJoCo - GRILLE 0.0625m."""
        cell_map = {}
        n_rows = int(self.corridor_length / self.cell_size) + 10  # Grille 0.0625m
        n_cols = int(self.corridor_width / self.cell_size)  # 48 colonnes
        
        # Tout est trou par défaut (comme dans l'image)
        for r in range(n_rows):
            for c in range(n_cols):
                cell_map[(r, c)] = 2  # Trou par défaut
        
        # Parcourir géométries
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            
            pos = self.model.geom_pos[geom_id]
            size = self.model.geom_size[geom_id]
            
            # Type
            name_lower = name.lower()
            if 'ramp' in name_lower:
                cell_type = 1
            elif 'flat' in name_lower or 'floor' in name_lower or 'cell' in name_lower:
                cell_type = 0
            else:
                continue
            
            # Marquer cellules - GRILLE 0.25m
            min_x = pos[0] - size[0]
            max_x = pos[0] + size[0]
            min_y = pos[1] - size[1]
            max_y = pos[1] + size[1]
            
            for r in range(n_rows):
                cx = (r + 0.5) * self.cell_size  # Grille 0.0625m
                if min_x <= cx <= max_x:
                    for c in range(n_cols):
                        cy = (c + 0.5) * self.cell_size - self.corridor_width/2  # Grille 0.0625m
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