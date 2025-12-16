"""
Environnement Gymnasium pour robot 4 roues dans corridor avec obstacles.
Simple et efficace.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


class CorridorEnv(gym.Env):
    """
    Robot 4 roues naviguant un corridor avec trous et rampes.
    
    Observation (30 valeurs):
        - Position robot (x, y, z): 3
        - Vitesse robot (vx, vy, vz): 3  
        - Grille sol 4x6 devant robot: 24 (0=sol, 1=rampe, 2=trou)
    
    Action (4 valeurs):
        - Couple roue avant-gauche, avant-droite, arrière-gauche, arrière-droite
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, corridor_xml="corridor_3x100.xml", max_steps=2000):
        super().__init__()
        
        # Charger modèle MuJoCo
        self.model = self._build_model("four_wheels_robot.xml", corridor_xml)
        self.data = mujoco.MjData(self.model)
        
        # Paramètres
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_size = 0.25  # Taille cellule grille (25cm)
        
        # Grille d'observation: 16 lignes devant (4m), 12 colonnes (3m)
        self.grid_rows = 16  # 16 × 0.25m = 4m devant
        self.grid_cols = 12  # 12 × 0.25m = 3m largeur
        
        # Construire carte des cellules
        self.cell_map = self._build_cell_map()
        
        # Espaces
        obs_size = 6 + (self.grid_rows * self.grid_cols)  # 6 + 24 = 30
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        
        # État
        self.step_count = 0
        self.prev_x = 0.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.prev_x = self.data.qpos[0]
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Appliquer action (couple roues)
        action = np.clip(action, -1.0, 1.0) * 20.0
        self.data.ctrl[:] = action
        
        # Simuler
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Calculer récompense et terminaison
        reward, terminated, info = self._compute_reward()
        truncated = self.step_count >= self.max_steps
        
        info.update(self._get_info())
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation: état robot + grille devant."""
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        grid = self._get_grid_obs(pos[0], pos[1])
        return np.concatenate([pos, vel, grid.flatten()]).astype(np.float32)
    
    def _get_grid_obs(self, robot_x, robot_y):
        """Grille 4x6 des cellules devant le robot."""
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        robot_row = int(robot_x / self.cell_size)
        robot_col = int((robot_y + self.corridor_width/2) / self.cell_size)
        
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                row = robot_row + i + 1  # Devant le robot
                col = j
                grid[i, j] = self.cell_map.get((row, col), 2)  # 2=trou par défaut
        
        return grid
    
    def _compute_reward(self):
        """Récompense simple: avancer = bien, tomber = mal."""
        x = self.data.qpos[0]
        z = self.data.qpos[2]
        
        terminated = False
        info = {}
        
        # Succès: fin du corridor
        if x >= self.corridor_length:
            info['reason'] = 'success'
            return 500.0, True, info
        
        # Échec: tombé dans un trou
        if z < 0.15:
            info['reason'] = 'fell'
            return -100.0, True, info
        
        # Échec: retourné (upside down)
        quat = self.data.qpos[3:7]
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0:
            info['reason'] = 'flipped'
            return -50.0, True, info
        
        # Récompense de progression
        delta_x = x - self.prev_x
        self.prev_x = x
        
        reward = delta_x * 10.0  # Avancer = bien
        
        # Petit malus temps pour encourager vitesse
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
        """Construire carte des cellules depuis géométries MuJoCo."""
        cell_map = {}
        n_rows = int(self.corridor_length / self.cell_size) + 10
        n_cols = int(self.corridor_width / self.cell_size)
        
        # Tout est trou par défaut
        for r in range(n_rows):
            for c in range(n_cols):
                cell_map[(r, c)] = 2
        
        # Parcourir géométries
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name:
                continue
            
            pos = self.model.geom_pos[geom_id]
            size = self.model.geom_size[geom_id]
            
            # Déterminer type
            name_lower = name.lower()
            if 'ramp' in name_lower:
                cell_type = 1
            elif 'flat' in name_lower or 'floor' in name_lower or 'cell' in name_lower:
                cell_type = 0
            else:
                continue
            
            # Marquer cellules couvertes
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

    def _build_model(self, robot_xml, corridor_xml):
        """Combiner robot et corridor en un seul modèle."""
        robot_tree = ET.parse(robot_xml)
        corridor_tree = ET.parse(corridor_xml)
        
        robot_root = robot_tree.getroot()
        corridor_root = corridor_tree.getroot()
        
        # Créer nouveau XML
        root = ET.Element('mujoco')
        root.set('model', 'robot_in_corridor')
        
        # Compiler du robot
        for child in robot_root:
            if child.tag == 'compiler':
                root.append(child)
                break
        
        # Options physique
        option = ET.SubElement(root, 'option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        
        # Size
        size = ET.SubElement(root, 'size')
        size.set('njmax', '4000')
        size.set('nconmax', '1000')
        
        # Default du robot
        for child in robot_root:
            if child.tag == 'default':
                root.append(child)
                break
        
        # Visual du robot
        for child in robot_root:
            if child.tag == 'visual':
                root.append(child)
                break
        
        # Assets combinés
        asset = ET.SubElement(root, 'asset')
        added_names = set()
        
        for src in [robot_root, corridor_root]:
            asset_elem = src.find('asset')
            if asset_elem is not None:
                for mat in asset_elem:
                    name = mat.get('name', '')
                    if name not in added_names:
                        asset.append(mat)
                        added_names.add(name)
        
        # Worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Ajouter corridor
        corridor_wb = corridor_root.find('worldbody')
        if corridor_wb is not None:
            for elem in corridor_wb:
                worldbody.append(elem)
        
        # Ajouter robot avec position ajustée
        robot_wb = robot_root.find('worldbody')
        if robot_wb is not None:
            for body in robot_wb:
                if body.get('name') == 'robot':
                    body.set('pos', '1 0 0.45')
                    worldbody.append(body)
        
        # Actuateurs du robot
        robot_act = robot_root.find('actuator')
        if robot_act is not None:
            root.append(robot_act)
        
        # Créer modèle
        xml_str = ET.tostring(root, encoding='unicode')
        return mujoco.MjModel.from_xml_string(xml_str)
    
    def render(self):
        pass
    
    def close(self):
        pass
