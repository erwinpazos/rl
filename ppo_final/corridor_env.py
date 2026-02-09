"""
Environnement Gymnasium SIMPLIFIÉ pour robot 4 roues dans corridor.
Version avec contrôle par VOLANT (steering + speed) au lieu de 4 roues indépendantes.
Utilise corridor fixe.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


def clip_steering_angle(angle, max_angle=30.0):
    """Clip steering angle to within +/- max_angle degrees."""
    if angle > max_angle:
        return max_angle
    elif angle < -max_angle:
        return -max_angle
    return angle


def yaw_rate_to_wheel_speeds(yaw_rate_rps: float,
                             speed: float,
                             track_width: float = 0.4,
                             wheel_radius: float = 0.15,
                             max_wheel_rad_s: float = None):
    """
    Arcade-style control: command (v, yaw_rate) -> 4 wheel angular speeds (rad/s)
    For a 4-wheel skid-steer robot:
      - yaw_rate > 0 turns left
      - v > 0 goes forward

    Parameters
    ----------
    yaw_rate_rps : float
        Desired yaw rate in radians per second.
    speed : float
        Desired forward speed in meters per second.
    track_width : float
        Distance between left and right wheels (meters).
    wheel_radius : float
        Radius of the wheels (meters).
    max_wheel_rad_s : float | None
        Optional maximum wheel angular speed (rad/s) for saturation.

    Returns
    -------
    (w_fl, w_fr, w_rl, w_rr) in rad/s
        4-tuple of wheel angular speeds in radians per second: (front-left, front-right, rear-left, rear-right)
    """
    # Linear speed for each side (m/s)
    v_left  = speed - yaw_rate_rps * (track_width / 2.0)
    v_right = speed + yaw_rate_rps * (track_width / 2.0)

    # Convert to wheel angular speeds (rad/s)
    w_left  = v_left / wheel_radius
    w_right = v_right / wheel_radius

    # Optional saturation (keeps arcade feel without exploding commands)
    if max_wheel_rad_s is not None:
        w_left  = float(np.clip(w_left,  -max_wheel_rad_s, max_wheel_rad_s))
        w_right = float(np.clip(w_right, -max_wheel_rad_s, max_wheel_rad_s))

    # FL, FR, RL, RR
    return (w_left, w_right, w_left, w_right)


def steer_angle_to_wheel_speeds(steering_angle, speed=0.5, 
                               wheelbase_length=0.8, wheel_radius=0.15, 
                               track_width=0.4, max_steering_angle=30.0):
    """
    Convert steering angle (degrees) + speed (m/s) to individual wheel speeds.
    Uses arcade-style yaw rate control for simplicity.
    
    Parameters:
    - steering_angle: angle in degrees (positive = left turn)
    - speed: forward speed in m/s (positive = forward)
    - track_width: distance between left/right wheels (m)
    - wheel_radius: wheel radius (m)
    - max_steering_angle: maximum steering angle (degrees)
    
    Returns:
    - (w_fl, w_fr, w_rl, w_rr): wheel angular speeds in rad/s
    """
    steering_angle = clip_steering_angle(steering_angle, max_steering_angle)
    
    # Convert steering angle to yaw rate (simple linear mapping)
    max_yaw_rate = np.radians(400.0)  # rad/s at max steering angle (vitesse réduite)
    steering_angle_rad = np.radians(steering_angle)
    yaw_rate = steering_angle_rad * (max_yaw_rate / np.radians(max_steering_angle))
    
    return yaw_rate_to_wheel_speeds(yaw_rate_rps=yaw_rate,
                                    speed=speed,
                                    track_width=track_width,
                                    wheel_radius=wheel_radius,
                                    max_wheel_rad_s=100.0)  # Limit to ±100 rad/s


class CorridorEnv(gym.Env):
    """
    Robot 4 roues naviguant un corridor avec trous et bumps.
    CONTRÔLE PAR VOLANT : steering_angle + speed au lieu de 4 roues indépendantes.
    
    Observation:
        - Position robot (x, y, z): 3
        - Vitesse robot (vx, vy, vz): 3  
        - Bounding box coins (4 coins × 2 coords): 8
        - Grille environnement 60×30: 1800 (0=sol, 0.5=bump, 1=trou)
    
    Action (2 valeurs):
        - steering_angle: angle de volant en degrés (±30°)
        - speed: vitesse avant/arrière en m/s (±2 m/s)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, max_steps=3000, corridor_xml="corridor_3x100_no_full_obstacles.xml", obstacle_type="both", use_fixed_seed=False, random_percentage=None):
        super().__init__()
        
        self.robot_xml = "four_wheel_robot.xml"
        self.corridor_xml = corridor_xml
        self.use_random_corridor = corridor_xml is None
        self.obstacle_type = obstacle_type  # "holes", "bumps", "both"
        self.use_fixed_seed = use_fixed_seed  # True = seed fixe, False = seed aléatoire
        self.random_percentage = random_percentage  # Pourcentage pour relancer le dé à chaque reset
        
        # Créer un générateur aléatoire indépendant pour cet environnement
        import random
        import time
        import os
        # Utiliser PID + temps + adresse mémoire pour avoir un seed unique par environnement
        unique_seed = hash((os.getpid(), time.time(), id(self))) % (2**32)
        self.env_random = random.Random(unique_seed)
        
        # CHARGER LA CONFIG EN PREMIER pour avoir corridor_length et corridor_width
        self.max_steps = max_steps
        self.corridor_length = 110.0  # Valeur par défaut
        self.corridor_width = 3.0     # Valeur par défaut
        self.success_distance = 100.0 # Valeur par défaut
        
        # Paramètres de vision (par défaut si pas de config)
        self.cell_size = 0.1
        self.vision_front = 5.2
        self.vision_behind = 0.5
        self.vision_left = 1.5
        self.vision_right = 1.5
        self.history_length = 4
        self.history_interval = 15
        
        # Charger config pour override les valeurs par défaut
        self.load_config_if_available()
        
        # MAINTENANT générer le modèle avec les bonnes valeurs
        if self.use_random_corridor:
            # Générer premier corridor aléatoire avec le nouveau système
            from corridor_generator_similar import CorridorGenerator
            self.corridor_generator = CorridorGenerator()
            self.current_corridor_type = f"{self.obstacle_type}-unknown"  # Initialiser
            self.model = self._build_model_from_new_generator()
        else:
            # Utiliser corridor XML fixe
            self.current_corridor_type = f"{self.obstacle_type}-fixed"  # Initialiser
            self.model = self._build_model_from_xml(corridor_xml)
        
        self.data = mujoco.MjData(self.model)
        
        # Calculer dimensions dynamiquement
        self._calculate_vision_dimensions()
        
    def load_config_if_available(self):
        """Charge la configuration depuis config.yaml si disponible."""
        import yaml
        import os
        
        config_path = "config.yaml"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Charger paramètres de vision
                if 'vision' in config:
                    vision_config = config['vision']
                    self.cell_size = vision_config.get('cell_size', self.cell_size)
                    self.vision_front = vision_config.get('vision_front', self.vision_front)
                    self.vision_behind = vision_config.get('vision_behind', self.vision_behind)
                    self.vision_left = vision_config.get('vision_left', self.vision_left)
                    self.vision_right = vision_config.get('vision_right', self.vision_right)
                
                # Charger paramètres d'historique
                if 'history' in config:
                    history_config = config['history']
                    self.history_length = history_config.get('history_length', self.history_length)
                    self.history_interval = history_config.get('history_interval', self.history_interval)
                
                # Charger paramètres robot
                if 'robot' in config:
                    robot_config = config['robot']
                    self.max_steering_angle = robot_config.get('max_steering_angle', 30.0)
                    self.max_speed = robot_config.get('max_speed', 1.0)
                    self.spawn_angle_max_deg = robot_config.get('spawn_angle_max', 60.0)
                else:
                    self.max_steering_angle = 30.0
                    self.max_speed = 1.0
                    self.spawn_angle_max_deg = 60.0
                
                # Charger paramètres des corridors
                if 'corridor' in config:
                    corridor_config = config['corridor']
                    self.corridor_length = corridor_config.get('corridor_length', 110.0)
                    self.corridor_width = corridor_config.get('corridor_width', 3.0)
                    self.success_distance = corridor_config.get('success_distance', 100.0)
                else:
                    # Valeurs par défaut
                    self.corridor_length = 110.0
                    self.corridor_width = 3.0
                    self.success_distance = 100.0
                
                # Charger paramètres de récompenses
                if 'rewards' in config:
                    rewards_config = config['rewards']
                    self.success_reward = rewards_config.get('success_reward', 100.0)
                    self.failure_penalty = rewards_config.get('failure_penalty', -10.0)
                    self.progress_multiplier = rewards_config.get('progress_multiplier', 2.0)
                    self.collision_penalty = rewards_config.get('collision_penalty', -0.01)
                    self.fell_threshold = rewards_config.get('fell_threshold', 0.15)
                    
                    # Charger paramètres de terminaison par manque de progrès
                    self.no_progress_check_interval = rewards_config.get('no_progress_check_interval', 200)
                    self.no_progress_min_distance = rewards_config.get('no_progress_min_distance', 0.5)
                    self.no_progress_penalty = rewards_config.get('no_progress_penalty', -4.0)
                else:
                    # Valeurs par défaut
                    self.success_reward = 100.0
                    self.failure_penalty = -10.0
                    self.progress_multiplier = 2.0
                    self.collision_penalty = -0.01
                    self.fell_threshold = 0.15
                    self.no_progress_check_interval = 200
                    self.no_progress_min_distance = 0.5
                    self.no_progress_penalty = -4.0
                    
            except Exception as e:
                print(f"WARNING: Error loading config: {e}")
                # Valeurs par défaut si erreur
                self.max_steering_angle = 30.0
                self.max_speed = 1.0
                self.spawn_angle_max_deg = 60.0
                self.corridor_length = 110.0
                self.corridor_width = 3.0
                self.success_distance = 100.0
                self.success_reward = 100.0
                self.failure_penalty = -10.0
                self.progress_multiplier = 2.0
                self.collision_penalty = -0.01
                self.fell_threshold = 0.15
                self.no_progress_check_interval = 200
                self.no_progress_min_distance = 0.5
                self.no_progress_penalty = -4.0
        else:
            # Valeurs par défaut si pas de config
            self.max_steering_angle = 30.0
            self.max_speed = 1.0
            self.spawn_angle_max_deg = 60.0
            self.success_reward = 100.0
            self.failure_penalty = -10.0
            self.progress_multiplier = 2.0
            self.collision_penalty = -0.01
            self.fell_threshold = 0.15
    
    def _calculate_vision_dimensions(self):
        """Calcule toutes les dimensions basées sur les paramètres de vision."""
        # Dimensions physiques
        self.vision_length = self.vision_behind + self.vision_front
        self.vision_width = self.vision_left + self.vision_right
        
        # Dimensions de grille
        self.grid_rows = int(self.vision_length / self.cell_size)
        self.grid_cols = int(self.vision_width / self.cell_size)
        self.robot_row_in_grid = round(self.vision_behind / self.cell_size)
        self.robot_col_in_grid = round(self.vision_left / self.cell_size)
        
        # Dimensions pour le réseau de neurones
        self.history_dim = self.history_length * 6  # positions (3) + vitesses (3) par frame
        self.grid_dim = self.grid_rows * self.grid_cols * 2  # 2 canaux
        
        # Afficher les dimensions seulement pour le premier environnement
        if not hasattr(CorridorEnv, '_dimensions_printed'):
            print(f"DIMENSIONS: Calculated dimensions:")
            print(f"   Vision: {self.vision_length:.1f}m x {self.vision_width:.1f}m")
            print(f"   Grid: {self.grid_rows} x {self.grid_cols} cells of {self.cell_size}m")
            print(f"   Robot: row {self.robot_row_in_grid}, column {self.robot_col_in_grid}")
            print(f"   History: {self.history_dim} values ({self.history_length} frames)")
            print(f"   Grid NN: {self.grid_dim} values")
            CorridorEnv._dimensions_printed = True
        
        # Paramètres contrôle par volant (lus depuis le XML du robot)
        self._load_robot_params_from_xml()
        
    def _load_robot_params_from_xml(self):
        """Charge les paramètres physiques du robot depuis le fichier XML."""
        import xml.etree.ElementTree as ET
        
        try:
            # Parser le XML du robot
            tree = ET.parse(self.robot_xml)
            root = tree.getroot()
            
            # Extraire les paramètres depuis les positions des roues
            # Roues avant : pos=" 0.25  0.20 -0.10" et pos=" 0.25 -0.20 -0.10"
            # Roues arrière : pos="-0.25  0.20 -0.10" et pos="-0.25 -0.20 -0.10"
            
            # Trouver les positions des roues
            wheel_positions = {}
            for body in root.findall(".//body[@name]"):
                name = body.get('name')
                if name and name.startswith('wheel_'):
                    pos = body.get('pos', '0 0 0')
                    x, y, z = map(float, pos.split())
                    wheel_positions[name] = (x, y, z)
            
            if len(wheel_positions) >= 4:
                # Calculer wheelbase_length (distance avant/arrière)
                front_x = wheel_positions.get('wheel_fl', (0.25, 0, 0))[0]
                rear_x = wheel_positions.get('wheel_rl', (-0.25, 0, 0))[0]
                self.wheelbase_length = abs(front_x - rear_x)
                
                # Calculer track_width (distance gauche/droite)
                left_y = wheel_positions.get('wheel_fl', (0, 0.20, 0))[1]
                right_y = wheel_positions.get('wheel_fr', (0, -0.20, 0))[1]
                self.track_width = abs(left_y - right_y)
                
                # Extraire wheel_radius depuis la géométrie cylindrique
                for geom in root.findall(".//geom[@type='cylinder']"):
                    size = geom.get('size', '0.15 0.03')
                    radius, half_width = map(float, size.split())
                    self.wheel_radius = radius
                    break
                
                # Afficher les paramètres robot seulement pour le premier environnement
                if not hasattr(CorridorEnv, '_robot_params_printed'):
                    print(f"ROBOT: Parameters read from {self.robot_xml}:")
                    print(f"   Wheelbase: {self.wheelbase_length:.2f}m")
                    print(f"   Track width: {self.track_width:.2f}m") 
                    print(f"   Wheel radius: {self.wheel_radius:.2f}m")
                    print(f"ROBOT: Control parameters from config:")
                    print(f"   Max steering: {self.max_steering_angle:.1f} deg")
                    print(f"   Max speed: {self.max_speed:.1f} m/s")
                    print(f"   Spawn angle: +/-{self.spawn_angle_max_deg:.1f} deg")
                    CorridorEnv._robot_params_printed = True
                
            else:
                raise ValueError("Impossible de trouver les 4 roues dans le XML")
                
        except Exception as e:
            print(f"WARNING: Error reading robot XML: {e}")
            print("   Using default values")
            # Valeurs par défaut si échec
            self.wheelbase_length = 0.5
            self.track_width = 0.4
            self.wheel_radius = 0.15
        
        # Historique des positions pour anticipation
        self.position_history = []  # Buffer des positions + vitesses
        
        # Période de stabilisation
        self.stabilization_steps = 20  # Pas d'actions pendant 20 steps
        
        # Construire carte (sera regénérée à chaque reset si random_corridor=True)
        self.cell_map = self._build_cell_map()
        
        # Espaces - CNN 2 canaux + historique simplifié (sans bounding box)
        obs_size = 7 + self.history_dim + self.grid_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        # CONTRÔLE PAR VOLANT : 2 actions au lieu de 4
        # action[0] = steering_angle (±1.0 → ±30°)
        # action[1] = speed (±1.0 → ±2 m/s)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        
        # État
        self.step_count = 0
        self.prev_x = 0.0
        
        # IDs corps MuJoCo
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Vérifier si on doit forcer la régénération (changement d'obstacle_type)
        force_regen = getattr(self, '_force_regeneration', False)
        if force_regen:
            # Supprimer le flag et forcer la génération
            self._force_regeneration = False
            if hasattr(self, '_fixed_model_generated'):
                delattr(self, '_fixed_model_generated')
        
        # Décider si on génère un nouveau corridor ou on réutilise le modèle existant
        should_generate_new_corridor = False
        use_fixed_seed_for_generation = False
        
        if self.use_random_corridor:
            # Cas 1: Toujours générer aléatoirement (obstacle_type="both" ou pas de random_percentage)
            if self.random_percentage is None or self.obstacle_type == "both":
                should_generate_new_corridor = True
                use_fixed_seed_for_generation = self.use_fixed_seed
            else:
                # Cas 2: Décider selon random_percentage (obstacle_type="holes" ou "bumps")
                dice_roll = self.env_random.random()  # Utiliser le générateur indépendant
                use_random_this_reset = dice_roll < self.random_percentage
                
                if use_random_this_reset:
                    # Générer un corridor aléatoire (nouveau à chaque fois)
                    should_generate_new_corridor = True
                    use_fixed_seed_for_generation = False  # Random seed
                else:
                    # Utiliser un corridor fixe (même modèle à chaque reset)
                    # Générer seulement si c'est le premier reset ou si on n'a pas encore de modèle fixe
                    if not hasattr(self, '_fixed_model_generated'):
                        should_generate_new_corridor = True
                        use_fixed_seed_for_generation = True   # Fixed seed
                        self._fixed_model_generated = True
                    else:
                        should_generate_new_corridor = False  # Réutiliser le modèle existant
        
        if should_generate_new_corridor or force_regen:
            # Générer un nouveau corridor
            if hasattr(self, 'corridor_generator'):
                # Utiliser le nouveau générateur avec la décision prise
                self.model = self._build_model_from_new_generator(use_fixed_seed_for_generation)
                self.data = mujoco.MjData(self.model)
                self.cell_map = self._build_cell_map_from_xml()
            
            # Mettre à jour robot_body_id pour le nouveau modèle
            self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        else:
            # Réutiliser le corridor existant, juste reset les données
            mujoco.mj_resetData(self.model, self.data)
            # IMPORTANT: Initialiser le seed à -1 si on ne régénère pas
            if not hasattr(self, 'last_corridor_seed'):
                self.last_corridor_seed = -1
        
        # SPAWN aléatoire normal
        spawn_x = 0.75  # Position sûre sur floor_flat_70 à 77
        spawn_y = np.random.uniform(-1.0, 1.0)  # Y aléatoire entre -1 et 1
        spawn_angle_rad = np.radians(self.spawn_angle_max_deg)  # Convertir degrés en radians
        spawn_angle = np.random.uniform(-spawn_angle_rad, spawn_angle_rad)  # Angle aléatoire depuis config
        
        # Position
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = 0.30  # Sol à 0.05m + roue 0.15m + chassis 0.10m = 0.30m
        
        # Orientation (quaternion Z-axis)
        self.data.qpos[3] = np.cos(spawn_angle / 2)
        self.data.qpos[4] = 0
        self.data.qpos[5] = 0
        self.data.qpos[6] = np.sin(spawn_angle / 2)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.prev_x = self.data.qpos[0]
        self.success_reached = False  # Reset du flag de succès
        
        # Tracking pour terminaison par manque de progrès
        self.last_progress_check_x = self.data.qpos[0]
        self.last_progress_check_step = 0
        
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
            # CONTRÔLE PAR VOLANT : convertir steering + speed en vitesses des 4 roues
            action = np.clip(action, -1.0, 1.0)
            
            # Extraire steering angle et speed des actions
            steering_angle = action[0] * self.max_steering_angle  # ±30 degrés
            speed = action[1] * self.max_speed                    # ±2 m/s
            
            # Convertir en vitesses des 4 roues
            w_fl, w_fr, w_rl, w_rr = steer_angle_to_wheel_speeds(
                steering_angle=steering_angle,
                speed=speed,
                wheelbase_length=self.wheelbase_length,
                wheel_radius=self.wheel_radius,
                track_width=self.track_width,
                max_steering_angle=self.max_steering_angle
            )
            
            # Appliquer les vitesses aux roues (ordre: FL, FR, RL, RR)
            self.data.ctrl[:] = [w_fl, w_fr, w_rl, w_rr]
        
        # Simuler (4 substeps)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Mettre à jour historique des positions
        if self.step_count % self.history_interval == 0:
            self._update_position_history()
        
        # Vérifier le progrès tous les N steps
        if hasattr(self, 'no_progress_check_interval') and self.step_count > 0:
            steps_since_check = self.step_count - self.last_progress_check_step
            if steps_since_check >= self.no_progress_check_interval:
                current_x = self.data.qpos[0]
                progress = current_x - self.last_progress_check_x
                
                if progress < self.no_progress_min_distance:
                    # Pas assez de progrès, terminer l'épisode
                    reward = self.no_progress_penalty
                    terminated = True
                    info = {
                        'reason': 'no_progress',
                        'progress': progress,
                        'required': self.no_progress_min_distance
                    }
                    info.update(self._get_info())
                    return self._get_obs(), reward, terminated, False, info
                
                # Mettre à jour pour le prochain check
                self.last_progress_check_x = current_x
                self.last_progress_check_step = self.step_count
        
        # Récompense et terminaison
        reward, terminated, info = self._compute_reward()
        
        # Vérifier si c'est un succès pour truncate l'épisode
        if info.get('reason') == 'success':
            truncated = True  # Succès = truncated (pas terminated car pas un échec)
        else:
            truncated = self.step_count >= self.max_steps
        
        info.update(self._get_info())
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation simplifiée avec historique des positions pour anticipation."""
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        
        # Angle du robot par rapport au corridor (orientation)
        quat = self.data.qpos[3:7]
        robot_angle = 2 * np.arctan2(quat[3], quat[0])  # Angle en radians
        
        # Historique des positions (plus de bounding box dans l'obs principale)
        position_history = self._get_position_history_obs()
        
        # Grille environnement 2 canaux
        grid = self._get_grid_obs(pos[0], pos[1])  # 60×30×2
        
        return np.concatenate([
            pos,                    # 3 valeurs (position actuelle)
            vel,                    # 3 valeurs (vitesse actuelle)
            [robot_angle],          # 1 valeur (angle du robot)
            position_history,       # 48 valeurs (8 frames × 6 valeurs: 3 positions + 3 vitesses)
            grid.flatten()          # 3600 valeurs (60×30×2)
        ]).astype(np.float32)       # Total: 7 + 48 + 3600 = 3655 valeurs
    
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
        robot_grid_row = self.robot_row_in_grid  # Position Y basée sur vision_behind
        robot_grid_col = self.robot_col_in_grid  # Position X basée sur vision_left
        
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
        """Mettre à jour l'historique des positions + vitesses (sans bounding box)."""
        # Position actuelle (x, y, z)
        current_position = self.data.qpos[:3].copy()
        
        # Vitesses actuelles
        current_velocities = self.data.qvel[:3].copy()  # vx, vy, vz
        
        # Stocker position + vitesses (3 + 3 = 6 valeurs par frame)
        frame_data = np.concatenate([current_position, current_velocities])
        self.position_history.append(frame_data)
        
        # Garder seulement les N dernières positions
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
    
    def _get_position_history_obs(self):
        """Obtenir l'historique des positions + vitesses en coordonnées RELATIVES."""
        # État actuel
        current_position = self.data.qpos[:3]
        current_velocities = self.data.qvel[:3]
        
        history_obs = []
        
        for i in range(self.history_length):
            if i < len(self.position_history):
                # Frame passée (3 positions + 3 vitesses = 6 valeurs)
                past_frame = self.position_history[i]
                past_position = past_frame[:3]  # 3 premiers = position
                past_velocities = past_frame[3:]  # 3 derniers = vitesses
                
                # Position relative (différence par rapport à position actuelle)
                relative_position = past_position - current_position
                
                # Vitesses relatives (différence par rapport à vitesse actuelle)
                relative_velocities = past_velocities - current_velocities
                
                # Combiner position + vitesses (3 + 3 = 6 valeurs)
                history_obs.extend(relative_position)
                history_obs.extend(relative_velocities)
            else:
                # Remplir avec zéros si pas assez d'historique
                history_obs.extend([0.0] * 6)  # 3 positions + 3 vitesses
        
        return np.array(history_obs, dtype=np.float32)
    
    def _get_grid_obs(self, robot_x, robot_y):
        """Grille 2 canaux 60×30×2 avec environnement, CENTRÉE ET ORIENTÉE selon le robot.
        
        Canal 0: Obstacles/Bumps (1.0 = bump OU murs latéraux, 0.0 = navigable)
        Canal 1: Trous (1.0 = trou OU extérieur avant/arrière, 0.0 = navigable)
        
        Sol navigable = les deux canaux à 0.0
        Logique physique:
        - Côtés gauche/droite du couloir = murs infinis (obstacles)
        - Devant/derrière du couloir = vide (trous, robot tombe)
        """
        # 2 canaux binaires : [obstacles, trous]
        grid = np.zeros((self.grid_rows, self.grid_cols, 2), dtype=np.float32)
        
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
                relative_y = (j - self.robot_col_in_grid) * self.cell_size     # Distance gauche/droite
                
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
                # Le couloir navigable va de -width/2 à +width/2
                # Tout ce qui est en dehors = obstacle (murs latéraux)
                if world_y <= -self.corridor_width/2 or world_y >= self.corridor_width/2:
                    # En dehors du couloir sur les CÔTÉS = obstacle (mur)
                    grid[i, j, 0] = 1.0  # Obstacle (mur latéral)
                    grid[i, j, 1] = 0.0  # Pas de trou
                elif world_x < 0 or world_x > self.corridor_length:
                    # En dehors du couloir DEVANT/DERRIÈRE = trou (vide)
                    grid[i, j, 0] = 0.0  # Pas d'obstacle
                    grid[i, j, 1] = 1.0  # Trou (extérieur avant/arrière)
                else:
                    # Dans le couloir : chercher dans la carte des cellules
                    # Vérifier que les indices sont valides
                    if world_row < 0 or world_col < 0:
                        # Hors limites négatives = trou
                        grid[i, j, 0] = 0.0
                        grid[i, j, 1] = 1.0
                    else:
                        cell_type = self.cell_map.get((world_row, world_col), 2)  # Défaut trou
                        
                        # Remplir les 2 canaux binaires
                        if cell_type == 0:  # Sol
                            grid[i, j, 0] = 0.0  # Pas d'obstacle
                            grid[i, j, 1] = 0.0  # Pas de trou
                            # Sol = les deux canaux à 0.0
                        elif cell_type == 1:  # Bump
                            grid[i, j, 0] = 1.0  # Obstacle (bump)
                            grid[i, j, 1] = 0.0  # Pas de trou
                        else:  # cell_type == 2, Trou
                            grid[i, j, 0] = 0.0  # Pas d'obstacle
                            grid[i, j, 1] = 1.0  # Trou
        
        return grid
        
        return grid
    
    def _compute_reward(self):
        """Récompense SIMPLE et naturelle: avancer = bien, échouer = mal."""
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        z = self.data.qpos[2]
        
        terminated = False
        info = {}
        
        # Échec: tombé dans un trou
        if z < self.fell_threshold:
            info['reason'] = 'fell'
            return self.failure_penalty, True, info
        
        # Échec: robot retourné
        quat = self.data.qpos[3:7]
        up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
        if up_z < 0:
            info['reason'] = 'flipped'
            return self.failure_penalty, True, info
        
        # Collision désactivée pour terminaison, mais pénalité légère appliquée
        collision_penalty_value = 0.0
        if self._is_colliding_with_bump():
            collision_penalty_value = self.collision_penalty
            # Debug optionnel (commenté pour éviter le spam)
            # print(f"DEBUG: collision penalty at step {self.step_count}, pos=({x:.3f},{y:.3f},{z:.3f})")
        
        # Récompense SIMPLE: progression + pénalité collision
        delta_x = x - self.prev_x
        self.prev_x = x
        progress_reward = delta_x * self.progress_multiplier
        reward = progress_reward + collision_penalty_value
        
        # Succès: atteindre la distance de succès (termine l'épisode en truncated)
        if x >= self.success_distance:
            if not hasattr(self, 'success_reached') or not self.success_reached:
                self.success_reached = True
                info['reason'] = 'success'
                reward += self.success_reward
                # Terminer l'épisode en truncated (succès, pas un échec)
                # On retourne directement pour que step() gère le truncated
                return reward, False, info  # terminated=False, truncated sera géré dans step()
        
        if info.get('reason') is None:
            info['reason'] = None
        
        return reward, terminated, info
    
    def _is_colliding_with_bump(self):
        """Détecte si le robot est en collision avec un bump (pilier) ou un mur (wall)."""
        # Parcourir tous les contacts actifs
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Récupérer les noms des géométries en contact
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Vérifier si une des géométries est un bump/wall et l'autre fait partie du robot
            robot_geoms = ['chassis', 'geom_fl', 'geom_fr', 'geom_rl', 'geom_rr']
            
            geom1_is_robot = geom1_name in robot_geoms if geom1_name else False
            geom2_is_robot = geom2_name in robot_geoms if geom2_name else False
            
            # Détecter bumps ET walls
            geom1_is_obstacle = geom1_name and ('bump' in geom1_name or 'wall' in geom1_name)
            geom2_is_obstacle = geom2_name and ('bump' in geom2_name or 'wall' in geom2_name)
            
            # Collision détectée si robot touche un bump ou un wall
            if (geom1_is_robot and geom2_is_obstacle) or (geom2_is_robot and geom1_is_obstacle):
                return True
        
        return False
    
    def _get_info(self):
        corridor_type = getattr(self, 'current_corridor_type', f"{self.obstacle_type}-unknown")
        corridor_seed = getattr(self, 'last_corridor_seed', -1)
        return {
            'x': float(self.data.qpos[0]),
            'y': float(self.data.qpos[1]),
            'z': float(self.data.qpos[2]),
            'step': self.step_count,
            'corridor_type': corridor_type.split('-')[0],  # "holes", "bumps", "both"
            'is_random': 'random' in corridor_type,
            'corridor_seed': corridor_seed
        }
    
    def _build_cell_map(self):
        """Construire carte initiale."""
        if self.use_random_corridor:
            if hasattr(self, 'corridor_generator'):
                # Nouveau système: construire depuis le modèle XML
                return self._build_cell_map_from_xml()
        else:
            return self._build_cell_map_from_xml()

    def _build_model_from_new_generator(self, use_fixed_seed=None):
        """Construire modèle MuJoCo avec robot + corridor généré par le nouveau système."""
        import numpy as np
        
        # Utiliser le paramètre passé ou la valeur par défaut
        if use_fixed_seed is None:
            use_fixed_seed = self.use_fixed_seed
        
        # Convertir obstacle_type en bump_ratio
        bump_ratio = getattr(self, 'bump_ratio', 0.0)
        
        # Générer les paramètres du corridor
        if use_fixed_seed:
            # Seed fixe pour reproductibilité
            seed = 12345 + int(bump_ratio * 1000)  # Seed différent selon bump_ratio
            self.current_corridor_type = f"holes+{int(bump_ratio*100)}%bumps-fixed"
            
            length = self.corridor_length  # UTILISER LA CONFIG
            width = self.corridor_width    # UTILISER LA CONFIG
        else:
            # Seed aléatoire - utiliser le générateur indépendant de l'environnement
            seed = self.env_random.randint(0, 10000)
            # Varier autour de la longueur configurée (±10%)
            length = self.corridor_length * (0.9 + self.env_random.random() * 0.2)
            width = self.corridor_width  # Largeur fixe pour cohérence avec les obstacles
            
            self.current_corridor_type = f"holes+{int(bump_ratio*100)}%bumps-random"
        
        # Stocker le seed pour logging (dans tous les cas)
        self.last_corridor_seed = seed
        
        # Générer XML en mémoire (pas de sauvegarde fichier)
        corridor_xml_str = self.corridor_generator.generate_corridor_xml(
            length=length, 
            width=width, 
            seed=seed, 
            name="random_corridor", 
            obstacle_type="holes",  # Toujours holes
            bump_ratio=bump_ratio   # + ratio de bumps
        )
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
                    body.set('pos', '0.75 0 0.30')  # Hauteur corrigée
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
                    body.set('pos', '0.75 0 0.30')  # Hauteur corrigée
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
            if 'hole' in name_lower:
                cell_type = 2  # Trou (vérifier AVANT 'floor' car 'floor_hole' contient 'floor')
            elif 'bump' in name_lower:
                cell_type = 1  # Bump
            elif 'wall' in name_lower:
                cell_type = 1  # Mur = obstacle (comme bump)
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
    
    def update_curriculum_params(self, random_percentage=None, obstacle_type=None, max_steps=None, bump_ratio=None):
        """Mettre à jour les paramètres du curriculum pour cet environnement."""
        force_regeneration = False
        
        if random_percentage is not None:
            self.random_percentage = random_percentage
        
        if bump_ratio is not None:
            old_bump_ratio = getattr(self, 'bump_ratio', 0.0)
            if bump_ratio != old_bump_ratio:
                # Si le ratio de bumps change, forcer la régénération
                self.bump_ratio = bump_ratio
                force_regeneration = True
                
                # Mettre à jour current_corridor_type immédiatement pour l'affichage
                if hasattr(self, 'current_corridor_type'):
                    old_corridor_type = self.current_corridor_type
                    if 'random' in old_corridor_type:
                        self.current_corridor_type = f"holes+{int(bump_ratio*100)}%bumps-random"
                    else:
                        self.current_corridor_type = f"holes+{int(bump_ratio*100)}%bumps-fixed"
                else:
                    self.current_corridor_type = f"holes+{int(bump_ratio*100)}%bumps-unknown"
            else:
                self.bump_ratio = bump_ratio
        
        # Backward compatibility avec obstacle_type (deprecated)
        if obstacle_type is not None:
            self.obstacle_type = obstacle_type
            
        if max_steps is not None:
            self.max_steps = max_steps
        
        # Forcer la régénération du corridor au prochain reset
        if force_regeneration:
            # Supprimer le modèle fixe en cache pour forcer régénération
            if hasattr(self, '_fixed_model_generated'):
                delattr(self, '_fixed_model_generated')
            # Marquer qu'on doit régénérer au prochain reset
            self._force_regeneration = True
    
    def close(self):
        pass