"""
Gymnasium environment for the 4-wheel robot in corridor with MuJoCo.
Compatible with PPO and other RL algorithms.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


class RobotCorridorEnv(gym.Env):
    """
    Custom Gymnasium environment for 4-wheel robot navigating a corridor.
    CORRECTED VERSION: Includes floor observation (Question 7 + Question 14).
    
    Observation Space: 
        - Robot position (x, y, z): 3 values
        - Robot velocity (vx, vy, vz): 3 values
        - Floor grid (4 rows × 6 cols): 24 values
        Total: 30 continuous values
    
    Action Space:
        - 4 continuous values in [-1, 1] for each wheel torque
    
    Reward:
        - Progress-based: +10 * delta_x (forward movement)
        - Terminal: +100 for reaching goal, -100 for falling
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 corridor_xml="corridor_3x100.xml",
                 robot_xml="four_wheels_robot.xml",
                 max_steps=3000, 
                 render_mode=None):
        super().__init__()
        
        # Load MuJoCo model by combining robot and corridor XMLs
        self.model = self._build_combined_model(robot_xml, corridor_xml)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_width = 0.5
        self.render_mode = render_mode
        
        # Floor observation parameters (from notebook Question 7)
        self.n_rows_ahead = 4      # Look 4 rows ahead (2m) for better anticipation
        self.n_rows_behind = 1     # Look 1 row behind
        self.n_cols = 6            # 6 cells per row (3m / 0.5m)
        self.n_observation_rows = self.n_rows_behind + 1 + self.n_rows_ahead  # 6 rows
        
        # Build cell map after model is loaded
        self.cell_map_semantic = self._build_cell_map()
        
        # Observation space: [x, y, z, vx, vy, vz] + floor grid (6 rows × 6 cols = 36 cells)
        # Total: 6 + 36 = 42 values
        obs_size = 6 + (self.n_observation_rows * self.n_cols)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=110.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: 4 wheel torques in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.previous_x = 0.0
        
        # Rendering
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Forward simulation to initialize qpos/qvel
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking variables
        self.current_step = 0
        # Safe initialization: use 0 if qpos is empty
        self.previous_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Apply action (clip to valid range and scale to actuator range)
        # No forward bias - let the robot learn to navigate around obstacles
        action = np.clip(action, -1.0, 1.0) * 15.0  # Scale to [-15, 15] for good speed
        self.data.ctrl[:] = action
        
        # Step simulation (2 substeps - balance between speed and stability)
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation
        observation = self._get_obs()
        
        # Compute reward
        reward, terminated, info = self._compute_reward()
        
        # Check truncation
        truncated = self.current_step >= self.max_steps
        
        # Update info
        info.update(self._get_info())
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get current observation (CORRECTED: includes floor observation)."""
        # Position and velocity
        pos = self.data.qpos[:3]  # x, y, z
        vel = self.data.qvel[:3]  # vx, vy, vz
        
        # Floor observation (grid of cells around robot)
        robot_x, robot_y = pos[0], pos[1]
        floor_obs = self._get_floor_observation(robot_x, robot_y)
        floor_flat = floor_obs.flatten()  # Flatten 4x6 grid to 24 values
        
        # Combine: [x, y, z, vx, vy, vz, floor_cells...]
        obs = np.concatenate([pos, vel, floor_flat]).astype(np.float32)
        return obs
    
    def _get_floor_observation(self, robot_x, robot_y):
        """
        Get floor cell observations around the robot (from notebook Question 7).
        
        Returns a grid of cell types:
        - 0 = flat
        - 1 = bump  
        - 2 = hole
        
        Shape: (n_rows_behind + 1 + n_rows_ahead, n_cols) = (6, 6)
        """
        # Calculate robot's cell position
        robot_row = int(robot_x / self.cell_width)
        robot_col = int((robot_y + self.corridor_width/2) / self.cell_width)
        
        # Total number of rows in observation
        total_rows = self.n_observation_rows
        
        # Create observation array
        observation = np.zeros((total_rows, self.n_cols), dtype=np.float32)
        
        # Fill observation array
        for i in range(total_rows):
            # Calculate actual row index (behind -> under -> ahead)
            row_offset = i - self.n_rows_behind
            actual_row = robot_row + row_offset
            
            for j in range(self.n_cols):
                # Get cell type from semantic map
                cell_type = self.cell_map_semantic.get((actual_row, j), 0)
                observation[i, j] = cell_type if cell_type is not None else 0
        
        return observation
    
    def _build_cell_map(self):
        """
        Build cell map from MuJoCo geometries (from notebook Question 7).
        Maps (row, col) -> cell_type (0=flat, 1=bump, 2=hole).
        """
        cell_map = {}
        n_x = int(self.corridor_length / self.cell_width)  # 200 rows
        n_y = self.n_cols  # 6 columns
        half_width = self.corridor_width / 2.0
        
        # Initialize all cells as holes (2) - zones without geometry are holes
        for r in range(n_x):
            for c in range(n_y):
                cell_map[(r, c)] = 2  # hole by default
        
        # Helper functions
        def idx_to_p(ix, iy):
            x = (ix + 0.5) * self.cell_width
            y = (iy + 0.5) * self.cell_width - half_width
            return x, y
        
        def contains_2D(p, center, sizes):
            x, y = p
            cx, cy, _ = center
            half_size_x, half_size_y, _ = sizes
            min_x = cx - half_size_x
            max_x = cx + half_size_x
            min_y = cy - half_size_y
            max_y = cy + half_size_y
            return (min_x <= x <= max_x) and (min_y <= y <= max_y)
        
        def get_cell_type_from_name(name):
            name_lower = name.lower()
            if 'bump' in name_lower or 'ramp' in name_lower:
                return 1  # bump (bridge/ramp)
            elif 'flat' in name_lower or 'floor' in name_lower:
                return 0  # flat (ground)
            else:
                return 0  # default to flat if geometry exists
        
        # Store geometries with their Z height for prioritization
        # Higher geometries (bridges) should override lower ones (ground)
        cell_map_with_z = {}  # (r, c) -> (cell_type, z_height)
        
        # Iterate through all geometries and update the map
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and ('cell' in geom_name or 'flat' in geom_name or 'bump' in geom_name.lower() or 'ramp' in geom_name.lower()):
                # Get geometry position and size
                geom_pos = self.model.geom_pos[geom_id]
                geom_size = self.model.geom_size[geom_id]
                geom_z = geom_pos[2]  # Z height
                
                # Get cell type from name
                cell_type = get_cell_type_from_name(geom_name)
                
                # Find which cells this geometry covers
                for r in range(n_x):
                    for c in range(n_y):
                        cell_center_x, cell_center_y = idx_to_p(r, c)
                        if contains_2D((cell_center_x, cell_center_y), geom_pos, geom_size):
                            # Only update if this geometry is higher than existing one
                            if (r, c) not in cell_map_with_z or geom_z > cell_map_with_z[(r, c)][1]:
                                cell_map_with_z[(r, c)] = (cell_type, geom_z)
        
        # Extract just the cell types (discard Z heights)
        for (r, c), (cell_type, _) in cell_map_with_z.items():
            cell_map[(r, c)] = cell_type
        
        return cell_map
    
    def _get_info(self):
        """Get additional information."""
        return {
            'x_position': float(self.data.qpos[0]),
            'y_position': float(self.data.qpos[1]),
            'z_position': float(self.data.qpos[2]),
            'distance_to_goal': float(max(0, self.corridor_length - self.data.qpos[0])),
            'step': self.current_step
        }
    
    def _compute_reward(self):
        """Compute reward and check termination."""
        robot_x = self.data.qpos[0]
        robot_y = self.data.qpos[1]
        robot_z = self.data.qpos[2]
        robot_vx = self.data.qvel[0]  # Forward velocity
        
        terminated = False
        info = {}
        
        # Terminal conditions
        if robot_x >= self.corridor_length:
            # Success: reached goal
            reward = 100.0
            terminated = True
            info['termination_reason'] = 'success'
            print(f"[TERM] SUCCESS at step {self.current_step}: x={robot_x:.2f}")
        elif robot_z < 0.1 and robot_x > 0:
            # Failure: fell in hole
            reward = -100.0
            terminated = True
            info['termination_reason'] = 'fell'
            print(f"[TERM] FELL at step {self.current_step}: x={robot_x:.2f}m, z={robot_z:.2f}m")
        elif robot_x < -1.0:
            # Failure: went too far backward
            reward = -50.0
            terminated = True
            info['termination_reason'] = 'backward'
            print(f"[TERM] BACKWARD at step {self.current_step}: x={robot_x:.2f}")
        else:
            # Progress reward: encourage forward movement
            delta_x = robot_x - self.previous_x
            self.previous_x = robot_x
            reward = delta_x * 10.0  # Scale reward
            
            # Penalty for being stuck (low forward velocity)
            if abs(robot_vx) < 0.1:  # If moving very slowly
                reward -= 0.5  # Strong penalty for being stuck
            
            # Small penalty for time (encourage efficiency)
            reward -= 0.01
            
            info['termination_reason'] = None
        
        return reward, terminated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="tracking")
            return self.renderer.render()
        elif self.render_mode == "human":
            self.renderer.update_scene(self.data, camera="tracking")
            # For human rendering, you'd typically display in a window
            # This is simplified for now
            return self.renderer.render()
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            self.renderer.close()
    
    def _extract_robot_from_xml(self, xml_file_path):
        """Extract robot components from XML file."""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        components = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'robot_body': None,
            'actuators': None,
            'visual': None
        }
        
        for child in root:
            if child.tag == 'compiler':
                components['compiler'] = child
            elif child.tag == 'option':
                components['option'] = child
            elif child.tag == 'default':
                components['default'] = child
            elif child.tag == 'asset':
                components['asset'] = child
            elif child.tag == 'worldbody':
                for body in child:
                    if body.get('name') == 'robot':
                        components['robot_body'] = body
            elif child.tag == 'actuator':
                components['actuators'] = child
            elif child.tag == 'visual':
                components['visual'] = child
        
        return components
    
    def _extract_corridor_from_xml(self, xml_file_path):
        """Extract corridor components from XML file."""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        components = {
            'compiler': None,
            'option': None,
            'default': None,
            'asset': None,
            'corridor_geom': None,
            'actuators': None
        }
        
        for child in root:
            if child.tag == 'compiler':
                components['compiler'] = child
            elif child.tag == 'option':
                components['option'] = child
            elif child.tag == 'default':
                components['default'] = child
            elif child.tag == 'asset':
                components['asset'] = child
            elif child.tag == 'worldbody':
                components['corridor_geom'] = child
            elif child.tag == 'actuator':
                components['actuators'] = child
        
        return components
    
    def _build_combined_model(self, robot_xml, corridor_xml, robot_height=0.45):
        """Build combined MuJoCo model from robot and corridor XMLs."""
        robot_components = self._extract_robot_from_xml(robot_xml)
        corridor_components = self._extract_corridor_from_xml(corridor_xml)
        
        # Create root mujoco element
        root = ET.Element('mujoco')
        root.set('model', 'robot_in_corridor')
        
        # Add compiler settings
        if robot_components['compiler'] is not None:
            root.append(robot_components['compiler'])
        
        # Add physics settings
        option = ET.Element('option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        root.append(option)
        
        # Add size settings
        size = ET.Element('size')
        size.set('njmax', '4000')
        size.set('nconmax', '1000')
        root.append(size)
        
        # Add default settings
        if robot_components['default'] is not None:
            root.append(robot_components['default'])
        
        # Add visual settings
        if robot_components['visual'] is not None:
            root.append(robot_components['visual'])
        
        # Combine assets
        asset = ET.Element('asset')
        added_material_names = set()
        
        if robot_components['asset'] is not None:
            for material in robot_components['asset']:
                material_name = material.get('name', '')
                if material_name not in added_material_names:
                    asset.append(material)
                    added_material_names.add(material_name)
        
        if corridor_components['asset'] is not None:
            for material in corridor_components['asset']:
                material_name = material.get('name', '')
                if material_name not in added_material_names:
                    asset.append(material)
                    added_material_names.add(material_name)
        
        root.append(asset)
        
        # Create worldbody with corridor and robot
        worldbody = ET.Element('worldbody')
        
        # Add corridor geometries
        if corridor_components['corridor_geom'] is not None:
            for geom in corridor_components['corridor_geom']:
                worldbody.append(geom)
        
        # Add robot body with adjusted height
        if robot_components['robot_body'] is not None:
            current_pos = robot_components['robot_body'].get('pos', '1 0 0.3')
            pos_parts = current_pos.split()
            if len(pos_parts) == 3:
                new_pos = f"{pos_parts[0]} {pos_parts[1]} {robot_height}"
                robot_components['robot_body'].set('pos', new_pos)
            worldbody.append(robot_components['robot_body'])
        
        root.append(worldbody)
        
        # Add actuators
        if robot_components['actuators'] is not None:
            root.append(robot_components['actuators'])
        
        # Convert to XML string and create model
        xml_string = ET.tostring(root, encoding='unicode')
        model = mujoco.MjModel.from_xml_string(xml_string)
        
        return model


# Register the environment with Gymnasium
gym.register(
    id='RobotCorridor-v0',
    entry_point='robot_corridor_env:RobotCorridorEnv',
    max_episode_steps=1000,
)
