"""
GPU-accelerated Gymnasium environment using MuJoCo MJX (JAX).
This version runs the physics simulation on GPU for massive speedup.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
from jax import jit, vmap
import mujoco
from mujoco import mjx
import xml.etree.ElementTree as ET


class RobotCorridorEnvMJX(gym.Env):
    """
    GPU-accelerated environment using MuJoCo MJX.
    Physics simulation runs on GPU via JAX.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 corridor_xml="corridor_3x100_no_full_obstacles.xml",
                 robot_xml="four_wheels_robot.xml",
                 max_steps=1000,
                 render_mode=None):
        super().__init__()
        
        # Build combined model
        xml_string = self._build_combined_xml(robot_xml, corridor_xml)
        
        # Load MuJoCo model (CPU)
        self.mj_model = mujoco.MjModel.from_xml_string(xml_string)
        
        # Convert to MJX model (GPU)
        self.model = mjx.put_model(self.mj_model)
        
        # Environment parameters
        self.max_steps = max_steps
        self.corridor_length = 100.0
        self.corridor_width = 3.0
        self.cell_width = 0.5
        self.render_mode = render_mode
        
        # Floor observation parameters
        self.n_rows_ahead = 2
        self.n_rows_behind = 1
        self.n_cols = 6
        self.n_observation_rows = 4
        
        # Build cell map (CPU, done once)
        self.cell_map_semantic = self._build_cell_map()
        
        # Observation space: [x, y, z, vx, vy, vz] + floor grid (24 cells)
        obs_size = 6 + (self.n_observation_rows * self.n_cols)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=110.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: 4 wheel torques
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # State tracking
        self.data = None
        self.current_step = 0
        self.previous_x = 0.0
        
        # JIT-compiled functions for speed
        self._jit_step = jit(self._mjx_step)
        self._jit_reset = jit(self._mjx_reset)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset using MJX (GPU)
        self.data = self._jit_reset(self.model)
        
        # Reset tracking
        self.current_step = 0
        self.previous_x = float(self.data.qpos[0])
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step."""
        # Convert action to JAX array
        action_jax = jnp.array(action, dtype=jnp.float32)
        
        # Step simulation on GPU (5 substeps)
        self.data = self._jit_step(self.model, self.data, action_jax)
        
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
    
    @staticmethod
    def _mjx_reset(model):
        """Reset MJX data (runs on GPU)."""
        data = mjx.make_data(model)
        return data
    
    @staticmethod
    def _mjx_step(model, data, action):
        """Step MJX simulation (runs on GPU)."""
        # Apply action
        data = data.replace(ctrl=action)
        
        # Step 5 times
        def step_once(data, _):
            return mjx.step(model, data), None
        
        data, _ = jax.lax.scan(step_once, data, None, length=5)
        return data
    
    def _get_obs(self):
        """Get current observation."""
        # Convert from JAX to numpy
        pos = np.array(self.data.qpos[:3], dtype=np.float32)
        vel = np.array(self.data.qvel[:3], dtype=np.float32)
        
        # Floor observation
        robot_x, robot_y = float(pos[0]), float(pos[1])
        floor_obs = self._get_floor_observation(robot_x, robot_y)
        floor_flat = floor_obs.flatten()
        
        # Combine
        obs = np.concatenate([pos, vel, floor_flat]).astype(np.float32)
        return obs
    
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
        robot_x = float(self.data.qpos[0])
        robot_z = float(self.data.qpos[2])
        
        terminated = False
        info = {}
        
        # Terminal conditions
        if robot_x >= self.corridor_length:
            reward = 100.0
            terminated = True
            info['termination_reason'] = 'success'
        elif robot_z < 0.1 and robot_x > 0:
            reward = -100.0
            terminated = True
            info['termination_reason'] = 'fell'
        elif robot_x < -1.0:
            reward = -50.0
            terminated = True
            info['termination_reason'] = 'backward'
        else:
            delta_x = robot_x - self.previous_x
            self.previous_x = robot_x
            reward = delta_x * 10.0
            reward -= 0.01
            info['termination_reason'] = None
        
        return reward, terminated, info
    
    def _get_floor_observation(self, robot_x, robot_y):
        """Get floor cell observations around the robot."""
        robot_row = int(robot_x / self.cell_width)
        robot_col = int((robot_y + self.corridor_width/2) / self.cell_width)
        
        observation = np.zeros((self.n_observation_rows, self.n_cols), dtype=np.float32)
        
        for i in range(self.n_observation_rows):
            row_offset = i - self.n_rows_behind
            actual_row = robot_row + row_offset
            
            for j in range(self.n_cols):
                cell_type = self.cell_map_semantic.get((actual_row, j), 0)
                observation[i, j] = cell_type if cell_type is not None else 0
        
        return observation
    
    def _build_cell_map(self):
        """Build cell map - SIMPLIFIED for MJX (flat floor only)."""
        cell_map = {}
        n_x = int(self.corridor_length / self.cell_width)
        n_y = self.n_cols
        
        # For simplified MJX version: all cells are flat (no obstacles)
        # This allows MJX to work without complex collision detection
        for r in range(n_x):
            for c in range(n_y):
                cell_map[(r, c)] = 0  # All flat
        
        return cell_map
    
    def _build_combined_xml(self, robot_xml, corridor_xml):
        """Build combined XML string - SIMPLIFIED for MJX compatibility."""
        robot_components = self._extract_robot_from_xml(robot_xml)
        
        root = ET.Element('mujoco')
        root.set('model', 'robot_in_corridor_mjx')
        
        # Compiler settings
        compiler = ET.Element('compiler')
        compiler.set('angle', 'degree')
        compiler.set('autolimits', 'true')
        root.append(compiler)
        
        # Physics settings
        option = ET.Element('option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        option.set('integrator', 'RK4')  # Better for MJX
        root.append(option)
        
        # Reduced size for MJX
        size = ET.Element('size')
        size.set('njmax', '500')
        size.set('nconmax', '200')
        root.append(size)
        
        # Simplified default (MJX compatible)
        default = ET.Element('default')
        default_geom = ET.SubElement(default, 'geom')
        default_geom.set('condim', '3')  # Standard 3D contacts (MJX compatible)
        default_geom.set('friction', '1 0.005 0.0001')
        root.append(default)
        
        # Simple assets
        asset = ET.Element('asset')
        mat_floor = ET.SubElement(asset, 'material')
        mat_floor.set('name', 'mat_floor')
        mat_floor.set('rgba', '0.8 0.8 0.8 1')
        
        mat_chassis = ET.SubElement(asset, 'material')
        mat_chassis.set('name', 'mat_chassis')
        mat_chassis.set('rgba', '0.8 0.2 0.2 1')
        
        mat_wheel = ET.SubElement(asset, 'material')
        mat_wheel.set('name', 'mat_wheel')
        mat_wheel.set('rgba', '0.1 0.1 0.1 1')
        root.append(asset)
        
        # Worldbody with SIMPLIFIED corridor (MJX compatible)
        worldbody = ET.Element('worldbody')
        
        # Simple floor plane (instead of complex corridor)
        floor = ET.SubElement(worldbody, 'geom')
        floor.set('name', 'floor')
        floor.set('type', 'plane')
        floor.set('size', '50 1.5 0.1')  # 100m long, 3m wide
        floor.set('pos', '50 0 0')
        floor.set('material', 'mat_floor')
        
        # Walls (simple boxes)
        wall_left = ET.SubElement(worldbody, 'geom')
        wall_left.set('name', 'wall_left')
        wall_left.set('type', 'box')
        wall_left.set('size', '50 0.05 0.5')
        wall_left.set('pos', '50 -1.5 0.5')
        
        wall_right = ET.SubElement(worldbody, 'geom')
        wall_right.set('name', 'wall_right')
        wall_right.set('type', 'box')
        wall_right.set('size', '50 0.05 0.5')
        wall_right.set('pos', '50 1.5 0.5')
        
        # Add robot
        if robot_components['robot_body'] is not None:
            robot_body = robot_components['robot_body']
            robot_body.set('pos', '2.0 0 0.3')
            worldbody.append(robot_body)
        
        root.append(worldbody)
        
        if robot_components['actuators'] is not None:
            root.append(robot_components['actuators'])
        
        return ET.tostring(root, encoding='unicode')
    
    def _extract_robot_from_xml(self, xml_file_path):
        """Extract robot components from XML."""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        components = {
            'compiler': None, 'option': None, 'default': None,
            'asset': None, 'robot_body': None, 'actuators': None, 'visual': None
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
        """Extract corridor components from XML."""
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        components = {
            'compiler': None, 'option': None, 'default': None,
            'asset': None, 'corridor_geom': None, 'actuators': None
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
    
    def render(self):
        """Render not implemented for MJX version."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment
gym.register(
    id='RobotCorridorMJX-v0',
    entry_point='robot_corridor_env_mjx:RobotCorridorEnvMJX',
    max_episode_steps=1000,
)
