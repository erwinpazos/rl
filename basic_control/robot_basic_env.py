"""
Basic robot control environment - Learn locomotion on flat ground.
No obstacles, just learn to move forward, turn, and control 4 wheels.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import xml.etree.ElementTree as ET


class RobotBasicEnv(gym.Env):
    """
    Basic environment for learning robot locomotion.
    
    Observation Space: 
        - Robot position (x, y, z): 3 values
        - Robot orientation (quaternion): 4 values  
        - Robot velocity (vx, vy, vz): 3 values
        - Robot angular velocity (wx, wy, wz): 3 values
        Total: 13 continuous values
    
    Action Space:
        - 4 continuous values in [-1, 1] for each wheel torque
        - Robot must learn differential steering from scratch
    
    Reward:
        - Forward movement: +1 * delta_x
        - Staying upright: +0.1 if not tilted
        - Penalty for spinning: -0.1 * |angular_velocity|
        - Target reaching: +10 for reaching random targets
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 robot_xml="four_wheels_robot.xml",
                 max_steps=2000, 
                 render_mode=None):
        super().__init__()
        
        # Load robot model with flat ground
        self.model = self._build_flat_world_model(robot_xml)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_distance_from_origin = 15.0  # Terminate if too far
        
        # Observation space: [x, y, z, quat(4), vx, vy, vz, wx, wy, wz]
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(13,),
            dtype=np.float32
        )
        
        # Action space: 4 wheel torques in [-1, 1] - raw control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.previous_x = 0.0
        self.target_x = 10.0  # Random target to reach
        self.target_y = 0.0
        
        # Rendering
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set random starting position and orientation
        if seed is not None:
            np.random.seed(seed)
        
        # Random start position (small variations)
        self.data.qpos[0] = np.random.uniform(-1, 1)  # x
        self.data.qpos[1] = np.random.uniform(-1, 1)  # y
        self.data.qpos[2] = 0.35  # z (fixed height)
        
        # Random start orientation (small variations)
        angle = np.random.uniform(-0.2, 0.2)  # ±11 degrees
        self.data.qpos[3] = np.cos(angle/2)  # quat w
        self.data.qpos[4] = 0  # quat x
        self.data.qpos[5] = 0  # quat y  
        self.data.qpos[6] = np.sin(angle/2)  # quat z
        
        # Curriculum learning: start easy, get harder
        robot_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        robot_y = self.data.qpos[1] if len(self.data.qpos) > 0 else 0.0
        
        # Curriculum learning: aggressive progression to force turning
        if not hasattr(self, 'total_steps'):
            self.total_steps = 0
            
        if self.total_steps < 200000:
            # Phase 1 (0-200k): Only forward targets (learn basic movement)
            angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30°
            distance = np.random.uniform(3, 6)  # Close targets
        elif self.total_steps < 400000:
            # Phase 2 (200k-400k): Side targets (force turning)
            angle = np.random.uniform(-np.pi/2, np.pi/2)  # ±90°
            distance = np.random.uniform(4, 8)
        else:
            # Phase 3 (400k+): Full 360° (complete navigation)
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(5, 12)
            
        self.target_x = robot_x + distance * np.cos(angle)
        self.target_y = robot_y + distance * np.sin(angle)
        
        # Update target marker position
        self._update_target_marker()
        
        # Forward simulation to initialize
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking variables
        self.current_step = 0
        if not hasattr(self, 'total_steps'):
            self.total_steps = 0
        self.previous_x = self.data.qpos[0]
        self.last_action = None
        
        # Get initial observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        # Apply raw wheel torques (no preprocessing - robot must learn)
        action = np.clip(action, -1.0, 1.0) * 20.0  # Scale to [-20, 20] Nm
        self.data.ctrl[:] = action
        
        # Store action for reward calculation
        self.last_action = action.copy()
        
        # Step simulation
        for _ in range(5):  # 5 substeps for stability
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        self.total_steps += 1  # Track total steps for curriculum
        
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
        """Get current observation."""
        # Robot state: position, orientation, velocities
        pos = self.data.qpos[:3]  # x, y, z
        quat = self.data.qpos[3:7]  # quaternion (w, x, y, z)
        vel = self.data.qvel[:3]  # vx, vy, vz
        angvel = self.data.qvel[3:6]  # wx, wy, wz
        
        # Combine all observations
        obs = np.concatenate([pos, quat, vel, angvel]).astype(np.float32)
        return obs
    
    def _get_info(self):
        """Get additional information."""
        return {
            'x_position': float(self.data.qpos[0]),
            'y_position': float(self.data.qpos[1]),
            'z_position': float(self.data.qpos[2]),
            'target_x': self.target_x,
            'target_y': self.target_y,
            'distance_to_target': float(np.sqrt((self.data.qpos[0] - self.target_x)**2 + 
                                              (self.data.qpos[1] - self.target_y)**2)),
            'step': self.current_step
        }
    
    def _compute_reward(self):
        """Compute reward for basic locomotion learning."""
        robot_x = self.data.qpos[0]
        robot_y = self.data.qpos[1]
        robot_z = self.data.qpos[2]
        
        # Get orientation (check if upright) - simplified approach
        quat = self.data.qpos[3:7]  # w, x, y, z
        # Simple tilt check using quaternion directly
        # For small tilts, quat[1] and quat[2] (x,y components) indicate tilt
        tilt_magnitude = np.sqrt(quat[1]**2 + quat[2]**2)  # x,y components
        tilt_angle = 2 * np.arcsin(np.clip(tilt_magnitude, 0, 1))  # Convert to angle
        
        # Get velocities
        vx, vy, vz = self.data.qvel[:3]
        wx, wy, wz = self.data.qvel[3:6]
        
        terminated = False
        info = {}
        
        # Terminal conditions
        if robot_z < 0.1:
            # Fell over
            reward = -10.0
            terminated = True
            info['termination_reason'] = 'fell'
            print(f"[TERM] FELL at step {self.current_step}")
        elif tilt_angle > np.pi/3:  # More than 60 degrees tilt
            # Tipped over
            reward = -5.0
            terminated = True
            info['termination_reason'] = 'tipped'
            print(f"[TERM] TIPPED at step {self.current_step}: tilt={np.degrees(tilt_angle):.1f}°")
        elif np.sqrt(robot_x**2 + robot_y**2) > self.max_distance_from_origin:
            # Too far from origin - prevent infinite wandering
            reward = -2.0
            terminated = True
            info['termination_reason'] = 'too_far'
            print(f"[TERM] TOO FAR at step {self.current_step}: distance={np.sqrt(robot_x**2 + robot_y**2):.1f}m")
        else:
            # Reward components for learning locomotion
            
            # 1. Movement reward (any movement is good initially)
            delta_x = robot_x - self.previous_x
            self.previous_x = robot_x
            movement_speed = np.sqrt(vx**2 + vy**2)
            movement_reward = movement_speed * 0.5  # Reward any movement
            
            # 1.5. Orientation reward (reward for facing the target)
            # Calculate angle to target
            target_dx = self.target_x - robot_x
            target_dy = self.target_y - robot_y
            target_angle = np.arctan2(target_dy, target_dx)
            
            # Get robot's current heading from quaternion
            quat = self.data.qpos[3:7]  # w, x, y, z
            # Simple heading extraction: robot faces +X initially
            robot_heading = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                                     1 - 2*(quat[2]**2 + quat[3]**2))
            
            # Angle difference (how well aligned with target)
            angle_diff = abs(np.arctan2(np.sin(target_angle - robot_heading), 
                                      np.cos(target_angle - robot_heading)))
            
            # Strong penalty for not facing target (encourages turning)
            orientation_penalty = -angle_diff * 2.0  # Linear penalty for misalignment
            orientation_reward = orientation_penalty  # Rename for clarity
            
            # 2. Target approaching reward (shaped reward)
            dist_to_target = np.sqrt((robot_x - self.target_x)**2 + (robot_y - self.target_y)**2)
            prev_dist = getattr(self, 'prev_target_dist', dist_to_target)
            self.prev_target_dist = dist_to_target
            
            # Dynamic target radius (can be set externally)
            target_radius = getattr(self, 'target_radius', 1.0)
            if dist_to_target < target_radius:  # Reached target
                target_reward = 100.0  # Big reward!
                # Set new random target (curriculum-based)
                if self.total_steps < 500000:
                    angle = np.random.uniform(-np.pi/4, np.pi/4)
                    distance = np.random.uniform(3, 8)
                elif self.total_steps < 1000000:
                    angle = np.random.uniform(-np.pi/2, np.pi/2)
                    distance = np.random.uniform(4, 10)
                else:
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(5, 15)
                self.target_x = robot_x + distance * np.cos(angle)
                self.target_y = robot_y + distance * np.sin(angle)
                self._update_target_marker()
                print(f"[TARGET] Reached! New target: ({self.target_x:.1f}, {self.target_y:.1f})")
            else:
                # Reward for getting closer (shaped reward)
                approach_reward = (prev_dist - dist_to_target) * 10.0
                target_reward = approach_reward - 0.001 * dist_to_target
            
            # 3. Stability reward (staying upright)
            stability_reward = 1.0 * (1.0 - tilt_angle / (np.pi/2))  # Max 1.0 when perfectly upright
            
            # 4. Turning behavior reward
            turning_reward = 0.0
            if angle_diff > np.pi/6:  # Need to turn (>30°)
                # Reward angular velocity in the right direction
                target_turn_direction = np.sign(np.arctan2(np.sin(target_angle - robot_heading), 
                                                         np.cos(target_angle - robot_heading)))
                actual_turn_rate = wz  # Z-axis angular velocity
                
                # Reward turning in the correct direction
                if target_turn_direction * actual_turn_rate > 0:
                    turning_reward = abs(actual_turn_rate) * 1.0  # Reward correct turning
                else:
                    turning_reward = -abs(actual_turn_rate) * 0.5  # Penalty for wrong direction
            
            # 5. No spin penalty - allow free rotation for learning to turn
            spin_penalty = 0.0
            
            # 6. Time penalty (increases with difficulty)
            time_multiplier = getattr(self, 'time_penalty_multiplier', 1.0)
            time_penalty = -0.001 * time_multiplier
            
            # Total reward
            reward = movement_reward + target_reward + stability_reward + spin_penalty + time_penalty + orientation_reward + turning_reward
            
            info['termination_reason'] = None
            info['movement_reward'] = movement_reward
            info['target_reward'] = target_reward
            info['stability_reward'] = stability_reward
            info['orientation_reward'] = orientation_reward
            info['turning_reward'] = turning_reward
            info['tilt_angle'] = np.degrees(tilt_angle)
            info['distance_to_target'] = dist_to_target
            info['angle_to_target'] = np.degrees(angle_diff)
        
        return reward, terminated, info
    
    def set_difficulty(self, target_radius=1.0, time_pressure=False):
        """Set curriculum difficulty parameters."""
        self.target_radius = target_radius
        self.time_pressure = time_pressure
        if time_pressure:
            # Increase time penalty for efficiency
            self.time_penalty_multiplier = 10.0
        else:
            self.time_penalty_multiplier = 1.0
        
        # Update visual marker size
        self._update_target_marker()
    
    def _update_target_marker(self):
        """Update the visual target marker position and visibility."""
        try:
            target_radius = getattr(self, 'target_radius', 1.0)
            
            # Determine which marker to show based on target radius
            if target_radius >= 0.9:
                active_marker = 'target_large'
                inactive_markers = ['target_medium', 'target_small']
            elif target_radius >= 0.6:
                active_marker = 'target_medium'
                inactive_markers = ['target_large', 'target_small']
            else:
                active_marker = 'target_small'
                inactive_markers = ['target_large', 'target_medium']
            
            # Update position of active marker
            active_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, active_marker)
            if active_body_id >= 0:
                self.model.body_pos[active_body_id] = [self.target_x, self.target_y, 0.1]
            
            # Hide inactive markers (move them far away)
            for inactive_marker in inactive_markers:
                inactive_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, inactive_marker)
                if inactive_body_id >= 0:
                    self.model.body_pos[inactive_body_id] = [1000, 1000, -10]  # Hide far away
                    
        except:
            # If target bodies don't exist, ignore silently
            pass
    
    def _build_flat_world_model(self, robot_xml):
        """Build a simple flat world with just the robot."""
        # Load robot XML
        robot_tree = ET.parse(robot_xml)
        robot_root = robot_tree.getroot()
        
        # Create new world with flat ground
        world_root = ET.Element('mujoco')
        world_root.set('model', 'robot_basic_control')
        
        # Add compiler settings
        compiler = ET.Element('compiler')
        compiler.set('angle', 'degree')
        compiler.set('coordinate', 'local')
        world_root.append(compiler)
        
        # Add physics settings
        option = ET.Element('option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        world_root.append(option)
        
        # Add size settings
        size = ET.Element('size')
        size.set('njmax', '1000')
        size.set('nconmax', '500')
        world_root.append(size)
        
        # Copy assets from robot
        for child in robot_root:
            if child.tag == 'asset':
                world_root.append(child)
            elif child.tag == 'default':
                world_root.append(child)
            elif child.tag == 'visual':
                world_root.append(child)
        
        # Create worldbody with flat ground
        worldbody = ET.Element('worldbody')
        
        # Add lighting
        light = ET.Element('light')
        light.set('directional', 'true')
        light.set('ambient', '0.2 0.2 0.2')
        light.set('diffuse', '0.8 0.8 0.8')
        light.set('specular', '0.3 0.3 0.3')
        light.set('castshadow', 'false')
        light.set('pos', '0 0 4')
        light.set('dir', '0 0 -1')
        worldbody.append(light)
        
        # Add flat ground plane
        ground = ET.Element('geom')
        ground.set('name', 'ground')
        ground.set('type', 'plane')
        ground.set('size', '50 50 0.1')
        ground.set('rgba', '0.8 0.8 0.8 1')
        ground.set('friction', '1 0.1 0.1')
        worldbody.append(ground)
        
        # Add multiple target markers for different difficulty levels
        # Large target (easy)
        target_large = ET.Element('body')
        target_large.set('name', 'target_large')
        target_large.set('pos', '10 0 0.1')
        
        geom_large = ET.Element('geom')
        geom_large.set('name', 'target_marker_large')
        geom_large.set('type', 'cylinder')
        geom_large.set('size', '0.5 0.02')  # 0.5m radius = 1.0m diameter
        geom_large.set('rgba', '0 1 0 0.8')  # Green
        geom_large.set('contype', '0')
        geom_large.set('conaffinity', '0')
        target_large.append(geom_large)
        worldbody.append(target_large)
        
        # Medium target
        target_medium = ET.Element('body')
        target_medium.set('name', 'target_medium')
        target_medium.set('pos', '10 0 0.1')
        
        geom_medium = ET.Element('geom')
        geom_medium.set('name', 'target_marker_medium')
        geom_medium.set('type', 'cylinder')
        geom_medium.set('size', '0.35 0.02')  # 0.35m radius = 0.7m diameter
        geom_medium.set('rgba', '1 1 0 0.8')  # Yellow
        geom_medium.set('contype', '0')
        geom_medium.set('conaffinity', '0')
        target_medium.append(geom_medium)
        worldbody.append(target_medium)
        
        # Small target (hard)
        target_small = ET.Element('body')
        target_small.set('name', 'target_small')
        target_small.set('pos', '10 0 0.1')
        
        geom_small = ET.Element('geom')
        geom_small.set('name', 'target_marker_small')
        geom_small.set('type', 'cylinder')
        geom_small.set('size', '0.2 0.02')  # 0.2m radius = 0.4m diameter
        geom_small.set('rgba', '1 0 0 0.8')  # Red
        geom_small.set('contype', '0')
        geom_small.set('conaffinity', '0')
        target_small.append(geom_small)
        worldbody.append(target_small)
        
        # Add robot body from original XML
        for child in robot_root:
            if child.tag == 'worldbody':
                for body in child:
                    if body.get('name') == 'robot':
                        # Set robot at proper height
                        body.set('pos', '0 0 0.35')
                        worldbody.append(body)
        
        world_root.append(worldbody)
        
        # Add actuators from robot
        for child in robot_root:
            if child.tag == 'actuator':
                world_root.append(child)
        
        # Convert to XML string and create model
        xml_string = ET.tostring(world_root, encoding='unicode')
        model = mujoco.MjModel.from_xml_string(xml_string)
        
        return model
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, camera="tracking")
            return self.renderer.render()
        elif self.render_mode == "human":
            self.renderer.update_scene(self.data, camera="tracking")
            return self.renderer.render()
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            self.renderer.close()


# Register the environment
gym.register(
    id='RobotBasic-v0',
    entry_point='robot_basic_env:RobotBasicEnv',
    max_episode_steps=2000,
)