# Question 14 - CORRECTED VERSION with Floor Observation
# This version integrates the floor observation from Question 7

import numpy as np
import mujoco

# Corridor parameters (from Question 7)
CORRIDOR_LENGTH = 100.0
CORRIDOR_WIDTH = 3.0
CELL_WIDTH = 0.5

class CorridorEnv:
    """
    Canonical RL environment wrapper for MuJoCo corridor.
    CORRECTED VERSION: Includes floor observation from Question 7.
    """
    
    def __init__(self, model_obj, data_obj, cell_map_semantic, max_steps=1000):
        """Initialize the environment.
        
        Args:
            model_obj: Existing MuJoCo model object
            data_obj: Existing MuJoCo data object
            cell_map_semantic: Dictionary mapping (row, col) -> cell_type from Question 7
            max_steps: Maximum steps before truncation
        """
        # Use existing MuJoCo model and data
        self.model = model_obj
        self.data = data_obj
        
        # Store cell map from Question 7
        self.cell_map_semantic = cell_map_semantic
        
        # Corridor parameters
        self.corridor_length = CORRIDOR_LENGTH
        self.corridor_width = CORRIDOR_WIDTH
        self.cell_width = CELL_WIDTH
        
        # Floor observation parameters
        self.n_rows_ahead = 2      # Look 2 rows ahead
        self.n_rows_behind = 1     # Look 1 row behind
        self.n_cols = 6            # 6 cells per row (3m / 0.5m)
        
        # Store max_steps
        self.max_steps = max_steps
        
        # Initialize step counter
        self.current_step = 0
        
        # Track previous position for reward calculation
        self.previous_x = 0.0
    
    def reset(self, seed=None):
        """Reset environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            initial_state: Initial observation (numpy array)
                - Shape: (6 + n_observation_cells,)
                - [x, y, z, vx, vy, vz, floor_cell_0, floor_cell_1, ...]
            info: Dictionary with initial information
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset previous position (after reset, robot is at starting position)
        self.previous_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        
        # Get initial state (position + velocity + floor observation)
        robot_x = self.data.qpos[0]
        robot_y = self.data.qpos[1]
        
        # Position and velocity
        pos_vel = np.concatenate([self.data.qpos[:3], self.data.qvel[:3]])
        
        # Floor observation
        floor_obs = self._get_floor_observation(robot_x, robot_y)
        floor_flat = floor_obs.flatten()
        
        # Combine
        initial_state = np.concatenate([pos_vel, floor_flat])
        
        # Create info dict
        info = {
            'position': (self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]),
            'step': 0,
            'floor_observation_shape': floor_obs.shape
        }
        
        return initial_state, info
    
    def step(self, action):
        """Take one step in the environment.
        
        Args:
            action: Numpy array of wheel controls [4]
            
        Returns:
            next_state: Next observation (numpy array)
                - Shape: (6 + n_observation_cells,)
                - [x, y, z, vx, vy, vz, floor_cell_0, floor_cell_1, ...]
            reward: Immediate reward (float)
            terminated: Whether episode ended naturally (bool)
            truncated: Whether episode was cut short (bool)
            info: Dictionary with diagnostic information
        """
        # Apply action to MuJoCo controls
        self.data.ctrl[:] = action
        
        # Step MuJoCo simulation (10 substeps for smoother physics)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Increment step counter
        self.current_step += 1
        
        # Get robot position
        robot_x = self.data.qpos[0]
        robot_y = self.data.qpos[1]
        robot_z = self.data.qpos[2]
        
        # Position and velocity
        pos_vel = np.concatenate([self.data.qpos[:3], self.data.qvel[:3]])
        
        # Floor observation
        floor_obs = self._get_floor_observation(robot_x, robot_y)
        floor_flat = floor_obs.flatten()
        
        # Combine
        next_state = np.concatenate([pos_vel, floor_flat])
        
        # Compute reward using reward_p2 logic
        # Terminal rewards
        if robot_x >= self.corridor_length:
            reward = +100.0
            terminated = True
            reason = "SUCCESS: Reached goal!"
        elif robot_z < 0.1 and robot_x > 0:
            reward = -100.0
            terminated = True
            reason = "FAILURE: Fell in hole!"
        elif robot_x < -1.0:
            reward = -50.0
            terminated = True
            reason = "FAILURE: Went backward!"
        else:
            # Progress reward
            delta_x = robot_x - self.previous_x
            self.previous_x = robot_x
            reward = delta_x * 10.0
            terminated = False
            reason = None
        
        # Check if truncated (max_steps reached)
        truncated = self.current_step >= self.max_steps
        
        # Build info dict
        info = {
            'position': (robot_x, robot_y, robot_z),
            'velocity': (self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]),
            'step': self.current_step,
            'termination_reason': reason,
            'floor_observation': floor_obs  # Include for debugging
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_floor_observation(self, robot_x, robot_y):
        """
        Get floor cell observations around the robot (from Question 7).
        
        Args:
            robot_x: Robot's x position
            robot_y: Robot's y position
        
        Returns:
            2D numpy array of shape (n_rows_behind + 1 + n_rows_ahead, n_cols)
            with cell types: 0=flat, 1=bump, 2=hole
        """
        # Calculate robot's cell position
        robot_row = int(robot_x / self.cell_width)
        robot_col = int((robot_y + self.corridor_width/2) / self.cell_width)
        
        # Total number of rows in observation
        total_rows = self.n_rows_behind + 1 + self.n_rows_ahead
        
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # This assumes you have already:
    # 1. Loaded the model and data (from earlier questions)
    # 2. Built the cell_map_semantic (from Question 7)
    
    print("=== Testing CORRECTED Canonical RL API (with floor observation) ===\n")
    
    # Create environment instance
    # env = CorridorEnv(model, data, cell_map_semantic, max_steps=500)
    
    # Random agent
    def random_agent():
        """Generate random action in valid range [-1, 1] for each wheel."""
        return np.random.uniform(-1.0, 1.0, size=4)
    
    # Run episodes
    # for episode in range(5):
    #     state, info = env.reset()
    #     print(f"Episode {episode + 1}:")
    #     print(f"  Initial state shape: {state.shape}")
    #     print(f"  Floor observation shape: {info['floor_observation_shape']}")
    #     
    #     done = False
    #     episode_reward = 0
    #     steps = 0
    #     
    #     while not done:
    #         action = random_agent()
    #         next_state, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         state = next_state
    #         episode_reward += reward
    #         steps += 1
    #     
    #     final_x = info['position'][0]
    #     reason = info['termination_reason'] if terminated else 'truncated'
    #     print(f"  Total Reward: {episode_reward:.2f}")
    #     print(f"  Steps: {steps}")
    #     print(f"  Final Position: x={final_x:.2f}m")
    #     print(f"  Termination: {reason}")
    #     print()
    
    print("\n✓ CORRECTED Canonical RL API implementation complete!")
    print("\nKey differences from original:")
    print("  - Observation now includes floor grid (4 rows × 6 cols = 24 cells)")
    print("  - Total observation size: 6 (pos+vel) + 24 (floor) = 30 values")
    print("  - Robot can now 'see' obstacles ahead and plan accordingly")
    print("  - This should significantly improve learning performance!")
