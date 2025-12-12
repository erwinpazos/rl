"""
Expert controller for generating demonstrations.
Simple differential drive controller that knows how to navigate.
"""
import numpy as np
from robot_basic_env import RobotBasicEnv
import mujoco
from mujoco import viewer
import time


class ExpertController:
    """Simple expert that knows differential drive."""
    
    def __init__(self):
        self.max_speed = 15.0
        self.turn_speed = 8.0
    
    def get_action(self, obs, target_x, target_y):
        """
        Generate expert action given observation and target.
        
        Args:
            obs: [x, y, z, quat(4), vx, vy, vz, wx, wy, wz] (13 values)
            target_x, target_y: Target position
        
        Returns:
            action: [fl, fr, rl, rr] wheel torques
        """
        # Extract robot state
        robot_x, robot_y, robot_z = obs[0], obs[1], obs[2]
        quat = obs[3:7]  # w, x, y, z
        
        # Calculate robot heading from quaternion
        robot_heading = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                                 1 - 2*(quat[2]**2 + quat[3]**2))
        
        # Calculate angle to target
        target_dx = target_x - robot_x
        target_dy = target_y - robot_y
        target_angle = np.arctan2(target_dy, target_dx)
        
        # Angle difference (how much to turn)
        angle_diff = np.arctan2(np.sin(target_angle - robot_heading), 
                               np.cos(target_angle - robot_heading))
        
        # Distance to target
        distance = np.sqrt(target_dx**2 + target_dy**2)
        
        # Expert policy: simple proportional controller
        if distance < 0.5:
            # Close to target - stop
            forward_speed = 0.0
            turn_rate = 0.0
        elif abs(angle_diff) > 0.3:  # Need to turn first
            # Turn towards target
            forward_speed = 2.0  # Slow forward while turning
            turn_rate = np.clip(angle_diff * 3.0, -1.0, 1.0)  # Proportional turning
        else:
            # Go forward
            forward_speed = min(8.0, distance * 2.0)  # Speed proportional to distance
            turn_rate = np.clip(angle_diff * 1.0, -0.3, 0.3)  # Small corrections
        
        # Convert to differential drive
        base_speed = forward_speed
        turn_diff = turn_rate * self.turn_speed
        
        # Calculate wheel speeds
        left_speed = base_speed - turn_diff
        right_speed = base_speed + turn_diff
        
        # Convert to torques (simple mapping)
        action = np.array([
            left_speed,   # Front left
            right_speed,  # Front right  
            left_speed,   # Rear left
            right_speed   # Rear right
        ])
        
        # Clip to action space
        action = np.clip(action, -15.0, 15.0) / 20.0  # Normalize to [-1, 1]
        
        return action


def collect_demonstrations(num_episodes=50, render=False):
    """Collect expert demonstrations."""
    env = RobotBasicEnv()
    expert = ExpertController()
    
    demonstrations = []
    
    print(f"Collecting {num_episodes} expert demonstrations...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_data = []
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done:
            # Get expert action
            action = expert.get_action(obs, info['target_x'], info['target_y'])
            
            # Store transition
            episode_data.append({
                'obs': obs.copy(),
                'action': action.copy(),
                'target_x': info['target_x'],
                'target_y': info['target_y']
            })
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if render and episode < 3:  # Render first few episodes
                time.sleep(0.01)
        
        demonstrations.append(episode_data)
        print(f"  Episode length: {len(episode_data)} steps")
    
    env.close()
    
    # Save demonstrations
    import pickle
    with open('expert_demonstrations.pkl', 'wb') as f:
        pickle.dump(demonstrations, f)
    
    print(f"Saved {len(demonstrations)} demonstrations to expert_demonstrations.pkl")
    return demonstrations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--render", action="store_true", help="Render collection")
    args = parser.parse_args()
    
    collect_demonstrations(args.episodes, args.render)