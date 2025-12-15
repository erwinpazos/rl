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
    
    def get_action(self, obs, target_x, target_y, target_radius=1.0, time_pressure=False):
        """
        Generate expert action given observation and target.
        
        Args:
            obs: [x, y, z, quat(4), vx, vy, vz, wx, wy, wz] (13 values)
            target_x, target_y: Target position
            target_radius: How close to get to target (curriculum)
            time_pressure: Whether to prioritize speed over precision
        
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
        
        # Expert policy: adaptive based on curriculum
        if distance < target_radius:
            # Close enough to target - stop or move to next
            forward_speed = 0.0
            turn_rate = 0.0
        elif abs(angle_diff) > (0.2 if time_pressure else 0.3):  # Need to turn first
            # Turn towards target (faster turning under time pressure)
            turn_gain = 4.0 if time_pressure else 3.0
            forward_speed = (3.0 if time_pressure else 2.0)  # Faster while turning
            turn_rate = np.clip(angle_diff * turn_gain, -1.0, 1.0)
        else:
            # Go forward (speed based on time pressure and distance)
            if time_pressure:
                # Fast and aggressive
                max_speed = min(12.0, distance * 3.0)
                forward_speed = max(6.0, max_speed)  # Minimum speed for efficiency
            else:
                # Careful and precise
                forward_speed = min(8.0, distance * 2.0)
            
            # Turning corrections (tighter under time pressure)
            turn_gain = 1.5 if time_pressure else 1.0
            max_turn = 0.5 if time_pressure else 0.3
            turn_rate = np.clip(angle_diff * turn_gain, -max_turn, max_turn)
        
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


def collect_demonstrations(num_episodes=150, render=False, curriculum=True):
    """Collect expert demonstrations with curriculum learning."""
    env = RobotBasicEnv()
    expert = ExpertController()
    
    demonstrations = []
    
    print(f"Collecting {num_episodes} expert demonstrations with curriculum...")
    
    for episode in range(num_episodes):
        # Curriculum: progressively smaller targets and faster movement
        if curriculum:
            progress = episode / num_episodes
            
            # Phase 1 (0-33%): Large targets, slow movement
            if progress < 0.33:
                target_radius = 1.0  # 1m radius
                speed_multiplier = 0.7  # Slow
                time_pressure = False
                phase = "EASY"
            # Phase 2 (33-66%): Medium targets, normal speed
            elif progress < 0.66:
                target_radius = 0.7  # 0.7m radius
                speed_multiplier = 1.0  # Normal
                time_pressure = False
                phase = "MEDIUM"
            # Phase 3 (66-100%): Small targets, fast movement, time pressure
            else:
                target_radius = 0.4  # 0.4m radius
                speed_multiplier = 1.5  # Fast
                time_pressure = True
                phase = "HARD"
        else:
            target_radius = 1.0
            speed_multiplier = 1.0
            time_pressure = False
            phase = "STANDARD"
        
        # Update expert and environment for this phase
        expert.max_speed = 15.0 * speed_multiplier
        expert.turn_speed = 8.0 * speed_multiplier
        
        obs, info = env.reset()
        done = False
        episode_data = []
        
        print(f"Episode {episode + 1}/{num_episodes} [{phase}] - Target radius: {target_radius:.1f}m, Speed: {speed_multiplier:.1f}x")
        
        if render and episode < 5:  # Render first 5 episodes
            # Use MuJoCo viewer for rendering
            import mujoco
            from mujoco import viewer
            
            m = env.model
            d = env.data
            
            robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            
            with viewer.launch_passive(m, d) as v:
                cam = v.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = robot_body_id
                cam.azimuth = 180
                cam.elevation = -20
                cam.distance = 15
                
                print(f"  Target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                
                while not done and v.is_running():
                    # Get expert action with curriculum parameters
                    action = expert.get_action(obs, info['target_x'], info['target_y'], 
                                             target_radius=target_radius, 
                                             time_pressure=time_pressure)
                    
                    # Store transition with curriculum info
                    episode_data.append({
                        'obs': obs.copy(),
                        'action': action.copy(),
                        'target_x': info['target_x'],
                        'target_y': info['target_y'],
                        'target_radius': target_radius,
                        'speed_multiplier': speed_multiplier,
                        'time_pressure': time_pressure,
                        'phase': phase
                    })
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Check for target reached
                    if reward > 50:
                        print(f"    Target reached! New target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                    
                    # Render
                    cam.trackbodyid = robot_body_id
                    v.sync()
                    time.sleep(0.02)  # 50Hz for smooth viewing
                    
                    if not v.is_running():
                        break
        else:
            # No rendering - just collect data
            while not done:
                # Get expert action with curriculum parameters
                action = expert.get_action(obs, info['target_x'], info['target_y'], 
                                         target_radius=target_radius, 
                                         time_pressure=time_pressure)
                
                # Store transition with curriculum info
                episode_data.append({
                    'obs': obs.copy(),
                    'action': action.copy(),
                    'target_x': info['target_x'],
                    'target_y': info['target_y'],
                    'target_radius': target_radius,
                    'speed_multiplier': speed_multiplier,
                    'time_pressure': time_pressure,
                    'phase': phase
                })
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        
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
    parser.add_argument("--episodes", type=int, default=150, help="Number of episodes to collect")
    parser.add_argument("--render", action="store_true", help="Render collection")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = parser.parse_args()
    
    collect_demonstrations(args.episodes, args.render, curriculum=not args.no_curriculum)