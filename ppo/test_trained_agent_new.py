"""
Test a trained PPO agent on the Robot Corridor environment.
"""
import torch
import torch.nn as nn
import numpy as np
from robot_corridor_env_new import RobotCorridorEnv
import gymnasium as gym
import mujoco
from mujoco import viewer
import time


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent (copied from train_ppo_simple.py to avoid tensorboard dependency)."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def test_agent(model_path, num_episodes=10, render=False):
    """
    Test a trained agent.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    # Create environment (no render mode for now)
    env = RobotCorridorEnv()
    
    # Create a dummy vectorized env for agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Testing agent from {model_path}")
    print(f"Device: {device}\n")
    
    episode_returns = []
    episode_lengths = []
    final_positions = []
    successes = 0
    
    if render:
        # Use MuJoCo viewer like simulation.py
        m = env.model
        d = env.data
        
        robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        
        with viewer.launch_passive(m, d) as v:
            cam = v.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = robot_body_id
            cam.azimuth = 180
            cam.elevation = -20
            cam.distance = 10
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                done = False
                episode_return = 0
                episode_length = 0
                
                print(f"\nEpisode {episode + 1} starting...")
                
                while not done and v.is_running():
                    # Get action from policy
                    with torch.no_grad():
                        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)
                        action = action.cpu().numpy()[0]
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_return += reward
                    episode_length += 1
                    
                    # Render
                    cam.trackbodyid = robot_body_id
                    v.sync()
                    time.sleep(0.01)  # 100Hz
                
                # Record statistics
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                final_x = info['x_position']
                final_positions.append(final_x)
                
                if final_x >= 100.0:
                    successes += 1
                
                print(f"Episode {episode + 1}:")
                print(f"  Return: {episode_return:.2f}")
                print(f"  Length: {episode_length}")
                print(f"  Final X: {final_x:.2f}m")
                print(f"  Termination: {info.get('termination_reason', 'truncated')}")
                
                if not v.is_running():
                    break
    else:
        # No rendering - just run episodes
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_return = 0
            episode_length = 0
            
            while not done:
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
            
            # Record statistics
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            final_x = info['x_position']
            final_positions.append(final_x)
            
            if final_x >= 100.0:
                successes += 1
            
            print(f"Episode {episode + 1}:")
            print(f"  Return: {episode_return:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Final X: {final_x:.2f}m")
            print(f"  Termination: {info.get('termination_reason', 'truncated')}")
            print()
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Final X: {np.mean(final_positions):.2f}m ± {np.std(final_positions):.2f}m")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Max Distance: {max(final_positions):.2f}m")
    print("="*50)
    
    env.close()
    
    return {
        'returns': episode_returns,
        'lengths': episode_lengths,
        'positions': final_positions,
        'success_rate': successes / num_episodes
    }


if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model (auto-finds latest if not provided)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    
    args = parser.parse_args()
    
    # Auto-find latest model if not provided
    model_path = args.model_path
    if model_path is None:
        # Search for models in models/ppo_robot_corridor_*/ppo_robot_corridor.pth
        model_files = glob.glob("models/ppo_robot_corridor_*/ppo_robot_corridor.pth")
        if not model_files:
            print("ERROR: No trained models found in models/ppo_robot_corridor_*/")
            print("Please train a model first using train_ppo_simple.py")
            exit(1)
        
        # Sort by timestamp (embedded in directory name) and take the latest
        model_files.sort(reverse=True)
        model_path = model_files[0]
        print(f"Auto-detected latest model: {model_path}\n")
    
    test_agent(model_path, args.episodes, args.render)
