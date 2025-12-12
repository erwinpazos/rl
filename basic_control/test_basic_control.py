"""
Test a trained basic control agent.
"""
import torch
import torch.nn as nn
import numpy as np
from robot_basic_env import RobotBasicEnv
import gymnasium as gym
import mujoco
from mujoco import viewer
import time
import argparse
import glob


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BasicControlAgent(nn.Module):
    """Agent for basic robot control (copied from train_basic_control.py)."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()  # 13 values
        action_shape = np.prod(envs.single_action_space.shape)  # 4 wheel torques
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # Value function (critic)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        
        # Policy (actor) - outputs mean for each wheel
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, action_shape), std=0.01),
        )
        
        # Learnable log standard deviation for exploration
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def get_value(self, x):
        features = self.feature_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


def test_basic_control(model_path, num_episodes=10, render=False):
    """
    Test a trained basic control agent.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment
    """
    # Create environment
    env = RobotBasicEnv()
    
    # Create a dummy vectorized env for agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = BasicControlAgent(dummy_env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Testing basic control agent from {model_path}")
    print(f"Device: {device}\n")
    
    episode_returns = []
    episode_lengths = []
    targets_reached = []
    
    if render:
        # Use MuJoCo viewer
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
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                done = False
                episode_return = 0
                episode_length = 0
                targets_hit = 0
                
                print(f"\nEpisode {episode + 1} starting...")
                print(f"Target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                
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
                    
                    # Check if target was reached (high reward spike)
                    if reward > 15:
                        targets_hit += 1
                        print(f"  Target reached! New target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
                    
                    # Render
                    cam.trackbodyid = robot_body_id
                    v.sync()
                    time.sleep(0.01)  # 100Hz
                
                # Record statistics
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                targets_reached.append(targets_hit)
                
                print(f"Episode {episode + 1}:")
                print(f"  Return: {episode_return:.2f}")
                print(f"  Length: {episode_length}")
                print(f"  Targets Hit: {targets_hit}")
                print(f"  Final Position: ({info['x_position']:.2f}, {info['y_position']:.2f})")
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
            targets_hit = 0
            
            print(f"\nEpisode {episode + 1} starting...")
            print(f"Target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
            
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
                
                # Check if target was reached
                if reward > 15:
                    targets_hit += 1
                    print(f"  Target reached! New target: ({info['target_x']:.1f}, {info['target_y']:.1f})")
            
            # Record statistics
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            targets_reached.append(targets_hit)
            
            print(f"Episode {episode + 1}:")
            print(f"  Return: {episode_return:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Targets Hit: {targets_hit}")
            print(f"  Final Position: ({info['x_position']:.2f}, {info['y_position']:.2f})")
            print(f"  Termination: {info.get('termination_reason', 'truncated')}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("BASIC CONTROL TEST SUMMARY")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Total Targets Hit: {sum(targets_reached)}")
    print(f"Avg Targets per Episode: {np.mean(targets_reached):.2f}")
    print(f"Success Rate: {100 * sum(t > 0 for t in targets_reached) / num_episodes:.1f}%")
    print("="*50)
    
    env.close()
    
    return {
        'returns': episode_returns,
        'lengths': episode_lengths,
        'targets': targets_reached,
        'success_rate': sum(t > 0 for t in targets_reached) / num_episodes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model (auto-finds latest if not provided)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    
    args = parser.parse_args()
    
    # Auto-find best model if not provided
    model_path = args.model_path
    if model_path is None:
        # First try to find best model
        best_models = glob.glob("models/basic_robot_control_*/best_model.pth")
        if best_models:
            best_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            model_path = best_models[0]
            print(f"Auto-detected BEST model: {model_path}\n")
        else:
            # Fallback to regular models
            model_files = glob.glob("models/basic_robot_control_*/basic_control.pth")
            if not model_files:
                print("ERROR: No trained models found in models/basic_robot_control_*/")
                print("Please train a model first using train_basic_control.py")
                exit(1)
            
            model_files.sort(reverse=True)
            model_path = model_files[0]
            print(f"Auto-detected latest model: {model_path}\n")
    
    test_basic_control(model_path, args.episodes, args.render)