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
    """PPO Agent with CNN architecture (copied from train_ppo_new_corridor.py)."""
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        # Separate robot state (6 values) + wheel contact (4 values) from grid observation (276 values)
        self.robot_state_size = 6  # x, y, z, vx, vy, vz
        self.wheel_contact_size = 4  # cell type under each wheel
        self.grid_size = obs_shape - self.robot_state_size - self.wheel_contact_size  # 23x12 = 276 grid cells
        
        # Robot state encoder (position + velocity + wheel contact)
        combined_robot_size = self.robot_state_size + self.wheel_contact_size  # 6 + 4 = 10
        self.robot_encoder = nn.Sequential(
            layer_init(nn.Linear(combined_robot_size, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.ReLU(),
        )
        
        # IMPROVED Grid encoder - preserves spatial relationships
        # No pooling to keep precise spatial information
        self.grid_conv = nn.Sequential(
            # First conv layer - detect basic patterns
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 19x12 -> 19x12
            nn.ReLU(),
            # Second conv layer - detect relationships
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 19x12 -> 19x12
            nn.ReLU(),
            # Third conv layer - higher level patterns (NO POOLING)
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 19x12 -> 19x12
            nn.ReLU(),
            # Keep spatial structure: 16 channels × 23×12 = 4416 features
            nn.Flatten(),
        )
        
        # Positional encoding - add explicit position information
        self.pos_encoding = nn.Parameter(torch.randn(1, 23, 12) * 0.1)
        
        # Spatial attention - learn which areas are important
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=1),   # 1 grid + 1 position -> 8
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),   # -> 1 attention map
            nn.Sigmoid()
        )
        
        # Grid encoder with spatial awareness
        self.grid_encoder = nn.Sequential(
            layer_init(nn.Linear(4416, 256)),  # 16 channels × 23×12 = 4416 features
            nn.ReLU(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
        )
        
        # Combined features (spatial processing)
        combined_size = 32 + 32  # robot_encoder + grid_encoder
        
        # Spatial feature combiner
        self.spatial_combiner = nn.Sequential(
            layer_init(nn.Linear(combined_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # LSTM for temporal memory
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(64, self.lstm_hidden_size, batch_first=True)
        
        # Initialize LSTM hidden states
        self.register_buffer("lstm_h", torch.zeros(1, 1, self.lstm_hidden_size))
        self.register_buffer("lstm_c", torch.zeros(1, 1, self.lstm_hidden_size))
        
        # Final processing after LSTM
        self.post_lstm = nn.Sequential(
            layer_init(nn.Linear(self.lstm_hidden_size, 64)),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        
        # Actor head (policy) - COMPATIBLE with behavioral cloning
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 64)),  # Match BC architecture
            nn.ReLU(),
            layer_init(nn.Linear(64, action_shape), std=0.01),  # 64->4 like BC
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def forward(self, x, lstm_state=None):
        batch_size = x.shape[0]
        
        # Split observation into robot state, wheel contact, and grid
        robot_state = x[:, :self.robot_state_size]  # First 6 values
        wheel_contact = x[:, self.robot_state_size:self.robot_state_size + self.wheel_contact_size]  # Next 4 values
        grid_obs = x[:, self.robot_state_size + self.wheel_contact_size:]  # Remaining 276 values
        
        # Combine robot state with wheel contact info
        combined_robot_state = torch.cat([robot_state, wheel_contact], dim=1)
        
        # Encode robot state (including wheel contact)
        robot_features = self.robot_encoder(combined_robot_state)
        
        # Reshape grid to 2D spatial format
        grid_2d = grid_obs.view(batch_size, 1, 23, 12)  # Batch x 1 x 23 x 12
        
        # Add positional encoding (robot position awareness)
        pos_encoded = grid_2d + self.pos_encoding.unsqueeze(0)  # Add position info
        
        # Combine grid with position for attention (need to add channel dimension to pos_encoding)
        pos_expanded = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Batch x 1 x 23 x 12
        grid_with_pos = torch.cat([grid_2d, pos_expanded], dim=1)  # Batch x 2 x 23 x 12
        
        # Compute spatial attention (where to focus)
        attention_map = self.spatial_attention(grid_with_pos)  # Batch x 1 x 23 x 12
        
        # Apply attention to position-encoded grid
        attended_grid = pos_encoded * attention_map
        
        # Process with CNN (preserves spatial structure)
        grid_conv_features = self.grid_conv(attended_grid)  # Batch x 4416
        
        # Final encoding
        grid_features = self.grid_encoder(grid_conv_features)
        
        # Combine spatial features
        combined = torch.cat([robot_features, grid_features], dim=1)
        spatial_features = self.spatial_combiner(combined)  # Batch x 64
        
        # LSTM for temporal processing
        if lstm_state is None:
            # Create fresh LSTM states for this batch
            device = spatial_features.device
            lstm_h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
            lstm_c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        else:
            lstm_h, lstm_c = lstm_state
        
        # Add sequence dimension for LSTM
        spatial_features_seq = spatial_features.unsqueeze(1)  # Batch x 1 x 64
        
        # LSTM forward pass
        lstm_out, (new_h, new_c) = self.lstm(spatial_features_seq, (lstm_h, lstm_c))
        
        # Remove sequence dimension
        lstm_features = lstm_out.squeeze(1)  # Batch x 128
        
        # Final processing
        features = self.post_lstm(lstm_features)  # Batch x 64
        
        return features, (new_h, new_c)

    def get_value(self, x, lstm_state=None):
        features, new_lstm_state = self.forward(x, lstm_state)
        return self.critic(features)

    def get_action_and_value(self, x, action=None, lstm_state=None):
        features, new_lstm_state = self.forward(x, lstm_state)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features), new_lstm_state
    
    def reset_lstm_state(self, batch_size=1):
        """Reset LSTM hidden states (call at episode start)."""
        device = next(self.parameters()).device
        self.lstm_h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        self.lstm_c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (self.lstm_h, self.lstm_c)


def test_agent(model_path, num_episodes=10, render=False, corridor_xml="corridor_3x100.xml"):
    """
    Test a trained agent.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        render: Whether to render the environment
        corridor_xml: Corridor XML file to use
    """
    # Create environment (no render mode for now)
    env = RobotCorridorEnv(corridor_xml=corridor_xml)
    
    # Create a dummy vectorized env for agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(dummy_env).to(device)
    
    # Load state dict but ignore LSTM buffer size mismatches
    checkpoint = torch.load(model_path, map_location=device)
    
    # Remove LSTM buffers from checkpoint (they have wrong size for testing)
    if 'lstm_h' in checkpoint:
        del checkpoint['lstm_h']
    if 'lstm_c' in checkpoint:
        del checkpoint['lstm_c']
    
    agent.load_state_dict(checkpoint, strict=False)
    agent.eval()
    
    print(f"Testing agent from {model_path}")
    print(f"Corridor XML: {corridor_xml}")
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
                
                # Reset LSTM state for new episode
                lstm_state = agent.reset_lstm_state(1)
                
                print(f"\nEpisode {episode + 1} starting...")
                
                while not done and v.is_running():
                    # Get action from policy with LSTM
                    with torch.no_grad():
                        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                        action, _, _, _, lstm_state = agent.get_action_and_value(obs_tensor, lstm_state=lstm_state)
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
            
            # Reset LSTM state for new episode
            lstm_state = agent.reset_lstm_state(1)
            
            while not done:
                # Get action from policy with LSTM
                with torch.no_grad():
                    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                    action, _, _, _, lstm_state = agent.get_action_and_value(obs_tensor, lstm_state=lstm_state)
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
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model (auto-finds latest if not provided)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml", help="Corridor XML file to use")
    
    args = parser.parse_args()
    
    # Auto-find latest model if not provided
    model_path = args.model_path
    if model_path is None:
        # Search for all PPO models (including pretrained ones)
        model_patterns = [
            "models/ppo_robot_corridor_pretrained_*/ppo_robot_corridor_pretrained.pth",
            "models/ppo_robot_corridor_*/ppo_robot_corridor.pth"
        ]
        
        all_models = []
        for pattern in model_patterns:
            all_models.extend(glob.glob(pattern))
        
        if not all_models:
            print("ERROR: No trained models found!")
            print("Searched for:")
            for pattern in model_patterns:
                print(f"  {pattern}")
            print("Please train a model first.")
            exit(1)
        
        # Sort by modification time and take the latest
        all_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = all_models[0]
        print(f"Auto-detected latest model: {model_path}\n")
    
    test_agent(model_path, args.episodes, args.render, args.corridor)
