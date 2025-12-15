"""
PPO training with pre-trained basic control model as initialization.
Uses the behavioral cloning model as a starting point for corridor navigation.
"""
import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.normal import Normal
import glob

# Import our custom environment
from robot_corridor_env_new import RobotCorridorEnv


# Simple configuration class
class Config:
    # Experiment
    exp_name = "ppo_robot_corridor_pretrained"
    seed = 1
    
    # Training
    total_timesteps = 4000000
    learning_rate = 3e-4
    num_envs = 28
    num_steps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    
    # PPO
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.05
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Computed
    batch_size = 0
    minibatch_size = 0
    num_iterations = 0


def make_env(corridor_xml="corridor_3x100.xml"):
    """Create a single environment."""
    def thunk():
        env = RobotCorridorEnv(corridor_xml=corridor_xml)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BehavioralCloningAgent(nn.Module):
    """Behavioral cloning agent architecture (matches the saved model)."""
    def __init__(self, input_dim=15, output_dim=4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)


class Agent(nn.Module):
    """PPO agent with CNN architecture that can be initialized from basic control."""
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
            # Keep spatial structure: 16 channels × 19×12 = 3648 features
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
        
        # LSTM for temporal memory (NEW!)
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(64, self.lstm_hidden_size, batch_first=True)
        
        # Initialize LSTM hidden states (will be managed during rollouts)
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
            # Create fresh LSTM states for this batch (used during optimization)
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
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features), new_lstm_state
    
    def reset_lstm_state(self, batch_size=1):
        """Reset LSTM hidden states (call at episode start)."""
        device = next(self.parameters()).device
        self.lstm_h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        self.lstm_c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (self.lstm_h, self.lstm_c)

    def initialize_from_basic_control(self, basic_model_path, device):
        """Initialize parts of the network from a pre-trained behavioral cloning model."""
        print(f"Loading behavioral cloning model from: {basic_model_path}")
        
        # Load behavioral cloning model
        bc_agent = BehavioralCloningAgent(input_dim=15, output_dim=4)
        bc_agent.load_state_dict(torch.load(basic_model_path, map_location=device))
        
        print("Initializing compatible layers...")
        
        # The behavioral cloning model outputs 4 wheel actions directly
        # We can use its layers to initialize our actor layers
        with torch.no_grad():
            # Copy the penultimate layer (64->64 in both)
            bc_penult_layer = bc_agent.network[4]  # Linear(128, 64) - index 4
            ppo_penult_layer = self.actor_mean[0]  # Our first actor layer (64->64)
            
            if bc_penult_layer.weight.shape[1] == ppo_penult_layer.weight.shape[0]:  # 64 inputs match
                # Initialize with a subset of the BC weights
                ppo_penult_layer.weight.copy_(bc_penult_layer.weight[:64, :64])  # Take 64x64 subset
                ppo_penult_layer.bias.copy_(bc_penult_layer.bias[:64])  # Take first 64 biases
                print("✓ Copied penultimate layer (partial initialization)")
            
            # Copy the final layer (64->4 in both)
            bc_final_layer = bc_agent.network[6]  # The final Linear layer (64->4)
            ppo_final_layer = self.actor_mean[2]  # Our final actor layer (64->4)
            
            if bc_final_layer.weight.shape == ppo_final_layer.weight.shape:
                ppo_final_layer.weight.copy_(bc_final_layer.weight)
                ppo_final_layer.bias.copy_(bc_final_layer.bias)
                print("✓ Copied final actor layer (wheel control mapping)")
            else:
                print(f"⚠️  Shape mismatch: BC {bc_final_layer.weight.shape} vs PPO {ppo_final_layer.weight.shape}")
            
            # Initialize actor_logstd to reasonable values for exploration
            # Start with low exploration since we have good initialization
            self.actor_logstd.fill_(-1.0)  # exp(-1) ≈ 0.37 std
            print("✓ Initialized exploration parameters (conservative)")
        
        print("✓ Behavioral cloning initialization complete!")
        print("  - Wheel control mapping initialized from behavioral cloning")
        print("  - Spatial processing (CNN) will be learned from scratch")
        print("  - Robot should start with basic locomotion skills")


def find_best_basic_model():
    """Find the best basic control model."""
    # Try different possible paths
    search_paths = [
        "../basic_control/behavioral_cloning_model.pth",
        "../basic_control/models/basic_robot_control_*/best_model.pth",
        "../basic_control/models/basic_robot_control_*/basic_control.pth", 
        "basic_control/behavioral_cloning_model.pth",
        "basic_control/models/basic_robot_control_*/best_model.pth",
        "basic_control/models/basic_robot_control_*/basic_control.pth"
    ]
    
    for pattern in search_paths:
        files = glob.glob(pattern)
        if files:
            if "*" in pattern:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"Found model using pattern: {pattern}")
            return files[0]
    
    return None


def train(corridor_xml="corridor_3x100.xml", basic_model_path=None):
    """Main training function."""
    args = Config()
    
    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    
    print("="*70)
    print("PPO TRAINING WITH PRE-TRAINED BASIC CONTROL")
    print("="*70)
    print(f"Run name: {run_name}")
    print(f"Corridor XML: {corridor_xml}")
    print(f"Basic model: {basic_model_path or 'Auto-detect'}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Parallel envs: {args.num_envs}")
    print("="*70)
    print()

    # Check if basic model exists
    if not os.path.exists(basic_model_path):
        print(f"❌ ERROR: Basic control model not found at: {basic_model_path}")
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"Absolute path would be: {os.path.abspath(basic_model_path)}")
        
        # Try to find it
        alternative = find_best_basic_model()
        if alternative:
            print(f"Found alternative model: {alternative}")
            basic_model_path = alternative
        else:
            print("\nNo basic control model found!")
            print("Please train a basic control model first or specify correct path with --basic-model")
            return
    
    print(f"Using basic model: {basic_model_path}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create environments
    envs = gym.vector.AsyncVectorEnv([make_env(corridor_xml) for _ in range(args.num_envs)])

    # Create agent and initialize from basic control
    agent = Agent(envs).to(device)
    agent.initialize_from_basic_control(basic_model_path, device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Initialize LSTM states for all environments
    lstm_states = agent.reset_lstm_state(args.num_envs)
    
    # Track statistics
    episode_returns = []
    episode_lengths = []

    print("Starting training...\n")

    for iteration in range(1, args.num_iterations + 1):
        # Collect rollouts
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action with LSTM
            with torch.no_grad():
                action, logprob, _, value, new_lstm_states = agent.get_action_and_value(next_obs, lstm_state=lstm_states)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Update LSTM states
            lstm_states = new_lstm_states

            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset LSTM states for environments that finished episodes
            if next_done.any():
                # Reset LSTM states for finished environments
                done_envs = next_done.nonzero(as_tuple=True)[0]
                if len(done_envs) > 0:
                    lstm_h, lstm_c = lstm_states
                    lstm_h[:, done_envs, :] = 0
                    lstm_c[:, done_envs, :] = 0
                    lstm_states = (lstm_h, lstm_c)

            # Log episodes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        
                        # Print every episode
                        print(f"Step {global_step:>8} | Return: {info['episode']['r']:>7.2f} | Length: {info['episode']['l']:>4}")

        # Compute advantages (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs, lstm_state=lstm_states).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Print iteration summary
        sps = int(global_step / (time.time() - start_time))
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{args.num_iterations}")
        print(f"{'='*80}")
        print(f"Steps: {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,} | Time: {elapsed_time:.1f}s")
        
        if len(episode_returns) > 0:
            recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
            recent_lengths = episode_lengths[-10:] if len(episode_lengths) >= 10 else episode_lengths
            
            print(f"\nEpisodes completed: {len(episode_returns)}")
            print(f"  Last 10 episodes:")
            print(f"    Avg Return:  {np.mean(recent_returns):>8.2f} (min: {np.min(recent_returns):>7.2f}, max: {np.max(recent_returns):>7.2f})")
            print(f"    Avg Length:  {np.mean(recent_lengths):>8.0f} (min: {np.min(recent_lengths):>7.0f}, max: {np.max(recent_lengths):>7.0f})")
            print(f"  All time:")
            print(f"    Best Return: {np.max(episode_returns):>8.2f}")
            print(f"    Avg Return:  {np.mean(episode_returns):>8.2f}")
        
        print(f"{'='*80}\n")

    # Save model
    model_dir = f"models/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ppo_robot_corridor_pretrained.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")

    # Final statistics
    if len(episode_returns) > 0:
        print("Final Statistics:")
        print(f"  Episodes completed: {len(episode_returns)}")
        print(f"  Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"  Best return: {max(episode_returns):.2f}")
        print(f"  Average length: {np.mean(episode_lengths):.2f}")

    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent with pre-trained basic control")
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml", 
                       help="Corridor XML file to use")
    parser.add_argument("--basic-model", type=str, default="../basic_control/behavioral_cloning_model.pth",
                       help="Path to pre-trained basic control model")
    args = parser.parse_args()
    
    train(corridor_xml=args.corridor, basic_model_path=args.basic_model)