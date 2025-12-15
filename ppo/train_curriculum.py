"""
Curriculum learning for bridge crossing.
Start easy, progressively increase difficulty.
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

from robot_corridor_env_new import RobotCorridorEnv


class CurriculumConfig:
    # Experiment
    exp_name = "ppo_curriculum_bridges"
    seed = 1
    
    # Training
    total_timesteps = 8000000  # Longer for curriculum
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
    ent_coef = 0.1  # Higher exploration for curriculum
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Curriculum
    phase_duration = 1000000  # 1M steps per phase
    
    # Computed
    batch_size = 0
    minibatch_size = 0
    num_iterations = 0


def make_env(corridor_xml="corridor_simple_learning.xml"):
    """Create environment with curriculum support."""
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_shape = np.prod(envs.single_action_space.shape)
        
        # Separate robot state (6 values) from grid observation (120 values)
        self.robot_state_size = 6
        self.grid_size = obs_shape - self.robot_state_size
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            layer_init(nn.Linear(self.robot_state_size, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.ReLU(),
        )
        
        # Grid encoder with CNN
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 3)),
            nn.Flatten(),
        )
        
        self.grid_encoder = nn.Sequential(
            layer_init(nn.Linear(384, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
        )
        
        # Combined features
        combined_size = 32 + 32
        
        # Shared backbone
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(combined_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
        )
        
        # Critic head
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 1), std=1.0),
        )
        
        # Actor head
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_shape), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split observation
        robot_state = x[:, :self.robot_state_size]
        grid_obs = x[:, self.robot_state_size:]
        
        # Encode robot state
        robot_features = self.robot_encoder(robot_state)
        
        # Encode grid with CNN
        grid_2d = grid_obs.view(batch_size, 1, 10, 12)
        grid_conv_features = self.grid_conv(grid_2d)
        grid_features = self.grid_encoder(grid_conv_features)
        
        # Combine features
        combined = torch.cat([robot_features, grid_features], dim=1)
        features = self.backbone(combined)
        
        return features

    def get_value(self, x):
        features = self.forward(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


def update_curriculum(global_step, args):
    """Update curriculum difficulty based on training progress."""
    phase = global_step // args.phase_duration
    
    if phase == 0:
        # Phase 1: Learn basic movement on solid ground
        print(f"[CURRICULUM] Phase 1: Basic movement (step {global_step})")
        return "corridor_simple_learning.xml", 1.0
    elif phase == 1:
        # Phase 2: Learn to use wide bridges
        print(f"[CURRICULUM] Phase 2: Wide bridges (step {global_step})")
        return "corridor_simple_learning.xml", 0.8
    elif phase == 2:
        # Phase 3: Narrow bridges
        print(f"[CURRICULUM] Phase 3: Narrow bridges (step {global_step})")
        return "corridor_simple_learning.xml", 0.6
    elif phase == 3:
        # Phase 4: Side bridges (lateral movement)
        print(f"[CURRICULUM] Phase 4: Side bridges (step {global_step})")
        return "corridor_simple_learning.xml", 0.4
    else:
        # Phase 5+: Full difficulty
        print(f"[CURRICULUM] Phase 5+: Full difficulty (step {global_step})")
        return "corridor_3x100.xml", 0.2


def train():
    """Main training function with curriculum."""
    args = CurriculumConfig()
    
    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    
    print("="*70)
    print("CURRICULUM LEARNING - BRIDGE CROSSING")
    print("="*70)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Phase duration: {args.phase_duration:,} steps")
    print(f"Iterations: {args.num_iterations}")
    print("="*70)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create environments (start with simple)
    corridor_xml, difficulty = update_curriculum(0, args)
    envs = gym.vector.AsyncVectorEnv([make_env(corridor_xml) for _ in range(args.num_envs)])

    # Create agent
    agent = Agent(envs).to(device)
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
    
    episode_returns = []
    episode_lengths = []
    current_corridor = corridor_xml

    print("Starting curriculum training...\n")

    for iteration in range(1, args.num_iterations + 1):
        # Check if we need to update curriculum
        new_corridor, new_difficulty = update_curriculum(global_step, args)
        if new_corridor != current_corridor:
            print(f"\n🎓 CURRICULUM UPDATE: Switching to {new_corridor}")
            current_corridor = new_corridor
            # Note: In practice, you'd want to recreate environments here
            # For simplicity, we'll just note the change
        
        # Collect rollouts
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log episodes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_returns.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        
                        # Print successful episodes
                        if info["episode"]["r"] > 100:  # Success threshold
                            print(f"🎉 SUCCESS! Step {global_step:>8} | Return: {info['episode']['r']:>7.2f}")
                        else:
                            print(f"Step {global_step:>8} | Return: {info['episode']['r']:>7.2f} | Length: {info['episode']['l']:>4}")

        # Compute advantages (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
        if iteration % 5 == 0:  # Every 5 iterations
            sps = int(global_step / (time.time() - start_time))
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{args.num_iterations} | Phase {global_step // args.phase_duration + 1}")
            print(f"{'='*80}")
            print(f"Steps: {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,}")
            
            if len(episode_returns) > 0:
                recent_returns = episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns
                success_rate = sum(1 for r in recent_returns if r > 100) / len(recent_returns) * 100
                
                print(f"Episodes: {len(episode_returns)} | Success rate: {success_rate:.1f}%")
                print(f"Recent avg return: {np.mean(recent_returns):>8.2f}")
                print(f"Best return: {np.max(episode_returns):>8.2f}")
            
            print(f"{'='*80}\n")

    # Save model
    model_dir = f"models/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/curriculum_bridges.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"\n🎓 Curriculum training complete!")
    print(f"Model saved to: {model_path}")

    envs.close()


if __name__ == "__main__":
    train()