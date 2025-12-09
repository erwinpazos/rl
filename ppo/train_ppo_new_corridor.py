"""
Simplified PPO training without tyro and tensorboard.
Just the essentials for training the robot.
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.normal import Normal

# Import our custom environment (CPU optimized - MJX not compatible with complex collisions)
from robot_corridor_env_new import RobotCorridorEnv


# Simple configuration class
class Config:
    # Experiment
    exp_name = "ppo_robot_corridor"
    seed = 1
    
    # Training
    total_timesteps = 8000000  # Faster training for testing
    learning_rate = 3e-4
    num_envs = 28  # Optimized for 32 CPU cores
    num_steps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    
    # PPO
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.05  # Increased from 0.01 to 0.05 for more exploration
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Computed
    batch_size = 0
    minibatch_size = 0
    num_iterations = 0


def make_env():
    """Create a single environment (CPU optimized)."""
    def thunk():
        env = RobotCorridorEnv()
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
        
        # Critic (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor (policy)
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
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train():
    """Main training function."""
    args = Config()
    
    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    
    print("="*60)
    print("PPO TRAINING - ROBOT CORRIDOR")
    print("="*60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Steps per iteration: {args.num_steps}")
    print("="*60)
    print()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create environments
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(args.num_envs)])

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
                        
                        # Print every episode
                        print(f"Step {global_step:>8} | Return: {info['episode']['r']:>7.2f} | Length: {info['episode']['l']:>4}")
            
            # Also count manual terminations (terminated=True but not in final_info)
            for env_id in range(args.num_envs):
                if terminations[env_id] or truncations[env_id]:
                    # Episode ended, but we don't have the full stats from RecordEpisodeStatistics
                    # Just count it
                    if len(episode_returns) == 0 or step > 0:  # Avoid counting initial resets
                        pass  # Stats will come from final_info if available
            
            # DEBUG: Print position of first 3 envs at step 2000 of each episode
            # This allows comparison across iterations
            for env_id in range(min(3, args.num_envs)):
                # Unwrap to get the actual RobotCorridorEnv
                env = envs.envs[env_id]
                while hasattr(env, 'env'):
                    env = env.env
                
                # Print only when episode reaches step 2000
                if env.current_step == 2000:
                    x_pos = env.data.qpos[0]
                    y_pos = env.data.qpos[1]
                    z_pos = env.data.qpos[2]
                    print(f"  [Env {env_id}] Step 2000: x={x_pos:.2f}m, y={y_pos:.2f}m, z={z_pos:.2f}m")

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

        # Print iteration summary with detailed stats
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
        else:
            print(f"\n⚠️  Episode stats not available")
            print(f"   (Episodes are terminating - see [TERM] messages above)")
            print(f"   RecordEpisodeStatistics may not be capturing all terminations")
        
        print(f"{'='*80}\n")
        
        # Flush output to see it immediately
        import sys
        sys.stdout.flush()

    # Save model
    model_dir = f"models/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ppo_robot_corridor.pth"
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
        print()

    envs.close()


if __name__ == "__main__":
    train()
