"""
Train robot basic locomotion - Learn to control 4 wheels from scratch.
No pre-made steering, robot must discover differential drive.
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

# Import our basic control environment
from robot_basic_env import RobotBasicEnv


# Configuration for basic control learning
class Config:
    # Experiment
    exp_name = "basic_robot_control"
    seed = 1
    
    # Training - Longer episodes for exploration
    total_timesteps = 8000000  # 2M steps for basic control
    learning_rate = 3e-4
    num_envs = 16  # Fewer envs, more exploration per env
    num_steps = 2048
    gamma = 0.99
    gae_lambda = 0.95
    
    # PPO - Higher exploration for discovery
    num_minibatches = 16
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.2  # Very high exploration for learning control
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # Computed
    batch_size = 0
    minibatch_size = 0
    num_iterations = 0


def make_env():
    """Create a single basic control environment."""
    def thunk():
        env = RobotBasicEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BasicControlAgent(nn.Module):
    """
    Agent for learning basic robot control.
    Simpler architecture focused on motor control learning.
    """
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
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


def train(resume_from=None):
    """Main training function for basic control."""
    args = Config()
    
    # Compute batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    
    print("="*60)
    print("BASIC ROBOT CONTROL TRAINING")
    print("="*60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Steps per iteration: {args.num_steps}")
    print(f"Learning: 4-wheel differential drive from scratch")
    print("="*60)
    print()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create environments
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(args.num_envs)])

    # Create agent
    agent = BasicControlAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Resume from checkpoint if available
    start_iteration = 1
    if resume_from:
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        global_step = checkpoint['global_step']
        best_avg_return = checkpoint.get('best_avg_return', float('-inf'))
        best_success_rate = checkpoint.get('best_success_rate', 0.0)
        print(f"Resumed from iteration {start_iteration}, global step {global_step}")
        print(f"Previous best: Avg Return {best_avg_return:.2f}, Success Rate {best_success_rate:.2f}")
    else:
        global_step = 0
        best_avg_return = float('-inf')
        best_success_rate = 0.0

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Track statistics
    episode_returns = []
    episode_lengths = []
    targets_reached = []
    
    # Best model tracking
    best_avg_return = float('-inf')
    best_success_rate = 0.0
    iterations_without_improvement = 0

    print("Starting basic control training...\n")

    for iteration in range(start_iteration, args.num_iterations + 1):
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
                        
                        # Count targets reached (estimate from high rewards)
                        if info["episode"]["r"] > 10:
                            targets_reached.append(1)
                        else:
                            targets_reached.append(0)
                        
                        # Print every episode with control info
                        print(f"Step {global_step:>8} | Return: {info['episode']['r']:>7.2f} | Length: {info['episode']['l']:>4} | Targets: {sum(targets_reached[-10:])}/10")

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
        sps = int(global_step / (time.time() - start_time))
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{args.num_iterations}")
        print(f"{'='*80}")
        print(f"Steps: {global_step:,}/{args.total_timesteps:,} | SPS: {sps:,} | Time: {elapsed_time:.1f}s")
        
        if len(episode_returns) > 0:
            recent_returns = episode_returns[-10:] if len(episode_returns) >= 10 else episode_returns
            recent_targets = targets_reached[-10:] if len(targets_reached) >= 10 else targets_reached
            
            print(f"\nEpisodes completed: {len(episode_returns)}")
            print(f"  Last 10 episodes:")
            print(f"    Avg Return:  {np.mean(recent_returns):>8.2f}")
            print(f"    Targets Hit: {sum(recent_targets)}/10 ({100*np.mean(recent_targets):.1f}%)")
            print(f"  All time:")
            print(f"    Best Return: {np.max(episode_returns):>8.2f}")
            print(f"    Success Rate: {100*np.mean(targets_reached):.1f}%")
        else:
            print(f"\n⚠️  No episodes completed yet")
        
        print(f"{'='*80}\n")
        
        # Evaluate performance and save best model
        checkpoint_dir = f"models/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Calculate current performance metrics
        current_avg_return = np.mean(episode_returns[-50:]) if len(episode_returns) >= 50 else (np.mean(episode_returns) if episode_returns else float('-inf'))
        current_success_rate = np.mean(targets_reached[-50:]) if len(targets_reached) >= 50 else (np.mean(targets_reached) if targets_reached else 0.0)
        
        # Check if this is the best model so far
        is_best = False
        if current_avg_return > best_avg_return or (current_avg_return == best_avg_return and current_success_rate > best_success_rate):
            best_avg_return = current_avg_return
            best_success_rate = current_success_rate
            iterations_without_improvement = 0
            is_best = True
            
            # Save best model
            best_model_path = f"{checkpoint_dir}/best_model.pth"
            torch.save({
                'iteration': iteration,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_avg_return': best_avg_return,
                'best_success_rate': best_success_rate,
                'args': args,
            }, best_model_path)
            print(f"🏆 NEW BEST MODEL! Avg Return: {current_avg_return:.2f}, Success Rate: {current_success_rate:.2f}")
            print(f"💾 Best model saved: {best_model_path}")
        else:
            iterations_without_improvement += 1
        
        # Save regular checkpoint every 10 iterations
        if iteration % 10 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_iter_{iteration}.pth"
            torch.save({
                'iteration': iteration,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_avg_return': best_avg_return,
                'best_success_rate': best_success_rate,
                'args': args,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Keep only last 3 checkpoints to save space
            import glob
            checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_iter_*.pth")
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(checkpoints) > 3:
                for old_checkpoint in checkpoints[:-3]:
                    os.remove(old_checkpoint)
                    print(f"🗑️  Removed old checkpoint: {old_checkpoint}")
        
        # Early stopping if no improvement for too long
        if iterations_without_improvement > 50:
            print(f"⚠️  No improvement for {iterations_without_improvement} iterations. Consider stopping.")
        
        # Flush output
        import sys
        sys.stdout.flush()

    # Save model
    model_dir = f"models/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/basic_control.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"\n{'='*60}")
    print(f"Basic control training complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")

    # Final statistics
    if len(episode_returns) > 0:
        print("Final Statistics:")
        print(f"  Episodes completed: {len(episode_returns)}")
        print(f"  Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"  Best return: {max(episode_returns):.2f}")
        print(f"  Target success rate: {100*np.mean(targets_reached):.1f}%")
        print()

    envs.close()


def find_latest_checkpoint():
    """Find the latest checkpoint to resume from."""
    import glob
    checkpoints = glob.glob("models/basic_robot_control_*/checkpoint_iter_*.pth")
    if not checkpoints:
        return None
    
    # Sort by modification time and return the latest
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def find_best_model():
    """Find the best model for testing."""
    import glob
    best_models = glob.glob("models/basic_robot_control_*/best_model.pth")
    if not best_models:
        return None
    
    # Sort by modification time and return the latest
    best_models.sort(key=os.path.getmtime, reverse=True)
    return best_models[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train basic robot control")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to resume from")
    args = parser.parse_args()
    
    resume_from = None
    if args.checkpoint:
        resume_from = args.checkpoint
        print(f"Resuming from specified checkpoint: {resume_from}")
    elif args.resume:
        resume_from = find_latest_checkpoint()
        if resume_from:
            print(f"Auto-detected latest checkpoint: {resume_from}")
        else:
            print("No checkpoint found, starting fresh training")
    
    train(resume_from=resume_from)