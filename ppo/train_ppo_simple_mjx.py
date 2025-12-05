"""
PPO training with MuJoCo MJX (GPU-accelerated physics).
Simplified environment without complex obstacles.
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
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from jax import jit


# Configuration
class Config:
    exp_name = "ppo_robot_mjx"
    seed = 1
    total_timesteps = 2000000
    learning_rate = 3e-4
    num_envs = 2048  # MASSIVE parallelization on GPU!
    num_steps = 128   # Shorter rollouts for GPU efficiency
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 32
    update_epochs = 4
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    batch_size = 0
    minibatch_size = 0
    num_iterations = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

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


# MJX Vectorized Environment
class MJXVectorEnv:
    def __init__(self, num_envs, max_steps=1000):
        self.num_envs = num_envs
        self.max_steps = max_steps
        
        # Load model
        print(f"Loading MJX model for {num_envs} environments...")
        mj_model = mujoco.MjModel.from_xml_path("robot_simple_mjx.xml")
        self.model = mjx.put_model(mj_model)
        
        # Observation and action spaces
        self.obs_size = 6  # x, y, z, vx, vy, vz
        self.action_size = 4  # 4 wheel velocities
        
        # Initialize data for all environments
        self.reset_jax()
        
        # JIT compile step function
        self._jit_step = jit(self._step_fn)
        self._jit_reset = jit(self._reset_fn)
        
        print(f"✓ MJX environment ready with {num_envs} parallel envs on GPU")
    
    def reset_jax(self):
        """Reset all environments (JAX)."""
        self.data = jax.vmap(lambda _: mjx.make_data(self.model))(jnp.arange(self.num_envs))
        self.steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
        self.previous_x = jnp.zeros(self.num_envs, dtype=jnp.float32)
        return self._get_obs()
    
    def _reset_fn(self, data):
        """Reset single environment."""
        return mjx.make_data(self.model)
    
    def _step_fn(self, model, data, action):
        """Step single environment (5 substeps)."""
        data = data.replace(ctrl=action)
        def step_once(d, _):
            return mjx.step(model, d), None
        data, _ = jax.lax.scan(step_once, data, None, length=5)
        return data
    
    def step(self, actions):
        """Step all environments."""
        # Convert actions to JAX
        actions_jax = jnp.array(actions, dtype=jnp.float32)
        
        # Step all environments in parallel on GPU
        self.data = jax.vmap(self._jit_step, in_axes=(None, 0, 0))(
            self.model, self.data, actions_jax
        )
        
        self.steps += 1
        
        # Get observations
        obs = self._get_obs()
        
        # Compute rewards
        rewards, dones = self._compute_rewards()
        
        # Reset done environments
        reset_mask = dones
        reset_data = jax.vmap(self._jit_reset)(jnp.arange(self.num_envs))
        
        # Manually reset each field
        self.data = self.data.replace(
            qpos=jnp.where(reset_mask[:, None], reset_data.qpos, self.data.qpos),
            qvel=jnp.where(reset_mask[:, None], reset_data.qvel, self.data.qvel),
            ctrl=jnp.where(reset_mask[:, None], reset_data.ctrl, self.data.ctrl),
        )
        self.steps = jnp.where(reset_mask, 0, self.steps)
        
        return obs, rewards, dones, {}
    
    def _get_obs(self):
        """Get observations from all environments."""
        pos = self.data.qpos[:, :3]  # (num_envs, 3)
        vel = self.data.qvel[:, :3]  # (num_envs, 3)
        obs = jnp.concatenate([pos, vel], axis=1)  # (num_envs, 6)
        return np.array(obs, dtype=np.float32)
    
    def _compute_rewards(self):
        """Compute rewards for all environments."""
        x = self.data.qpos[:, 0]
        z = self.data.qpos[:, 2]
        
        # Terminal conditions
        success = x >= 100.0
        fell = (z < 0.1) & (x > 0)
        backward = x < -1.0
        timeout = self.steps >= self.max_steps
        
        # Rewards
        rewards = jnp.where(success, 100.0,
                   jnp.where(fell, -100.0,
                   jnp.where(backward, -50.0,
                   (x - self.previous_x) * 10.0 - 0.01)))
        
        self.previous_x = x
        
        dones = success | fell | backward | timeout
        
        return np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool_)


if __name__ == "__main__":
    args = Config()
    
    # Compute derived values
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("PPO TRAINING - MJX GPU ACCELERATION")
    print("="*60)
    print(f"\nRun name: {args.exp_name}_{int(time.time())}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Parallel envs: {args.num_envs} (GPU vectorized!)")
    print(f"Steps per iteration: {args.num_steps}")
    print("="*60)
    print(f"\nUsing device: {device}")
    print()
    
    # Create MJX environment
    envs = MJXVectorEnv(args.num_envs)
    
    # Create agent
    agent = Agent(envs.obs_size, envs.action_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs, envs.obs_size)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.action_size)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Start training
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset_jax()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    print("Starting training...\n")
    
    for iteration in range(1, args.num_iterations + 1):
        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            next_obs_np, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
        
        # Bootstrap value
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
        b_obs = obs.reshape((-1, envs.obs_size))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, envs.action_size))
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
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
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
        
        # Print progress
        sps = int(global_step / (time.time() - start_time))
        elapsed_time = time.time() - start_time
        print(f"Iter {iteration}/{args.num_iterations} | Step {global_step}/{args.total_timesteps} | SPS: {sps:,} | Time: {elapsed_time:.1f}s")
        
        import sys
        sys.stdout.flush()
    
    # Save model
    run_name = f"{args.exp_name}_{int(time.time())}"
    model_dir = f"models/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ppo_robot_mjx.pth"
    torch.save(agent.state_dict(), model_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*60}\n")
