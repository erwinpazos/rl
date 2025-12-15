"""
Entraînement PPO simple et efficace pour robot corridor.
Sans LSTM, sans complications. Juste CNN + MLP.
"""
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Normal

from corridor_env import CorridorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Agent PPO avec CNN pour la grille + MLP pour l'état robot."""
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        self.robot_dim = 6  # x, y, z, vx, vy, vz
        self.grid_dim = obs_dim - self.robot_dim  # 192 (16x12 grid)
        
        # Encodeur état robot
        self.robot_net = nn.Sequential(
            layer_init(nn.Linear(self.robot_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
        )
        
        # Encodeur grille (CNN pour 16x12)
        self.grid_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 16x12 -> 8x6
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8x6 -> 4x3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 4x3 -> 4x3
            nn.ReLU(),
            nn.Flatten(),  # 64 * 4 * 3 = 768
            layer_init(nn.Linear(768, 128)),
            nn.Tanh(),
        )
        
        # Backbone commun (32 robot + 128 grid = 160)
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(32 + 128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )
        
        # Têtes actor et critic
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
    
    def forward(self, obs):
        # Séparer robot et grille
        robot = obs[:, :self.robot_dim]
        grid = obs[:, self.robot_dim:].view(-1, 1, 16, 12)
        
        # Encoder
        robot_feat = self.robot_net(robot)
        grid_feat = self.grid_net(grid)
        
        # Combiner
        combined = torch.cat([robot_feat, grid_feat], dim=1)
        features = self.backbone(combined)
        
        return features
    
    def get_value(self, obs):
        return self.critic(self.forward(obs))
    
    def get_action_and_value(self, obs, action=None):
        features = self.forward(obs)
        
        mean = self.actor_mean(features)
        std = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        return (
            action,
            dist.log_prob(action).sum(1),
            dist.entropy().sum(1),
            self.critic(features)
        )


def make_env(corridor_xml):
    def thunk():
        env = CorridorEnv(corridor_xml=corridor_xml)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def train(args):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Environnements parallèles
    envs = gym.vector.AsyncVectorEnv([make_env(args.corridor) for _ in range(args.num_envs)])
    
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    
    # Agent
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    
    # Calculs
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    
    print(f"\n{'='*60}")
    print(f"PPO Training - Robot Corridor")
    print(f"{'='*60}")
    print(f"Corridor: {args.corridor}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, act_dim)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Init
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    episode_returns = []
    episode_distances = []
    best_return = -float('inf')
    best_distance = 0.0
    successes = 0
    total_episodes = 0

    for iteration in range(1, num_iterations + 1):
        # Collecte rollouts
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            
            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)
            
            # Log épisodes terminés
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ret = info["episode"]["r"]
                        dist = info.get("x", 0)
                        reason = info.get("reason", "?")
                        
                        episode_returns.append(ret)
                        episode_distances.append(dist)
                        total_episodes += 1
                        
                        if ret > best_return:
                            best_return = ret
                        if dist > best_distance:
                            best_distance = dist
                        if reason == "success":
                            successes += 1
        
        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards).to(device)
            lastgae = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgae = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgae
            returns = advantages + values
        
        # Flatten
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Optimisation
        b_inds = np.arange(batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy
                ent_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        
        # Stats itération
        if iteration % 2 == 0:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps} | Time: {elapsed:.0f}s")
            print(f"{'='*70}")
            
            if episode_returns:
                recent_ret = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns
                recent_dist = episode_distances[-50:] if len(episode_distances) >= 50 else episode_distances
                
                print(f"Episodes: {total_episodes} | Succès: {successes} ({100*successes/max(1,total_episodes):.1f}%)")
                print(f"Return   - Recent: {np.mean(recent_ret):>7.1f} ± {np.std(recent_ret):>5.1f} | Best: {best_return:>7.1f}")
                print(f"Distance - Recent: {np.mean(recent_dist):>7.1f}m ± {np.std(recent_dist):>5.1f}m | Best: {best_distance:>7.1f}m")
            print(f"{'='*70}\n")
    
    # Sauvegarder
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_corridor_{int(time.time())}.pth"
    torch.save(agent.state_dict(), model_path)
    
    # Résumé final
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*70}")
    print(f"Durée totale: {elapsed/60:.1f} minutes")
    print(f"Episodes joués: {total_episodes}")
    print(f"Succès (100m): {successes} ({100*successes/max(1,total_episodes):.1f}%)")
    print(f"Meilleur return: {best_return:.1f}")
    print(f"Meilleure distance: {best_distance:.1f}m")
    if episode_returns:
        print(f"Return moyen (derniers 100): {np.mean(episode_returns[-100:]):.1f}")
        print(f"Distance moyenne (derniers 100): {np.mean(episode_distances[-100:]):.1f}m")
    print(f"Modèle sauvegardé: {model_path}")
    print(f"{'='*70}\n")
    
    envs.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    train(args)
