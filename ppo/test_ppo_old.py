"""
Test d'un agent PPO entraîné sur le corridor.
"""
import argparse
import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import mujoco
from mujoco import viewer

from corridor_env import CorridorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Même architecture que train_ppo.py"""
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        self.robot_dim = 6
        self.grid_dim = obs_dim - self.robot_dim  # 192 (16x12 grid)
        
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
        
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
    
    def forward(self, obs):
        robot = obs[:, :self.robot_dim]
        grid = obs[:, self.robot_dim:].view(-1, 1, 16, 12)
        
        robot_feat = self.robot_net(robot)
        grid_feat = self.grid_net(grid)
        
        combined = torch.cat([robot_feat, grid_feat], dim=1)
        return self.backbone(combined)
    
    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        
        if deterministic:
            return mean
        
        std = self.actor_logstd.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std).sample()


def display_vision(obs, step, ret):
    """Afficher la vision du robot dans le terminal."""
    print("\033[2J\033[H", end="")  # Clear screen
    
    robot = obs[:6]
    grid = obs[6:].reshape(16, 12)  # 16 lignes × 12 colonnes
    
    print("=" * 60)
    print(f"Step: {step} | Return: {ret:.1f}")
    print(f"Position: x={robot[0]:.2f}m, y={robot[1]:.2f}m, z={robot[2]:.2f}m")
    print(f"Velocity: vx={robot[3]:.2f}, vy={robot[4]:.2f}, vz={robot[5]:.2f}")
    print("=" * 60)
    print("\nVision 16×12 (4m devant × 3m large) - cellules 25cm:")
    print("-" * 40)
    
    symbols = {0: '▓', 1: '△', 2: '░'}
    colors = {0: '\033[92m', 1: '\033[93m', 2: '\033[91m'}
    reset = '\033[0m'
    
    # Afficher de loin (haut) à proche (bas)
    for i in range(15, -1, -1):
        dist = (i + 1) * 0.25  # Distance en mètres
        line = f"{dist:4.2f}m: "
        for val in grid[i]:
            v = int(val)
            line += f"{colors.get(v, '')}{symbols.get(v, '?')}{reset}"
        print(line)
    
    print("-" * 40)
    print("Légende: \033[92m▓\033[0m=sol  \033[93m△\033[0m=rampe  \033[91m░\033[0m=trou")
    print("=" * 60)


def test(model_path, corridor_xml, num_episodes, render, show_vision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Créer env
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Charger agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Model: {model_path}")
    print(f"Corridor: {corridor_xml}")
    print(f"Device: {device}\n")
    
    returns = []
    distances = []
    
    if render:
        # Avec visualisation MuJoCo
        m = env.model
        d = env.data
        robot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
        
        with viewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = robot_id
            v.cam.azimuth = 180
            v.cam.elevation = -20
            v.cam.distance = 8
            
            for ep in range(num_episodes):
                obs, _ = env.reset()
                done = False
                ep_return = 0
                step = 0
                
                print(f"\nEpisode {ep + 1}...")
                
                while not done and v.is_running():
                    if show_vision:
                        display_vision(obs, step, ep_return)
                    
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                        action = agent.get_action(obs_t, deterministic=True)
                        action = action.cpu().numpy()[0]
                    
                    obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    ep_return += reward
                    step += 1
                    
                    v.sync()
                    time.sleep(0.02)
                
                returns.append(ep_return)
                distances.append(info['x'])
                print(f"Episode {ep + 1}: Return={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={info.get('reason', 'truncated')}")
                
                if not v.is_running():
                    break
    else:
        # Sans visualisation
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0
            step = 0
            
            while not done:
                if show_vision:
                    display_vision(obs, step, ep_return)
                    time.sleep(0.1)
                
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action = agent.get_action(obs_t, deterministic=True)
                    action = action.cpu().numpy()[0]
                
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                ep_return += reward
                step += 1
            
            returns.append(ep_return)
            distances.append(info['x'])
            print(f"Episode {ep + 1}: Return={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={info.get('reason', 'truncated')}")
    
    # Stats
    print(f"\n{'='*50}")
    print("RÉSULTATS")
    print(f"{'='*50}")
    print(f"Episodes: {num_episodes}")
    print(f"Return moyen: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"Distance moyenne: {np.mean(distances):.1f}m ± {np.std(distances):.1f}m")
    print(f"Meilleure distance: {max(distances):.1f}m")
    print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Chemin modèle (auto-détecte si vide)")
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Visualisation 3D")
    parser.add_argument("--show-vision", action="store_true", help="Afficher vision robot")
    args = parser.parse_args()
    
    # Auto-détection modèle
    model_path = args.model
    if model_path is None:
        models = glob.glob("models/ppo_corridor_*.pth")
        if not models:
            print("Aucun modèle trouvé! Entraîne d'abord avec train_ppo.py")
            exit(1)
        models.sort(key=os.path.getmtime, reverse=True)
        model_path = models[0]
        print(f"Auto-détecté: {model_path}\n")
    
    test(model_path, args.corridor, args.episodes, args.render, args.show_vision)
