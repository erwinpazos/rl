"""
Test d'un agent PPO entraîné.
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
        
        # Observation: pos(3) + vel(3) + bbox(8) + history(40) + grid(7200) = 7254
        self.robot_state_dim = 6   # pos(3) + vel(3)
        self.bbox_dim = 8          # 4 coins × 2 coords
        self.history_dim = 40      # 5 positions × 8 coords (4 coins × 2) = 40
        self.grid_dim = 7200       # 120×60 = 7200
        
        # MLP pour état robot (position + vitesse + bbox)
        self.robot_net = nn.Sequential(
            layer_init(nn.Linear(self.robot_state_dim + self.bbox_dim, 64)),  # 14
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # MLP pour historique des 4 coins (anticipation)
        self.history_net = nn.Sequential(
            layer_init(nn.Linear(self.history_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        
        # CNN UNIQUE pour grille 120×60
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 120×60 -> 60×30
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 60×30 -> 30×15
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 30×15 -> 15×8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 15×8 -> 8×4
            nn.ReLU(),
            nn.Flatten(),  # 256 × 8 × 4 = 8192
            layer_init(nn.Linear(8192, 256)),
            nn.Tanh(),
        )
        
        # Backbone combiné
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(64 + 32 + 256, 256)),  # robot + history + cnn
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
        )
        
        self.actor_mean = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
    
    def forward(self, obs):
        # Décoder observation
        robot_state = obs[:, :self.robot_state_dim]  # 0:6
        bbox = obs[:, self.robot_state_dim:self.robot_state_dim+self.bbox_dim]  # 6:14
        robot_and_bbox = torch.cat([robot_state, bbox], dim=1)  # 14 valeurs
        
        history_start = self.robot_state_dim + self.bbox_dim  # 14
        history = obs[:, history_start:history_start+self.history_dim]  # 14:54
        grid = obs[:, history_start+self.history_dim:].view(-1, 1, 120, 60)  # 54:7254
        
        # Traiter séparément
        robot_feat = self.robot_net(robot_and_bbox)
        history_feat = self.history_net(history)
        grid_feat = self.cnn(grid)
        
        combined = torch.cat([robot_feat, history_feat, grid_feat], dim=1)
        return self.backbone(combined)
    
    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        
        if deterministic:
            return mean
        
        std = self.actor_logstd.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std).sample()


def display_vision(obs, step, ret):
    """Afficher vision robot dans terminal."""
    print("\033[2J\033[H", end="")
    
    # Décoder observation: pos(3) + vel(3) + bbox(8) + history(40) + grid(7200)
    robot = obs[:6]
    bbox = obs[6:14]
    grid = obs[54:].reshape(120, 60)
    
    print("=" * 70)
    print(f"Step: {step} | Return: {ret:.1f}")
    print(f"Position: x={robot[0]:.2f}m, y={robot[1]:.2f}m, z={robot[2]:.2f}m")
    print(f"Velocity: vx={robot[3]:.2f}, vy={robot[4]:.2f}, vz={robot[5]:.2f}")
    print("=" * 70)
    print("\nVision 120×60 (centre 20×20 autour du robot):")
    print("-" * 40)
    
    # Afficher grille centre (lignes 30-49, colonnes 20-39)
    for i in range(30, 50):
        relative_dist = (i - 40) * 0.05
        line = f"{relative_dist:+.2f}m: "
        for j in range(20, 40):
            val = grid[i, j]
            if val == 0.0:
                line += '▓'
            elif val == 0.5:
                line += '△'
            else:
                line += '░'
        print(line)
    
    print("-" * 40)
    print("Légende: ▓=sol  △=rampe  ░=trou")
    print("=" * 70)


def test(model_path, corridor_xml, num_episodes, render, show_vision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = CorridorEnv(corridor_xml=corridor_xml)
    
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
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0
            step = 0
            
            while not done:
                if show_vision:
                    display_vision(obs, step, ep_return)
                    time.sleep(0.05)
                
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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--show-vision", action="store_true")
    args = parser.parse_args()
    
    model_path = args.model
    if model_path is None:
        models = glob.glob("models/ppo_corridor_*.pth")
        if not models:
            print("Aucun modèle trouvé!")
            exit(1)
        models.sort(key=os.path.getmtime, reverse=True)
        model_path = models[0]
        print(f"Auto-détecté: {model_path}\n")
    
    test(model_path, args.corridor, args.episodes, args.render, args.show_vision)
