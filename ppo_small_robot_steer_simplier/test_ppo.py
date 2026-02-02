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
        
        # Observation: pos(3) + vel(3) + bbox(8) + history(88) + grid(5400) = 5502
        self.robot_state_dim = 6   # pos(3) + vel(3)
        self.bbox_dim = 8          # 4 coins × 2 coords
        self.history_dim = 88      # 8 frames × 11 valeurs (8 coins + 3 vitesses) = 88
        self.grid_dim = 5400       # 60×30×3 = 5400
        
        # MLP pour état robot (position + vitesse + bbox)
        self.robot_net = nn.Sequential(
            layer_init(nn.Linear(self.robot_state_dim + self.bbox_dim, 64)),  # 14
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # MLP pour historique des positions + vitesses (anticipation)
        self.history_net = nn.Sequential(
            layer_init(nn.Linear(self.history_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        
        # CNN UNIQUE pour grille 60×30×3
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 60×30 -> 30×15
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 30×15 -> 15×8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 15×8 -> 8×4
            nn.ReLU(),
            nn.Flatten(),  # 128 × 8 × 4 = 4096
            layer_init(nn.Linear(4096, 128)),
            nn.Tanh(),
        )
        
        # Backbone combiné
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(64 + 32 + 128, 128)),  # robot + history + cnn
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )
        
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
    
    def forward(self, obs):
        # Décoder observation
        robot_state = obs[:, :self.robot_state_dim]  # 0:6
        bbox = obs[:, self.robot_state_dim:self.robot_state_dim+self.bbox_dim]  # 6:14
        robot_and_bbox = torch.cat([robot_state, bbox], dim=1)  # 14 valeurs
        
        history_start = self.robot_state_dim + self.bbox_dim  # 14
        history = obs[:, history_start:history_start+self.history_dim]  # 14:102
        grid = obs[:, history_start+self.history_dim:].view(-1, 3, 60, 30)  # 102:5502 → (batch, 3, 60, 30)
        
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
    
    # Décoder observation: pos(3) + vel(3) + bbox(8) + history(88) + grid(5400)
    robot = obs[:6]
    bbox = obs[6:14]
    grid = obs[102:].reshape(60, 30, 3)  # 3 canaux
    
    print("=" * 50)
    print(f"Step: {step} | Return: {ret:.1f}")
    print(f"Position: x={robot[0]:.2f}m, y={robot[1]:.2f}m, z={robot[2]:.2f}m")
    print(f"Velocity: vx={robot[3]:.2f}, vy={robot[4]:.2f}, vz={robot[5]:.2f}")
    print("=" * 50)
    print("\nVision 60×30×3 EGO-CENTRIQUE (robot à ligne 8):")
    print("Canal 0=Sol, Canal 1=Obstacles, Canal 2=Trous")
    print("-" * 40)
    
    # Afficher grille combinée (20 premières lignes)
    for i in range(20):
        relative_dist = (i - 8) * 0.1  # Robot à ligne 8
        line = f"{relative_dist:+.1f}m: "
        for j in range(30):
            # Combiner les 3 canaux pour affichage
            sol = grid[i, j, 0]
            obstacle = grid[i, j, 1]
            trou = grid[i, j, 2]
            
            if obstacle > 0.5:
                line += 'X'  # Obstacle (bump ou extérieur)
            elif trou > 0.5:
                line += '░'  # Trou
            elif sol > 0.5:
                line += '▓'  # Sol
            else:
                line += '?'  # Erreur
        print(line)
    
    print("-" * 40)
    print("Légende: X=obstacle  ▓=sol  ░=trou")
    print("3 canaux binaires: [sol, obstacles, trous]")
    print("=" * 50)


def test(model_path, num_episodes, render, show_vision, corridor_xml=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Model: {model_path}")
    if corridor_xml:
        print(f"Corridor: {corridor_xml} (fixe)")
    else:
        print(f"Corridor: généré aléatoirement")
    print(f"Device: {device}\n")
    
    returns = []
    distances = []
    
    if render:
        for ep in range(num_episodes):
            # Reset AVANT de créer le viewer (génère nouveau corridor + nouveau modèle)
            obs, _ = env.reset()
            
            # Créer viewer avec le nouveau modèle
            m = env.model
            d = env.data
            robot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            
            with viewer.launch_passive(m, d) as v:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                v.cam.trackbodyid = robot_id
                v.cam.azimuth = 180
                v.cam.elevation = -20
                v.cam.distance = 8
                
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
            print(f"Episode {ep + 1}: Reward={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={info.get('reason', 'truncated')}")
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
            print(f"Episode {ep + 1}: Reward={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={info.get('reason', 'truncated')}")
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    print(f"Average distance: {np.mean(distances):.1f}m +/- {np.std(distances):.1f}m")
    print(f"Best distance: {max(distances):.1f}m")
    print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--corridor", type=str, default=None, help="Corridor XML fixe (défaut: aléatoire)")
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
    
    test(model_path, args.episodes, args.render, args.show_vision, args.corridor)
