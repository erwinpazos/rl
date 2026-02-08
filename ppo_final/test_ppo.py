"""
Test d'un agent PPO entraîné.
Compatible avec le système de configuration YAML dynamique.
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


def load_config(config_path="config.yaml"):
    """Charge la configuration depuis un fichier YAML."""
    import yaml
    
    if not os.path.exists(config_path):
        print(f"WARNING: Config file {config_path} not found, using default values")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"OK: Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"ERROR: Failed to load {config_path}: {e}")
        return None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Architecture dynamique compatible avec le nouveau système YAML."""
    
    def __init__(self, obs_dim, act_dim, config=None):
        super().__init__()
        
        # Configuration simplifiée
        if config and 'network' in config:
            net_config = config['network']
            robot_hidden = net_config.get('robot_net_hidden', [32])
            history_hidden = net_config.get('history_net_hidden', [64, 32])
            cnn_channels = net_config.get('cnn_channels', [32, 64])
            cnn_kernel_size = net_config.get('cnn_kernel_size', 3)
            cnn_stride = net_config.get('cnn_stride', 2)
            backbone_hidden = net_config.get('backbone_hidden', [64])
        else:
            # Valeurs par défaut SIMPLIFIÉES
            robot_hidden = [32]
            history_hidden = [64, 32]
            cnn_channels = [32, 64]
            cnn_kernel_size = 3
            cnn_stride = 2
            backbone_hidden = [64]
        
        # Calculer dimensions dynamiquement depuis l'observation
        # Créer un environnement temporaire pour obtenir les dimensions exactes
        # Utiliser corridor_xml=None pour forcer la génération aléatoire (pas de dépendance fichier)
        temp_env = CorridorEnv(corridor_xml=None)
        self.history_dim = temp_env.history_dim
        self.grid_dim = temp_env.grid_dim
        self.grid_rows = temp_env.grid_rows
        self.grid_cols = temp_env.grid_cols
        temp_env.close()
        
        print(f"NETWORK: Using dimensions from environment:")
        print(f"   History: {self.history_dim} values")
        print(f"   Grid: {self.grid_rows} x {self.grid_cols} = {self.grid_dim} values")
        
        # MLP pour état robot SIMPLIFIÉ (position + vitesse + angle)
        robot_layers = []
        prev_dim = 7  # Toujours 7 valeurs (x,y,z,vx,vy,vz,theta)
        for hidden_dim in robot_hidden:
            robot_layers.extend([
                layer_init(nn.Linear(prev_dim, hidden_dim)),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        self.robot_net = nn.Sequential(*robot_layers)
        
        # MLP pour historique RÉDUIT (anticipation)
        history_layers = []
        prev_dim = self.history_dim  # 24 au lieu de 48
        for hidden_dim in history_hidden:
            history_layers.extend([
                layer_init(nn.Linear(prev_dim, hidden_dim)),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        self.history_net = nn.Sequential(*history_layers)
        
        # CNN SIMPLIFIÉ pour grille dynamique×2 (SEULEMENT 2 COUCHES)
        cnn_layers = []
        in_channels = 2  # 2 canaux : obstacles, trous
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=cnn_kernel_size, stride=cnn_stride, padding=1),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        # Calculer la taille après convolutions dynamiquement
        # Avec stride=2 et padding=1, chaque conv divise par 2 (arrondi vers le haut)
        conv_rows = self.grid_rows
        conv_cols = self.grid_cols
        for _ in cnn_channels:  # Pour chaque couche de convolution
            conv_rows = (conv_rows + 2 * 1 - cnn_kernel_size) // cnn_stride + 1  # padding=1
            conv_cols = (conv_cols + 2 * 1 - cnn_kernel_size) // cnn_stride + 1
        
        final_size = cnn_channels[-1] * conv_rows * conv_cols
        print(f"NETWORK: CNN output size: {cnn_channels[-1]} x {conv_rows} x {conv_cols} = {final_size}")
        
        cnn_layers.extend([
            nn.Flatten(),
            layer_init(nn.Linear(final_size, backbone_hidden[0])),
            nn.Tanh()
        ])
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Backbone combiné SIMPLIFIÉ
        backbone_input_dim = robot_hidden[-1] + history_hidden[-1] + backbone_hidden[0]  # 32 + 32 + 64 = 128
        backbone_layers = []
        prev_dim = backbone_input_dim
        for hidden_dim in backbone_hidden:
            backbone_layers.extend([
                layer_init(nn.Linear(prev_dim, hidden_dim)),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Actor/Critic
        final_dim = backbone_hidden[-1]
        self.actor_mean = layer_init(nn.Linear(final_dim, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(final_dim, 1), std=1.0)
    
    def forward(self, obs):
        # Décoder observation: pos(3) + vel(3) + angle(1) + history + grid
        robot_state = obs[:, :7]  # Toujours 7 valeurs (x,y,z,vx,vy,vz,theta)
        
        history_start = 7
        history = obs[:, history_start:history_start+self.history_dim]
        grid = obs[:, history_start+self.history_dim:].view(-1, 2, self.grid_rows, self.grid_cols)
        
        # Traiter séparément avec architecture SIMPLIFIÉE
        robot_feat = self.robot_net(robot_state)      # 7 → 32
        history_feat = self.history_net(history)      # 24 → 32
        grid_feat = self.cnn(grid)                    # dynamique → 64
        
        # Combiner les trois sources (32 + 32 + 64 = 128)
        combined = torch.cat([robot_feat, history_feat, grid_feat], dim=1)
        return self.backbone(combined)
    
    def get_action(self, obs, deterministic=False):
        features = self.forward(obs)
        mean = self.actor_mean(features)
        
        if deterministic:
            return mean
        
        std = self.actor_logstd.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std).sample()


def display_vision(obs, step, ret, env):
    """Afficher vision robot dans terminal."""
    print("\033[2J\033[H", end="")
    
    # Décoder observation: pos(3) + vel(3) + angle(1) + history + grid
    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
    history_start = 7
    history = obs[history_start:history_start+env.history_dim].reshape(env.history_length, 6)
    grid = obs[history_start+env.history_dim:].reshape(env.grid_rows, env.grid_cols, 2)
    
    print("=" * 60)
    print(f"Step: {step} | Return: {ret:.1f}")
    print(f"Position: x={robot_state[0]:.2f}m, y={robot_state[1]:.2f}m, z={robot_state[2]:.2f}m")
    print(f"Velocity: vx={robot_state[3]:.2f}, vy={robot_state[4]:.2f}, vz={robot_state[5]:.2f}")
    print(f"Angle: {robot_state[6]:.2f}rad ({np.degrees(robot_state[6]):.1f}°)")
    print("=" * 60)
    print(f"\nVision {env.grid_rows}×{env.grid_cols}×2 EGO-CENTRIQUE (robot à ligne {env.robot_row_in_grid}):")
    print("Canal 0=Obstacles, Canal 1=Trous")
    print("-" * 50)
    
    # Afficher grille combinée (20 premières lignes max)
    display_rows = min(env.grid_rows, 20)
    for i in range(display_rows):
        relative_dist = (i - env.robot_row_in_grid) * env.cell_size
        line = f"{relative_dist:+.1f}m: "
        for j in range(min(env.grid_cols, 40)):  # Limiter colonnes
            # Combiner les 2 canaux pour affichage
            obstacle = grid[i, j, 0]
            trou = grid[i, j, 1]
            
            if obstacle > 0.5:
                line += '#'  # Obstacle (bump)
            elif trou > 0.5:
                line += '.'  # Trou
            else:
                line += '/'  # Sol navigable
        print(line)
    
    print("-" * 50)
    print("Légende: /=sol navigable  #=obstacle/bump  .=trou")
    print("2 canaux binaires: [obstacles, trous]")
    print(f"Vision: {env.vision_length:.1f}m x {env.vision_width:.1f}m, cellules de {env.cell_size}m")
    print("=" * 60)


def make_env(config=None, bump_ratio=None):
    """Factory pour environnement avec configuration et bump_ratio."""
    if config and 'environment' in config:
        env_config = config['environment']
        max_steps = env_config.get('max_steps', 1000)
        use_random = env_config.get('use_random_corridor', True)
        corridor_xml_file = env_config.get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
    else:
        max_steps = 1000
        use_random = True
        corridor_xml_file = 'corridor_3x100_no_full_obstacles.xml'
    
    # Utiliser corridor_xml=None pour générer aléatoirement
    corridor_xml = None if use_random else corridor_xml_file
    
    # Utiliser bump_ratio par défaut si pas fourni
    env_bump_ratio = bump_ratio if bump_ratio is not None else 0.0
    
    env = CorridorEnv(max_steps=max_steps, corridor_xml=corridor_xml, obstacle_type="holes")
    env.bump_ratio = env_bump_ratio  # Initialiser bump_ratio
    return env


def test(model_path, num_episodes, render, show_vision, config_path="config.yaml", corridor_xml=None, bump_ratio=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger configuration
    config = load_config(config_path)
    
    # Créer environnement avec config
    if corridor_xml:
        # Override avec corridor spécifique
        env = CorridorEnv(corridor_xml=corridor_xml)
        if bump_ratio is not None:
            env.bump_ratio = bump_ratio
    else:
        env = make_env(config, bump_ratio=bump_ratio)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim, config).to(device)
    
    # Charger le checkpoint (peut contenir model_state_dict ou être directement le state_dict)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    
    print(f"Model: {model_path}")
    if corridor_xml:
        print(f"Corridor: {corridor_xml} (fixe)")
    elif config and config.get('environment', {}).get('use_random_corridor', True):
        print(f"Corridor: généré aléatoirement")
    else:
        corridor_file = config.get('environment', {}).get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
        print(f"Corridor: {corridor_file} (fixe)")
    
    # Afficher le bump_ratio
    bump_pct = (bump_ratio if bump_ratio is not None else 0.0) * 100
    print(f"Obstacles: 100% holes + {bump_pct:.0f}% bumps")
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
                        display_vision(obs, step, ep_return, env)
                    
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
            # Déterminer la vraie raison de terminaison
            if term:
                reason = info.get('reason', 'terminated')
            elif trunc:
                reason = 'truncated'
            else:
                reason = 'unknown'
            print(f"Episode {ep + 1}: Reward={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={reason}")
    else:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0
            step = 0
            
            while not done:
                if show_vision:
                    display_vision(obs, step, ep_return, env)
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
            # Déterminer la vraie raison de terminaison
            if term:
                reason = info.get('reason', 'terminated')
            elif trunc:
                reason = 'truncated'
            else:
                reason = 'unknown'
            print(f"Episode {ep + 1}: Reward={ep_return:.1f}, Distance={info['x']:.1f}m, Reason={reason}")
    
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
    parser.add_argument("--config", type=str, default="config.yaml", help="Fichier de configuration YAML")
    parser.add_argument("--corridor", type=str, default=None, help="Corridor XML fixe (défaut: selon config)")
    parser.add_argument("--bump", type=float, default=None, help="Pourcentage de bumps en plus des holes (0.0-1.0, ex: 0.3 = 30%%)")
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
    
    test(model_path, args.episodes, args.render, args.show_vision, args.config, args.corridor, args.bump)
