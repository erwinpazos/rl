"""
Entraînement PPO OPTIMISÉ pour robot dans corridor.
- Environnements parallèles (AsyncVectorEnv)
- Gros batches pour GPU
- Logging efficace
- Configuration via fichier JSON
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import mujoco
from mujoco import viewer

from corridor_env import CorridorEnv

# Import matplotlib pour les graphiques (backend non-interactif)
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt


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


def plot_training_progress(metrics_list, current_iteration):
    """Sauvegarde les graphiques de progression (pas d'affichage)."""
    if not metrics_list:
        return
    
    # Extraire les données - utiliser episode_end pour l'axe X
    episodes = [m['episode_end'] for m in metrics_list]
    returns = [m['mean_return'] for m in metrics_list]
    distances = [m['mean_distance'] for m in metrics_list]
    success_rates = [m['success_rate'] for m in metrics_list]
    survivals = [m['mean_survival'] for m in metrics_list]
    
    # Créer figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Progression entraînement - Itération {current_iteration} ({len(episodes)} points, batch=20 épisodes)', fontsize=14)
    
    # 1. Return moyen
    ax = axes[0, 0]
    ax.plot(episodes, returns, 'b-o', linewidth=2, markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Return moyen')
    ax.set_title('Return moyen (par batch de 20)')
    ax.grid(True, alpha=0.3)
    
    # 2. Distance moyenne
    ax = axes[0, 1]
    ax.plot(episodes, distances, 'g-o', linewidth=2, markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance moyenne (par batch de 20)')
    ax.grid(True, alpha=0.3)
    
    # 3. Taux de succès
    ax = axes[1, 0]
    ax.plot(episodes, success_rates, 'r-o', linewidth=2, markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Taux de succès (%)')
    ax.set_title('Taux de succès (par batch de 20)')
    ax.grid(True, alpha=0.3)
    
    # 4. Survie moyenne
    ax = axes[1, 1]
    ax.plot(episodes, survivals, 'm-o', linewidth=2, markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Steps de survie')
    ax.set_title('Durée moyenne (par batch de 20)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder uniquement
    os.makedirs("models", exist_ok=True)
    output_file = f"models/training_progress_iter_{current_iteration}.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close(fig)  # Fermer pour libérer la mémoire
    
    print(f"PLOTS: Graphs saved to {output_file}")



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CNN SIMPLIFIÉ + MLP réduit - Architecture allégée."""
    
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
        temp_env = CorridorEnv()
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
        
        # CNN SIMPLIFIÉ pour grille 60×30×2 (SEULEMENT 2 COUCHES)
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
        grid_feat = self.cnn(grid)                    # 4560 → 64
        
        # Combiner les trois sources (32 + 32 + 64 = 128)
        combined = torch.cat([robot_feat, history_feat, grid_feat], dim=1)
        return self.backbone(combined)
    
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


def make_env(config=None):
    """Factory pour environnement avec corridors aléatoires générés EN MÉMOIRE."""
    def thunk():
        if config and 'environment' in config:
            env_config = config['environment']
            max_steps = env_config.get('max_steps', 1000)
            use_random = env_config.get('use_random_corridor', True)
            corridor_xml_file = env_config.get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
        else:
            max_steps = 1000
            use_random = True
            corridor_xml_file = 'corridor_3x100_no_full_obstacles.xml'
        
        # Utiliser corridor_xml=None pour générer aléatoirement EN MÉMOIRE
        corridor_xml = None if use_random else corridor_xml_file
            
        env = CorridorEnv(max_steps=max_steps, corridor_xml=corridor_xml)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def update_curriculum(envs, debug_env, iteration, num_iterations):
    """Pas de curriculum - toujours 3000 steps."""
    # Toujours 3000 steps maintenant
    current_steps = 3000
    
    # Mettre à jour tous les environnements (au cas où)
    try:
        # Pour les envs vectorisés, on doit accéder aux envs individuels
        for i in range(len(envs.envs)):
            if hasattr(envs.envs[i], 'env') and hasattr(envs.envs[i].env, 'env'):
                # Unwrap les wrappers
                base_env = envs.envs[i].env.env
                if hasattr(base_env, 'set_max_steps'):
                    base_env.set_max_steps(current_steps)
        
        # Debug env
        debug_env.set_max_steps(current_steps)
        
        return current_steps
    except:
        return current_steps


def debug_render_episode(agent, debug_env, device, max_steps=None):
    """Render un épisode de debug pour voir ce qui se passe."""
    print("\nDEBUG: Rendering episode visualization...")
    
    # Reset AVANT de créer le viewer (génère nouveau corridor + nouveau modèle)
    obs, _ = debug_env.reset()
    
    # Maintenant utiliser le nouveau modèle/data
    m = debug_env.model
    d = debug_env.data
    robot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    
    try:
        with viewer.launch_passive(m, d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = robot_id
            v.cam.azimuth = 180
            v.cam.elevation = -20
            v.cam.distance = 8
            
            done = False
            step = 0
            ep_return = 0
            
            # Utiliser le max_steps de l'environnement si pas spécifié
            if max_steps is None:
                max_steps = debug_env.max_steps
            
            print(f"Position initiale: x={debug_env.data.qpos[0]:.2f}, y={debug_env.data.qpos[1]:.2f}")
            print(f"Max steps pour cet épisode: {max_steps}")
            
            while not done and v.is_running() and step < max_steps:
                # Action de l'agent
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action, _, _, _ = agent.get_action_and_value(obs_t)
                    action = action.cpu().numpy()[0]
                
                obs, reward, term, trunc, info = debug_env.step(action)
                done = term or trunc
                ep_return += reward
                step += 1
                
                # Afficher info + vision
                if step % 25 == 0:
                    x = debug_env.data.qpos[0]
                    stabilizing = " (STABILISATION)" if step < debug_env.stabilization_steps else ""
                    print(f"Step {step}: x={x:.2f}m, reward={reward:.3f}, return={ep_return:.1f}{stabilizing}")
                    
                    # Décoder l'observation SIMPLIFIÉE AVEC HISTORIQUE RÉDUIT
                    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
                    history_simplified = obs[7:7+debug_env.history_dim].reshape(debug_env.history_length, 6)  # frames × 6 valeurs
                    grid = obs[7+debug_env.history_dim:].reshape(debug_env.grid_rows, debug_env.grid_cols, 2)  # Grille dynamique×2
                    
                    print(f"  Robot: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f}), vel=({robot_state[3]:.2f}, {robot_state[4]:.2f}, {robot_state[5]:.2f}), angle={robot_state[6]:.2f}rad ({np.degrees(robot_state[6]):.1f}°)")
                    
                    # Afficher historique simplifié (dernière frame)
                    last_frame = history_simplified[-1]  # 6 valeurs: pos(3) + vel(3)
                    last_pos = last_frame[:3]
                    last_vel = last_frame[3:]
                    print(f"  Historique (4 frames): dernière pos=({last_pos[0]:+.2f}, {last_pos[1]:+.2f}, {last_pos[2]:+.2f}), vel=({last_vel[0]:+.2f}, {last_vel[1]:+.2f}, {last_vel[2]:+.2f})")
                    
                    # Afficher grille (20 lignes × TOUTE la largeur 30 colonnes) - Canal 0 (obstacles)
                    obstacles_grid = grid[:, :, 0]  # Canal obstacles
                    trous_grid = grid[:, :, 1]      # Canal trous
                    
                    print(f"  GRILLE (lignes 0-{debug_env.grid_rows-1}, EGO-CENTRIQUE - tourne avec robot):")
                    for i in range(min(debug_env.grid_rows, 20)):  # Limiter l'affichage à 20 lignes max
                        line = "    "
                        for j in range(min(debug_env.grid_cols, 40)):  # Limiter l'affichage à 40 colonnes max
                            if obstacles_grid[i, j] > 0.5:
                                line += '#'  # Obstacle (bump)
                            elif trous_grid[i, j] > 0.5:
                                line += '.'  # Trou
                            else:
                                line += ' '  # Sol
                        relative_dist = (i - debug_env.robot_row_in_grid) * debug_env.cell_size  # Distance relative au robot
                        print(f"    {relative_dist:+.1f}m: {line}")
                    print("    (space=floor, #=obstacle/bump, .=hole)")
                    print("    (EGO-CENTRIC grid: rotates with robot, 'forward' = always up)")
                    print(f"    (Vision: {debug_env.vision_length}m x {debug_env.vision_width}m, {debug_env.cell_size}m cells)")
                
                v.sync()
                time.sleep(0.05)  # 20 FPS
            
            final_x = debug_env.data.qpos[0]
            reason = info.get('reason', 'truncated')
            print(f"Episode ended: {reason} | Steps: {step} | Distance: {final_x:.2f}m | Reward: {ep_return:.1f}")
            
            # Attendre un peu pour voir le résultat
            time.sleep(2.0)
            
    except Exception as e:
        print(f"Erreur render: {e}")
        print("Continuons sans render...")


def train(config_path="config.yaml", **kwargs):
    """Entraînement PPO optimisé avec configuration JSON."""
    
    # Charger configuration
    config = load_config(config_path)
    
    # Paramètres par défaut (si pas de config ou valeurs manquantes)
    defaults = {
        'total_timesteps': 8_000_000,
        'num_envs': 32,
        'num_steps': 1024,
        'num_minibatches': 32,
        'update_epochs': 10,
        'lr': 5e-4,
        'gamma': 0.995,
        'gae_lambda': 0.98,
        'clip_coef': 0.2,
        'ent_coef': 0.05,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'seed': 1,
    }
    
    # Fusionner config avec kwargs (kwargs prioritaires)
    if config:
        # Extraire les paramètres de training et ppo
        training_params = config.get('training', {})
        ppo_params = config.get('ppo', {})
        optimizer_params = config.get('optimizer', {})
        logging_params = config.get('logging', {})
        
        # Fusionner tous les paramètres
        params = {}
        params.update(training_params)
        params.update(ppo_params)
        params.update(optimizer_params)
        params.update(logging_params)
        
        # Appliquer les valeurs par défaut pour les paramètres manquants
        for key, default_value in defaults.items():
            params[key] = params.get(key, default_value)
    else:
        params = defaults.copy()
    
    # Appliquer les kwargs (priorité maximale)
    params.update(kwargs)
    
    # Extraire les paramètres
    total_timesteps = params['total_timesteps']
    num_envs = params['num_envs']
    num_steps = params['num_steps']
    num_minibatches = params['num_minibatches']
    update_epochs = params['update_epochs']
    lr = params['lr']
    gamma = params['gamma']
    gae_lambda = params['gae_lambda']
    clip_coef = params['clip_coef']
    ent_coef = params['ent_coef']
    vf_coef = params['vf_coef']
    max_grad_norm = params['max_grad_norm']
    seed = params['seed']
    
    # Paramètres de logging
    log_interval = params.get('log_interval', 2)
    save_interval = params.get('save_interval', 10)
    render_interval = params.get('render_interval', 5)
    plot_interval = params.get('plot_interval', 5)
    batch_size_metrics = params.get('batch_size_metrics', 20)
    
    # Paramètres optimizer
    optimizer_eps = params.get('eps', 1e-5)
    
    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculs batch
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size
    
    print("=" * 70)
    print("PPO OPTIMISÉ - ENTRAÎNEMENT PARALLÈLE")
    print("=" * 70)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Environnements parallèles: {num_envs}")
    print(f"Steps par rollout: {num_steps}")
    print(f"Batch size: {batch_size:,}")
    print(f"Minibatch size: {minibatch_size:,}")
    print(f"Iterations: {num_iterations}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"GAE Lambda: {gae_lambda}")
    print("=" * 70 + "\n")
    
    # Environnements parallèles avec corridors aléatoires
    envs = gym.vector.AsyncVectorEnv([make_env(config) for _ in range(num_envs)])
    
    # Environnement de debug pour visualisation avec corridor aléatoire EN MÉMOIRE
    if config and 'environment' in config:
        env_config = config['environment']
        debug_max_steps = env_config.get('max_steps', 1000)
        use_random = env_config.get('use_random_corridor', True)
        corridor_xml_file = env_config.get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
        debug_corridor_xml = None if use_random else corridor_xml_file
    else:
        debug_max_steps = 1000
        debug_corridor_xml = None  # Génération aléatoire
        
    debug_env = CorridorEnv(max_steps=debug_max_steps, corridor_xml=debug_corridor_xml)
    
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    
    # Agent avec configuration
    agent = Agent(obs_dim, act_dim, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=optimizer_eps)
    
    # DÉTECTION ET CHARGEMENT DE MODÈLE EXISTANT
    start_iteration = 1
    existing_models = []
    if os.path.exists("models") and not getattr(args, 'fresh_start', False):
        for file in os.listdir("models"):
            if file.startswith("ppo_corridor_") and file.endswith(".pth") and file != "ppo_corridor_final.pth":
                try:
                    # Extraire le numéro de step du nom de fichier
                    step_num = int(file.replace("ppo_corridor_", "").replace(".pth", ""))
                    existing_models.append((step_num, file))
                except ValueError:
                    continue
    
    if existing_models:
        # Prendre le modèle le plus récent
        existing_models.sort(key=lambda x: x[0], reverse=True)
        latest_step, latest_model = existing_models[0]
        model_path = os.path.join("models", latest_model)
        
        print(f"RESUME: Existing model detected: {latest_model}")
        print(f"   Loading from {latest_step:,} steps...")
        
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            start_iteration = (latest_step // batch_size) + 1
            print(f"   OK: Model loaded! Resuming at iteration {start_iteration}")
            print(f"   STATS: Steps completed: {latest_step:,}")
            print(f"   TARGET: Steps remaining: {total_timesteps - latest_step:,}")
        except Exception as e:
            print(f"   ERROR: Loading failed: {e}")
            print(f"   RESTART: Starting from scratch...")
            start_iteration = 1
    else:
        print("NEW: New model - Starting from scratch")
    
    # Buffers GPU
    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    
    # Init
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)
    
    # Stats
    episode_returns = []
    episode_distances = []
    episode_steps = []  # Nouveau : durée des épisodes
    best_return = -float('inf')
    best_distance = 0.0
    successes = 0
    total_episodes = 0
    
    # Métriques par batch (configurable)
    batch_metrics = []
    last_batch_episode = 0  # Dernier épisode traité pour les batches
    
    # Compteur raisons de terminaison
    termination_reasons = {
        'success': 0,
        'fell': 0,
        'flipped': 0,
        'collision': 0,
        'out_of_bounds': 0,
        'stuck': 0,
        'truncated': 0,
        'terminated': 0,
    }
    
    os.makedirs("models", exist_ok=True)

    for iteration in range(start_iteration, num_iterations + 1):
        # === CURRICULUM: Ajuster max_steps progressivement ===
        current_max_steps = update_curriculum(envs, debug_env, iteration, num_iterations)
        
        # === COLLECTE ROLLOUTS (parallèle sur num_envs) ===
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            
            # Step tous les envs en parallèle
            next_obs_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(term, trunc)
            
            rewards[step] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)
            
            # Log épisodes terminés - LOGIQUE CORRIGÉE
            for i in range(num_envs):
                if next_done_np[i]:  # Cet environnement s'est terminé
                    # Récupérer infos de manière simple et robuste
                    try:
                        # Essayer d'abord le format simple (liste d'infos)
                        if isinstance(infos, list) and i < len(infos) and infos[i]:
                            info = infos[i]
                            reason = info.get('reason', 'truncated' if trunc[i] else 'unknown')
                            dist = info.get('x', 0)
                            ret = info.get('episode', {}).get('r', reward[i] if i < len(reward) else 0)
                            steps = info.get('step', 0)
                        
                        # Sinon essayer le format vectorisé avec masques
                        elif isinstance(infos, dict):
                            # Récupérer reason avec masque - CORRECTION: reason_list est un numpy array
                            reason_list = infos.get('reason', [])
                            reason_mask = infos.get('_reason', [])
                            if i < len(reason_list) and i < len(reason_mask) and reason_mask[i] and reason_list[i] is not None:
                                reason = reason_list[i]
                            else:
                                reason = 'truncated' if trunc[i] else 'terminated'
                            
                            # Récupérer distance avec masque - CORRECTION: x_list est un numpy array
                            x_list = infos.get('x', [])
                            x_mask = infos.get('_x', [])
                            if i < len(x_list) and i < len(x_mask) and x_mask[i]:
                                dist = float(x_list[i])
                            else:
                                dist = 0.0
                            
                            # Récupérer steps avec masque - CORRECTION: step_list est un numpy array
                            step_list = infos.get('step', [])
                            step_mask = infos.get('_step', [])
                            if i < len(step_list) and i < len(step_mask) and step_mask[i]:
                                steps = int(step_list[i])
                            else:
                                steps = 0
                            
                            # Récupérer reward depuis episode stats - CORRECTION: episode est un dict avec arrays
                            episode_dict = infos.get('episode', {})
                            episode_mask = infos.get('_episode', [])
                            if i < len(episode_mask) and episode_mask[i] and 'r' in episode_dict:
                                episode_r = episode_dict['r']
                                episode_r_mask = episode_dict.get('_r', [])
                                if i < len(episode_r) and i < len(episode_r_mask) and episode_r_mask[i]:
                                    ret = float(episode_r[i])
                                else:
                                    ret = float(reward[i]) if i < len(reward) else 0.0
                            else:
                                ret = float(reward[i]) if i < len(reward) else 0.0
                        
                        else:
                            # Fallback complet
                            reason = 'truncated' if trunc[i] else 'unknown'
                            dist = 0.0
                            ret = float(reward[i]) if i < len(reward) else 0.0
                            steps = 0
                            
                    except (IndexError, KeyError, TypeError, AttributeError) as e:
                        # Fallback en cas d'erreur
                        reason = 'truncated' if trunc[i] else 'terminated'
                        dist = 0.0
                        ret = float(reward[i]) if i < len(reward) else 0.0
                        steps = 0
                        print(f"  WARNING: Episode info extraction failed for env {i}: {e}")
                    
                    episode_returns.append(ret)
                    episode_distances.append(dist)
                    episode_steps.append(steps)
                    total_episodes += 1
                    
                    # Log individuel pour chaque épisode
                    print(f"Episode {total_episodes}: {reason} | Steps: {steps} | Distance: {dist:.2f}m | Reward: {ret:.1f}")
                    
                    if ret > best_return:
                        best_return = ret
                    if dist > best_distance:
                        best_distance = dist
                    
                    # Compter raison de terminaison
                    if reason in termination_reasons:
                        termination_reasons[reason] += 1
                    if reason == "success":
                        successes += 1
        
        # === GAE ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgae = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgae = delta + gamma * gae_lambda * nextnonterminal * lastgae
            
            returns = advantages + values
        
        # === FLATTEN POUR UPDATE ===
        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # === UPDATE PPO ===
        b_inds = np.arange(batch_size)
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Normalize advantages
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy
                ent_loss = entropy.mean()
                
                # Total
                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        # === LOGGING ===
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        
        # === LOGGING PROPRE PAR ITÉRATION ===
        if iteration % log_interval == 0 or iteration == 1:
            # Vérifier si on a complété un nouveau batch d'épisodes
            while total_episodes >= last_batch_episode + batch_size_metrics:
                batch_start = last_batch_episode
                batch_end = last_batch_episode + batch_size_metrics
                
                # Calculer moyennes pour ce batch de 100 épisodes
                batch_returns = episode_returns[batch_start:batch_end]
                batch_distances = episode_distances[batch_start:batch_end]
                batch_steps_list = episode_steps[batch_start:batch_end]
                
                # Compter succès dans ce batch
                batch_successes = sum(1 for i in range(batch_start, batch_end) 
                                     if episode_distances[i] >= 100.0)
                
                batch_metrics.append({
                    'batch_num': len(batch_metrics) + 1,
                    'episode_end': batch_end,  # Numéro du dernier épisode de ce batch
                    'episodes_range': f"{batch_start+1}-{batch_end}",
                    'global_step': global_step,
                    'mean_return': np.mean(batch_returns),
                    'mean_distance': np.mean(batch_distances),
                    'mean_survival': np.mean(batch_steps_list),
                    'success_rate': 100 * batch_successes / batch_size_metrics,
                })
                
                last_batch_episode = batch_end
                print(f"BATCH: Batch {len(batch_metrics)} completed (episodes {batch_start+1}-{batch_end})")
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps:,} | Time: {elapsed:.0f}s")
            print(f"Max Steps Curriculum: {current_max_steps}")
            print(f"{'='*70}")
            
            if episode_returns:
                recent_ret = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
                recent_dist = episode_distances[-100:] if len(episode_distances) >= 100 else episode_distances
                recent_steps = episode_steps[-100:] if len(episode_steps) >= 100 else episode_steps
                
                success_rate = 100 * successes / max(1, total_episodes)
                print(f"EPISODES: {total_episodes} total | Success: {successes} ({success_rate:.1f}%)")
                print(f"BATCHES: {len(batch_metrics)} batches of {batch_size_metrics} episodes completed")
                print(f"REWARD  : Recent {np.mean(recent_ret):>7.1f} +/- {np.std(recent_ret):>5.1f} | Best {best_return:>7.1f}")
                print(f"DISTANCE: Recent {np.mean(recent_dist):>7.1f}m +/- {np.std(recent_dist):>5.1f}m | Best {best_distance:>7.1f}m")
                print(f"SURVIVAL: Recent {np.mean(recent_steps):>7.0f} steps +/- {np.std(recent_steps):>5.0f}")
                
                # Récapitulatif terminaisons (seulement si > 0)
                active_reasons = {k: v for k, v in termination_reasons.items() if v > 0}
                if active_reasons:
                    reasons_str = " | ".join([f"{k}:{v}" for k, v in active_reasons.items()])
                    print(f"TERMINATIONS: {reasons_str}")
            else:
                print(f"WARNING: No episodes completed (episodes in progress...)")
            
            print(f"{'='*70}")
        
        # Sauvegarde périodique
        if iteration % save_interval == 0:
            model_path = f"models/ppo_corridor_{global_step}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"SAVE: Model saved to {model_path}")
            
            # Sauvegarder métriques des batches en CSV
            if batch_metrics:
                import csv
                metrics_file = "models/training_metrics.csv"
                
                with open(metrics_file, 'w', newline='') as f:  # 'w' pour réécrire tout
                    writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                                           'global_step', 'mean_return', 'mean_distance', 
                                                           'mean_survival', 'success_rate'])
                    writer.writeheader()
                    
                    # Écrire tous les batches
                    for metrics in batch_metrics:
                        writer.writerow(metrics)
                
                print(f"METRICS: Batch metrics saved to {metrics_file}")
        
        # Afficher graphiques
        if iteration % plot_interval == 0 and iteration > 0 and batch_metrics:
            plot_training_progress(batch_metrics, iteration)
        
        # Debug render
        if iteration % render_interval == 0 or iteration == 1:
            debug_render_episode(agent, debug_env, device)
    
    # === FIN ===
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*70}")
    print(f"Durée: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"SPS moyen: {total_timesteps/elapsed:.0f}")
    print(f"Episodes: {total_episodes}")
    print(f"Batches de 100 épisodes: {len(batch_metrics)}")
    print(f"Succès: {successes} ({100*successes/max(1,total_episodes):.1f}%)")
    print(f"Meilleur return: {best_return:.1f}")
    print(f"Meilleure distance: {best_distance:.1f}m")
    
    if batch_metrics:
        last_batch = batch_metrics[-1]
        print(f"\nDernier batch complété:")
        print(f"  Return moyen: {last_batch['mean_return']:.1f}")
        print(f"  Distance moyenne: {last_batch['mean_distance']:.1f}m")
        print(f"  Survie moyenne: {last_batch['mean_survival']:.0f} steps")
        print(f"  Taux de succès: {last_batch['success_rate']:.1f}%")
    
    # Sauvegarder métriques finales
    if batch_metrics:
        import csv
        metrics_file = "models/training_metrics.csv"
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                                   'global_step', 'mean_return', 'mean_distance', 
                                                   'mean_survival', 'success_rate'])
            writer.writeheader()
            
            for metrics in batch_metrics:
                writer.writerow(metrics)
        
        print(f"\nMETRICS: All metrics saved to {metrics_file}")
        
        # Générer graphique final
        plot_training_progress(batch_metrics, num_iterations)
    
    # Sauvegarde finale
    final_path = "models/ppo_corridor_final.pth"
    torch.save(agent.state_dict(), final_path)
    print(f"\nModèle sauvegardé: {final_path}")
    print(f"{'='*70}\n")
    
    envs.close()
    return final_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Fichier de configuration YAML")
    parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--num-envs", type=int, help="Override nombre d'environnements")
    parser.add_argument("--num-steps", type=int, help="Override steps par rollout")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--fresh-start", action="store_true", help="Forcer un nouveau démarrage (ignorer modèles existants)")
    args = parser.parse_args()
    
    # Préparer les kwargs pour override
    kwargs = {}
    if args.timesteps is not None:
        kwargs['total_timesteps'] = args.timesteps
    if args.num_envs is not None:
        kwargs['num_envs'] = args.num_envs
    if args.num_steps is not None:
        kwargs['num_steps'] = args.num_steps
    if args.lr is not None:
        kwargs['lr'] = args.lr
    if args.seed is not None:
        kwargs['seed'] = args.seed
    
    # Rendre args accessible globalement pour fresh_start
    globals()['args'] = args
    
    train(config_path=args.config, **kwargs)
