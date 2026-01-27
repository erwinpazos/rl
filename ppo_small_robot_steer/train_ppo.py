"""
Entraînement PPO OPTIMISÉ pour robot dans corridor.
- Environnements parallèles (AsyncVectorEnv)
- Gros batches pour GPU
- Logging efficace
"""
import os
import time
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
    
    print(f"📊 Graphiques sauvegardés: {output_file}")



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CNN UNIQUE + MLP simplifié."""
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        # Observation: pos(3) + vel(3) + angle(1) + bbox(8) + history(88) + grid(1800) = 1903
        self.robot_state_dim = 7   # pos(3) + vel(3) + angle(1)
        self.bbox_dim = 8          # 4 coins × 2 coords
        self.history_dim = 88      # 8 frames × 11 valeurs (8 coins + 3 vitesses) = 88
        self.grid_dim = 1800       # 60×30 = 1800
        
        # MLP pour état robot (position + vitesse + angle)
        self.robot_net = nn.Sequential(
            layer_init(nn.Linear(self.robot_state_dim + self.bbox_dim, 64)),  # 7 + 8 = 15
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # MLP pour historique des positions + vitesses (anticipation)
        self.history_net = nn.Sequential(
            layer_init(nn.Linear(self.history_dim, 128)),  # Plus de neurones pour plus de données (88 → 128)
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
        )
        
        # CNN UNIQUE pour grille 60×30 (plus petite = plus facile)
        self.cnn = nn.Sequential(
            # 60×30 -> 30×15
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 30×15 -> 15×8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 15×8 -> 8×4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # 128 × 8 × 4 = 4096
            layer_init(nn.Linear(4096, 128)),
            nn.Tanh(),
        )
        
        # Backbone combiné
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(64 + 32 + 128, 128)),  # robot_state + history + cnn
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )
        
        # Actor/Critic
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
    
    def forward(self, obs):
        # Décoder observation: pos(3) + vel(3) + angle(1) + bbox(8) + history(88) + grid(1800)
        robot_state = obs[:, :self.robot_state_dim]  # 0:7 (pos + vel + angle)
        bbox = obs[:, self.robot_state_dim:self.robot_state_dim+self.bbox_dim]  # 7:15
        robot_and_bbox = torch.cat([robot_state, bbox], dim=1)  # 15 valeurs
        
        history_start = self.robot_state_dim + self.bbox_dim  # 15
        history = obs[:, history_start:history_start+self.history_dim]  # 15:103
        grid = obs[:, history_start+self.history_dim:].view(-1, 1, 60, 30)  # 103:1903
        
        # Traiter séparément
        robot_feat = self.robot_net(robot_and_bbox)  # 15 → 64
        history_feat = self.history_net(history)      # 88 → 32
        grid_feat = self.cnn(grid)                    # 1800 → 128
        
        # Combiner les trois sources
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


def make_env():
    """Factory pour environnement."""
    def thunk():
        env = CorridorEnv(max_steps=1000)  # Réduit à 1000 steps pour avoir plus d'épisodes
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
    print("\n🎬 DEBUG RENDER - Visualisation d'un épisode...")
    
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
                    
                    # Décoder l'observation AVEC ANGLE + HISTORIQUE ÉTENDU
                    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
                    bbox_corners = obs[7:15]  # 4 coins × 2 coords
                    history_extended = obs[15:103].reshape(8, 11)  # 8 frames × 11 valeurs
                    grid = obs[103:].reshape(60, 30)  # Grille 60×30
                    
                    print(f"  Robot: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f}), vel=({robot_state[3]:.2f}, {robot_state[4]:.2f}, {robot_state[5]:.2f}), angle={robot_state[6]:.2f}rad ({np.degrees(robot_state[6]):.1f}°)")
                    
                    # Afficher bounding box corners
                    corner_names = ['AV-G', 'AV-D', 'AR-G', 'AR-D']
                    bbox_str = ", ".join([f"{name}:({bbox_corners[i*2]:.0f},{bbox_corners[i*2+1]:.0f})" for i, name in enumerate(corner_names)])
                    print(f"  BBox (row,col): {bbox_str}")
                    
                    # Afficher historique étendu (derniers coins + vitesses relatifs)
                    last_frame = history_extended[-1]  # 11 valeurs
                    last_corners = last_frame[:8]  # 8 premiers = coins
                    last_velocities = last_frame[8:]  # 3 derniers = vitesses
                    avg_row_diff = np.mean([last_corners[i*2] for i in range(4)])
                    avg_col_diff = np.mean([last_corners[i*2+1] for i in range(4)])
                    avg_vel = np.mean(last_velocities)
                    print(f"  Historique: coins relatifs avg_row={avg_row_diff:+.1f}, avg_col={avg_col_diff:+.1f}, avg_vel={avg_vel:+.2f}")
                    
                    # Extraire positions des 4 coins pour les afficher sur la grille
                    bbox_positions = [(int(bbox_corners[i*2]), int(bbox_corners[i*2+1])) for i in range(4)]
                    
                    # Afficher grille (20 lignes × TOUTE la largeur 30 colonnes)
                    print("  GRILLE (lignes 0-19, EGO-CENTRIQUE - tourne avec robot):")
                    for i in range(20):  # Lignes 0-19 (robot à ligne 8)
                        line = "    "
                        for j in range(30):  # TOUTES les colonnes (0-29, robot au centre col 15)
                            # Vérifier si c'est un coin de la bounding box
                            is_corner = False
                            for idx, (row, col) in enumerate(bbox_positions):
                                if row == i and col == j:
                                    is_corner = True
                                    line += 'A' if idx < 2 else 'R'  # A=avant, R=arrière
                                    break
                            if not is_corner:
                                val = grid[i, j]
                                if val == -1.0:
                                    line += 'X'  # Extérieur
                                elif val == 0.0:
                                    line += '▓'  # Sol
                                elif val == 0.5:
                                    line += '△'  # Bump
                                else:  # val == 1.0
                                    line += '░'  # Trou
                        relative_dist = (i - 8) * 0.1  # Distance relative au robot (8 = 0.8m derrière)
                        print(f"    {relative_dist:+.1f}m: {line}")
                    print("    (X=extérieur, ▓=sol, △=bump, ░=trou, A=avant bbox, R=arrière bbox)")
                    print("    (Grille EGO-CENTRIQUE: tourne avec le robot, 'devant' = toujours vers le haut)")
                
                v.sync()
                time.sleep(0.05)  # 20 FPS
            
            final_x = debug_env.data.qpos[0]
            reason = info.get('reason', 'truncated')
            print(f"Épisode terminé: {reason} | Steps: {step} | Distance: {final_x:.2f}m | Return: {ep_return:.1f}")
            
            # Attendre un peu pour voir le résultat
            time.sleep(2.0)
            
    except Exception as e:
        print(f"Erreur render: {e}")
        print("Continuons sans render...")


def train(
    total_timesteps=8_000_000,
    num_envs=32,           # PARALLÉLISATION: 32 envs simultanés
    num_steps=1024,        # Steps par rollout par env
    num_minibatches=32,    # Minibatches pour update
    update_epochs=10,      # Epochs par update
    lr=5e-4,  # Apprentissage plus rapide
    gamma=0.995,  # Discount plus élevé pour mieux propager les récompenses lointaines
    gae_lambda=0.98,  # GAE plus élevé pour meilleur credit assignment
    clip_coef=0.2,
    ent_coef=0.05,  # Exploration augmentée pour découvrir les stratégies d'évitement
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=1,
):
    """Entraînement PPO optimisé."""
    
    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculs batch
    batch_size = num_envs * num_steps  # 32 × 1024 = 32768
    minibatch_size = batch_size // num_minibatches  # 1024
    num_iterations = total_timesteps // batch_size
    
    print("=" * 70)
    print("PPO OPTIMISÉ - ENTRAÎNEMENT PARALLÈLE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Environnements parallèles: {num_envs}")
    print(f"Steps par rollout: {num_steps}")
    print(f"Batch size: {batch_size:,}")
    print(f"Minibatch size: {minibatch_size:,}")
    print(f"Iterations: {num_iterations}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 70 + "\n")
    
    # Environnements parallèles avec corridors aléatoires
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    # Environnement de debug pour visualisation
    debug_env = CorridorEnv(max_steps=1000)  # Même durée que les envs d'entraînement
    
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    
    # Agent
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    
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
        
        print(f"🔄 MODÈLE EXISTANT DÉTECTÉ: {latest_model}")
        print(f"   Chargement depuis {latest_step:,} steps...")
        
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            start_iteration = (latest_step // batch_size) + 1
            print(f"   ✅ Modèle chargé ! Reprise à l'itération {start_iteration}")
            print(f"   📊 Steps déjà effectués: {latest_step:,}")
            print(f"   🎯 Steps restants: {total_timesteps - latest_step:,}")
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            print(f"   🔄 Démarrage depuis zéro...")
            start_iteration = 1
    else:
        print("🆕 NOUVEAU MODÈLE - Démarrage depuis zéro")
    
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
    
    # Métriques par batch de 20 épisodes (pour avoir plus de points)
    batch_metrics = []
    batch_size = 20  # Réduit à 20 pour avoir plus de points sur le graphique
    last_batch_episode = 0  # Dernier épisode traité pour les batches
    
    # Compteur raisons de terminaison
    termination_reasons = {
        'success': 0,
        'fell': 0,
        'flipped': 0,
        'out_of_bounds': 0,
        'stuck': 0,
        'truncated': 0,
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
            
            # Log épisodes terminés - LOGIQUE SÉCURISÉE
            for i in range(num_envs):
                if next_done_np[i]:  # Cet environnement s'est terminé
                    # Récupérer infos de manière sécurisée
                    try:
                        if isinstance(infos, dict):
                            # Structure vectorisée: {'key': [val0, val1, ...], '_key': [mask0, mask1, ...]}
                            x_list = infos.get('x', [])
                            x_mask = infos.get('_x', [])
                            dist = x_list[i] if i < len(x_list) and i < len(x_mask) and x_mask[i] else 0
                            
                            reason_list = infos.get('reason', [])
                            reason_mask = infos.get('_reason', [])
                            reason = reason_list[i] if i < len(reason_list) and i < len(reason_mask) and reason_mask[i] else ('truncated' if trunc[i] else 'unknown')
                            
                            # Episode stats du wrapper
                            episode_list = infos.get('episode', [])
                            episode_mask = infos.get('_episode', [])
                            if i < len(episode_list) and i < len(episode_mask) and episode_mask[i] and episode_list[i]:
                                ret = episode_list[i].get('r', 0)
                            else:
                                ret = reward[i] if i < len(reward) else 0
                        else:
                            # Fallback simple
                            ret = reward[i] if i < len(reward) else 0
                            dist = 0
                            reason = "truncated" if trunc[i] else "unknown"
                    except (IndexError, KeyError, TypeError):
                        # Fallback en cas d'erreur
                        ret = reward[i] if i < len(reward) else 0
                        dist = 0
                        reason = "truncated" if trunc[i] else "unknown"
                    
                    episode_returns.append(float(ret))
                    episode_distances.append(float(dist))
                    episode_steps.append(infos.get('step', [0] * num_envs)[i] if isinstance(infos, dict) else 0)
                    total_episodes += 1
                    
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
        if iteration % 2 == 0 or iteration == 1:
            # Vérifier si on a complété un nouveau batch de 100 épisodes
            while total_episodes >= last_batch_episode + batch_size:
                batch_start = last_batch_episode
                batch_end = last_batch_episode + batch_size
                
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
                    'success_rate': 100 * batch_successes / batch_size,
                })
                
                last_batch_episode = batch_end
                print(f"✅ Batch {len(batch_metrics)} complété (épisodes {batch_start+1}-{batch_end})")
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps:,} | Time: {elapsed:.0f}s")
            print(f"Max Steps Curriculum: {current_max_steps}")
            print(f"{'='*70}")
            
            if episode_returns:
                recent_ret = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
                recent_dist = episode_distances[-100:] if len(episode_distances) >= 100 else episode_distances
                recent_steps = episode_steps[-100:] if len(episode_steps) >= 100 else episode_steps
                
                success_rate = 100 * successes / max(1, total_episodes)
                print(f"📊 ÉPISODES: {total_episodes} total | Succès: {successes} ({success_rate:.1f}%)")
                print(f"📊 BATCHES: {len(batch_metrics)} batches de 100 épisodes complétés")
                print(f"📈 RETURN  : Récent {np.mean(recent_ret):>7.1f} ± {np.std(recent_ret):>5.1f} | Meilleur {best_return:>7.1f}")
                print(f"🎯 DISTANCE: Récent {np.mean(recent_dist):>7.1f}m ± {np.std(recent_dist):>5.1f}m | Meilleur {best_distance:>7.1f}m")
                print(f"⏱️  SURVIE  : Récent {np.mean(recent_steps):>7.0f} steps ± {np.std(recent_steps):>5.0f}")
                
                # Récapitulatif terminaisons (seulement si > 0)
                active_reasons = {k: v for k, v in termination_reasons.items() if v > 0}
                if active_reasons:
                    reasons_str = " | ".join([f"{k}:{v}" for k, v in active_reasons.items()])
                    print(f"🔚 TERMINAISONS: {reasons_str}")
            else:
                print(f"⚠️  AUCUN ÉPISODE TERMINÉ (épisodes en cours...)")
            
            print(f"{'='*70}")
        
        # Sauvegarde périodique
        if iteration % 10 == 0:
            model_path = f"models/ppo_corridor_{global_step}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"💾 Modèle sauvegardé: {model_path}")
            
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
                
                print(f"📊 Métriques sauvegardées: {metrics_file}")
        
        # Afficher graphiques toutes les 5 itérations
        if iteration % 5 == 0 and iteration > 0 and batch_metrics:
            plot_training_progress(batch_metrics, iteration)
        
        # Debug render toutes les 5 itérations
        if iteration % 5 == 0:
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
        
        print(f"\n📊 Toutes les métriques sauvegardées: {metrics_file}")
        
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
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fresh-start", action="store_true", help="Forcer un nouveau démarrage (ignorer modèles existants)")
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
    )
