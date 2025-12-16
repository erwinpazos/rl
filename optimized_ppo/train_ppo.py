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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """CNN + MLP optimisé pour GPU."""
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        self.robot_dim = 14  # pos(3) + vel(3) + wheels(8)
        self.env_grid_dim = 2048  # Couche 1: environnement 32×64
        self.robot_grid_dim = 2048  # Couche 2: robot 32×64
        
        # MLP robot
        self.robot_net = nn.Sequential(
            layer_init(nn.Linear(self.robot_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        
        # CNN COUCHE 1: Environnement (sol/trou/rampe)
        self.env_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 32×64 -> 16×32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16×32 -> 8×16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8×16 -> 4×8
            nn.ReLU(),
            nn.Flatten(),  # 64 × 4 × 8 = 2048
            layer_init(nn.Linear(2048, 128)),
            nn.Tanh(),
        )
        
        # CNN COUCHE 2: Robot (position du corps)
        self.robot_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 32×64 -> 16×32
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 16×32 -> 8×16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 8×16 -> 4×8
            nn.ReLU(),
            nn.Flatten(),  # 32 × 4 × 8 = 1024
            layer_init(nn.Linear(1024, 64)),
            nn.Tanh(),
        )
        
        # Backbone combiné - TROIS SOURCES
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(64 + 128 + 64, 256)),  # robot_state + env_cnn + robot_cnn
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
        )
        
        # Actor/Critic
        self.actor_mean = layer_init(nn.Linear(128, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
    
    def forward(self, obs):
        # Décoder les DEUX COUCHES
        robot_state = obs[:, :self.robot_dim]
        env_grid = obs[:, self.robot_dim:self.robot_dim+self.env_grid_dim].view(-1, 1, 32, 64)
        robot_grid = obs[:, self.robot_dim+self.env_grid_dim:].view(-1, 1, 32, 64)
        
        # Traiter chaque couche séparément
        robot_feat = self.robot_net(robot_state)
        env_feat = self.env_cnn(env_grid)      # CNN pour environnement
        robot_pos_feat = self.robot_cnn(robot_grid)  # CNN pour position robot
        
        # Combiner les trois sources d'information
        combined = torch.cat([robot_feat, env_feat, robot_pos_feat], dim=1)
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


def make_env(corridor_xml):
    """Factory pour environnement."""
    def thunk():
        env = CorridorEnv(corridor_xml=corridor_xml, max_steps=300)  # Commencer court
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def update_curriculum(envs, debug_env, iteration, num_iterations):
    """Curriculum progressif: augmenter max_steps avec les itérations."""
    # Progression de 1000 à 3000 steps (plus strict pour forcer apprentissage)
    min_steps = 1000
    max_steps = 3000
    progress = iteration / num_iterations
    
    # Progression linéaire
    current_steps = int(min_steps + (max_steps - min_steps) * progress)
    
    # Mettre à jour tous les environnements
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
            
            obs, _ = debug_env.reset()
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
                    print(f"Step {step}: x={x:.2f}m, reward={reward:.3f}, return={ep_return:.1f}")
                    
                    # Décoder l'observation - DEUX COUCHES
                    robot_state = obs[:6]
                    wheel_pos = obs[6:14]
                    env_grid = obs[14:14+2048].reshape(32, 64)  # Couche 1: environnement
                    robot_grid = obs[14+2048:].reshape(32, 64)  # Couche 2: robot
                    
                    print(f"  Robot: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f}), vel=({robot_state[3]:.2f}, {robot_state[4]:.2f}, {robot_state[5]:.2f})")
                    
                    # Afficher positions des roues
                    wheel_names = ['FL', 'FR', 'RL', 'RR']
                    wheel_str = ", ".join([f"{name}:({wheel_pos[i*2]:.0f},{wheel_pos[i*2+1]:.0f})" for i, name in enumerate(wheel_names)])
                    print(f"  Roues (row,col): {wheel_str}")
                    
                    # Afficher les DEUX COUCHES (échantillon centre)
                    print("  COUCHE 1 - Environnement (centre 16×16):")
                    for i in range(8, 24):  # Lignes 8-23 (centre)
                        line = "    ENV: "
                        for j in range(24, 40):  # Colonnes centre
                            val = env_grid[i, j]
                            if val == 0.0:
                                line += '▓'  # Sol
                            elif val == 0.5:
                                line += '△'  # Rampe
                            else:  # val == 1.0
                                line += '░'  # Trou
                        relative_dist = (i - 16) * 0.0625
                        print(f"    {relative_dist:+.3f}m: {line}")
                    
                    print("  COUCHE 2 - Robot (centre 16×16):")
                    for i in range(8, 24):  # Lignes 8-23 (centre)
                        line = "    ROB: "
                        for j in range(24, 40):  # Colonnes centre
                            val = robot_grid[i, j]
                            line += 'R' if val > 0.5 else '░'
                        relative_dist = (i - 16) * 0.0625
                        print(f"    {relative_dist:+.3f}m: {line}")
                    print("    (ENV: ▓=sol, △=rampe, ░=trou | ROB: R=robot, ░=vide)")
                
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
    total_timesteps=4_000_000,
    num_envs=32,           # PARALLÉLISATION: 32 envs simultanés
    num_steps=1024,        # Steps par rollout par env
    num_minibatches=32,    # Minibatches pour update
    update_epochs=10,      # Epochs par update
    lr=5e-4,  # Apprentissage plus rapide
    gamma=0.995,  # Discount plus élevé pour mieux propager les récompenses lointaines
    gae_lambda=0.98,  # GAE plus élevé pour meilleur credit assignment
    clip_coef=0.2,
    ent_coef=0.01,  # Retour à l'exploration normale
    vf_coef=0.5,
    max_grad_norm=0.5,
    corridor_xml="corridor_3x100.xml",
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
    
    # Environnements parallèles (CLEF DE LA VITESSE)
    envs = gym.vector.AsyncVectorEnv([make_env(corridor_xml) for _ in range(num_envs)])
    
    # Environnement de debug pour visualisation
    debug_env = CorridorEnv(corridor_xml=corridor_xml)
    
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
    best_return = -float('inf')
    best_distance = 0.0
    successes = 0
    total_episodes = 0
    
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
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps:,} | Time: {elapsed:.0f}s")
            print(f"Max Steps Curriculum: {current_max_steps}")
            print(f"{'='*70}")
            
            if episode_returns:
                recent_ret = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
                recent_dist = episode_distances[-100:] if len(episode_distances) >= 100 else episode_distances
                
                success_rate = 100 * successes / max(1, total_episodes)
                print(f"📊 ÉPISODES: {total_episodes} total | Succès: {successes} ({success_rate:.1f}%)")
                print(f"📈 RETURN  : Récent {np.mean(recent_ret):>7.1f} ± {np.std(recent_ret):>5.1f} | Meilleur {best_return:>7.1f}")
                print(f"🎯 DISTANCE: Récent {np.mean(recent_dist):>7.1f}m ± {np.std(recent_dist):>5.1f}m | Meilleur {best_distance:>7.1f}m")
                
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
    print(f"Succès: {successes} ({100*successes/max(1,total_episodes):.1f}%)")
    print(f"Meilleur return: {best_return:.1f}")
    print(f"Meilleure distance: {best_distance:.1f}m")
    
    if episode_returns:
        print(f"Return moyen (last 100): {np.mean(episode_returns[-100:]):.1f}")
        print(f"Distance moyenne (last 100): {np.mean(episode_distances[-100:]):.1f}m")
    
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
    parser.add_argument("--timesteps", type=int, default=4_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=1024)
    parser.add_argument("--corridor", type=str, default="corridor_3x100.xml")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fresh-start", action="store_true", help="Forcer un nouveau démarrage (ignorer modèles existants)")
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        corridor_xml=args.corridor,
        lr=args.lr,
        seed=args.seed,
    )
