"""
Entraînement PPO OPTIMISÉ pour robot dans corridor.
- Environnements parallèles (AsyncVectorEnv)
- Gros batches pour GPU
- Logging efficace
- Configuration via fichier JSON
# test
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

# Import des utilitaires
from utils.load_utils import find_latest_checkpoint, load_checkpoint, load_last_iteration_summary, get_mean_distance_from_temp, load_temp_metrics, get_last_batch_num
from utils.save_utils import save_checkpoint, save_metrics_to_csv, save_temp_batch_to_csv, save_iteration_summary, flush_temp_to_main_metrics, save_episode_to_temp_log, flush_temp_episode_logs, plot_training_curves
from utils.metrics_utils import (
    IterationTracker, clear_temp_metrics, save_checkpoint_summary_to_log, compute_checkpoint_metrics
)
from utils.display_utils import check_and_install_display_dependencies, VisionWindow

from corridor_env import CorridorEnv

# Vérifier et installer les dépendances d'affichage
check_and_install_display_dependencies()

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


def make_env(config=None, random_percentage=None, curriculum_max_steps=None, bump_ratio=None):
    """Factory pour environnement avec curriculum de corridors aléatoires, max_steps et bump ratio."""
    def thunk():
        if config and 'environment' in config:
            env_config = config['environment']
            base_max_steps = env_config.get('max_steps', 1000)
            use_random_base = env_config.get('use_random_corridor', True)
            corridor_xml_file = env_config.get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
        else:
            base_max_steps = 1000
            use_random_base = True
            corridor_xml_file = 'corridor_3x100_no_full_obstacles.xml'
        
        # Utiliser curriculum_max_steps si fourni, sinon base_max_steps
        max_steps = curriculum_max_steps if curriculum_max_steps is not None else base_max_steps
        
        # Utiliser bump_ratio par défaut si pas fourni
        env_bump_ratio = bump_ratio if bump_ratio is not None else 0.0
        
        # Toujours utiliser le générateur maintenant
        corridor_xml = None
        env_random_percentage = random_percentage
        use_fixed_seed = False
            
        env = CorridorEnv(max_steps=max_steps, corridor_xml=corridor_xml, obstacle_type="holes", use_fixed_seed=use_fixed_seed, random_percentage=env_random_percentage)
        env.bump_ratio = env_bump_ratio  # Initialiser bump_ratio
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk


def get_curriculum_state(config, batch_metrics):
    """Obtenir l'état actuel du curriculum basé sur la distance du dernier batch."""
    if not config or 'curriculum' not in config:
        return None, None, None, 1, 0.0, False
    
    curriculum_config = config['curriculum']
    if not curriculum_config.get('enabled', False):
        return None, None, None, 1, 0.0, False
    
    # Calculer la distance du dernier batch (= distance globale)
    if len(batch_metrics) < 1:
        current_distance = 0.0
    else:
        last_distance = batch_metrics[-1]['mean_distance']
        
        # Filtrer les valeurs NaN ou invalides
        if last_distance != last_distance or last_distance is None or last_distance < -10:
            current_distance = 0.0
            print(f"WARNING: Last batch distance is invalid ({last_distance}), using 0.0m for curriculum")
        else:
            current_distance = last_distance
            
        # Debug seulement si problème
        if current_distance == 0.0 and len(batch_metrics) >= 1:
            print(f"DEBUG: current_distance=0! last_distance={last_distance}")
            print(f"DEBUG: Last batch: batch_num={batch_metrics[-1]['batch_num']}")
    
    # Déterminer la phase actuelle
    phase_distance_history = getattr(get_curriculum_state, 'phase_distance_history', [])
    current_phase = len(phase_distance_history) + 1
    
    # Obtenir le nombre max de phases depuis le config
    bump_ratio_schedule = curriculum_config.get('bump_ratio_schedule', [])
    max_phase = len(bump_ratio_schedule) if bump_ratio_schedule else 3
    
    if current_phase > max_phase:
        current_phase = max_phase
    
    # Obtenir le seuil de distance pour passer à la phase suivante depuis le config
    phase_threshold = None
    for phase_config in bump_ratio_schedule:
        if phase_config['phase'] == current_phase:
            phase_threshold = phase_config.get('distance_threshold', None)
            break
    
    # Vérifier si on doit passer à la phase suivante
    phase_changed = False
    if current_phase < max_phase and phase_threshold is not None and current_distance >= phase_threshold:
        # Vérifier si on n'a pas déjà fait la transition pour cette phase
        if not hasattr(get_curriculum_state, 'phase_transition_done'):
            get_curriculum_state.phase_transition_done = set()
        
        # Calculer la nouvelle phase
        new_phase = len(phase_distance_history) + 2  # +2 car on va ajouter à l'historique
        if new_phase > max_phase:
            new_phase = max_phase
        
        # Vérifier si cette transition n'a pas déjà été faite
        if new_phase not in get_curriculum_state.phase_transition_done:
            # Transition vers la phase suivante
            phase_distance_history.append(current_distance)
            current_phase = len(phase_distance_history) + 1
            if current_phase > max_phase:
                current_phase = max_phase
            phase_changed = True
            get_curriculum_state.phase_transition_done.add(current_phase)  # Marquer cette phase comme traitée
            print(f"\n{'!'*70}")
            print(f"PHASE TRANSITION: Phase {current_phase-1} → Phase {current_phase}")
            print(f"  Trigger: Distance {current_distance:.1f}m ≥ threshold {phase_threshold}m")
            print(f"{'!'*70}\n")
    
    # Sauvegarder l'historique
    get_curriculum_state.phase_distance_history = phase_distance_history
    
    # Obtenir le ratio de bumps pour la phase actuelle
    current_bump_ratio = 0.0  # Défaut
    for phase_config in bump_ratio_schedule:
        if phase_config['phase'] == current_phase:
            current_bump_ratio = phase_config['bump_ratio']
            break
    
    # NOUVEAU: Tracker la distance maximale atteinte (irréversible)
    if not hasattr(get_curriculum_state, 'max_distance_achieved'):
        get_curriculum_state.max_distance_achieved = 0.0
    
    # Reset du palier max si changement de phase
    if phase_changed:
        # Nouvelle phase = reset du palier max
        get_curriculum_state.max_distance_achieved = 0.0
        print(f"  → Distance tracker reset for new phase {current_phase}")
    
    # Mettre à jour le palier max (ne peut qu'augmenter)
    if current_distance > get_curriculum_state.max_distance_achieved:
        get_curriculum_state.max_distance_achieved = current_distance
    
    # Random percentage fixé à 100% (toujours aléatoire)
    random_percentage = 1.0
    
    return random_percentage, current_bump_ratio, current_phase, current_distance, phase_changed


def get_curriculum_random_percentage(config, iteration, batch_metrics=None):
    """Obtenir le pourcentage de corridors aléatoires selon le curriculum (compatibilité)."""
    if batch_metrics is None:
        return None
    random_percentage, _, _, _, _ = get_curriculum_state(config, batch_metrics)
    return random_percentage


def get_curriculum_bump_ratio(config, iteration, batch_metrics=None):
    """Obtenir le ratio de bumps selon le curriculum."""
    if batch_metrics is None:
        return None
    _, bump_ratio, _, _, _ = get_curriculum_state(config, batch_metrics)
    return bump_ratio


def get_curriculum_max_steps(config, iteration, batch_metrics=None):
    """Obtenir le nombre max de steps selon le curriculum - DEPRECATED, utiliser environment.max_steps."""
    # Cette fonction n'est plus utilisée, max_steps est fixe dans environment
    return None


def update_curriculum(envs, debug_env, iteration, num_iterations, config=None, batch_metrics=None):
    """Curriculum learning basé sur la distance moyenne des 2 derniers batches.
    
    Note: batch_metrics doit contenir les métriques combinées (principales + temp).
    """
    
    # Vérifier si les batch_metrics ont changé depuis la dernière fois
    current_batch_count = len(batch_metrics) if batch_metrics else 0
    last_batch_count = getattr(update_curriculum, 'last_batch_count', -1)
    
    # Sauvegarder l'état précédent pour détecter les changements
    prev_cached = getattr(update_curriculum, 'cached_curriculum_values', None)
    
    # Vérifier aussi le numéro du dernier batch pour être sûr
    last_batch_num = batch_metrics[-1]['batch_num'] if batch_metrics else -1
    prev_last_batch_num = getattr(update_curriculum, 'last_batch_num', -1)
    
    # Ne mettre à jour le curriculum que si de nouveaux batches sont disponibles
    # Vérifier à la fois le count ET le numéro du dernier batch
    if current_batch_count <= last_batch_count and last_batch_num <= prev_last_batch_num:
        # Pas de nouveaux batches, utiliser les dernières valeurs calculées
        cached_values = getattr(update_curriculum, 'cached_curriculum_values', (None, None, None, None, None, False))
        random_percentage, bump_ratio, current_steps, current_phase, current_distance, phase_changed = cached_values
        
        # Valeurs par défaut si pas de cache
        if current_steps is None:
            if config and 'environment' in config:
                current_steps = config['environment'].get('max_steps', 7000)
            else:
                current_steps = 7000
        
        # AFFICHER L'ÉTAT ACTUEL même sans changement
        # Note: on utilise les valeurs du cache car pas de nouveau batch
        print(f"CURRICULUM STATE (Phase {current_phase}):")
        print(f"  Distance: {current_distance:.1f}m")
        print(f"  Random corridors: {random_percentage*100:.0f}% | Max steps: {current_steps} | Obstacles: holes + {int(bump_ratio*100)}% bumps")
        print(f"  → No new batch since last check (batch #{last_batch_num})")
        print(f"{'='*70}")
        
        return current_steps, random_percentage, bump_ratio, phase_changed
    
    # Nouveaux batches disponibles, recalculer le curriculum
    update_curriculum.last_batch_count = current_batch_count
    update_curriculum.last_batch_num = last_batch_num
    
    # Obtenir l'état du curriculum basé sur la distance
    random_percentage, bump_ratio, current_phase, current_distance, phase_changed = get_curriculum_state(config, batch_metrics or [])
    
    # max_steps est maintenant fixe depuis environment config
    if config and 'environment' in config:
        current_steps = config['environment'].get('max_steps', 7000)
    else:
        current_steps = 7000
    
    # Sauvegarder les valeurs calculées (inclure current_distance dans le cache)
    update_curriculum.cached_curriculum_values = (random_percentage, bump_ratio, current_steps, current_phase, current_distance, phase_changed)
    
    # Calculer la distance moyenne pour l'affichage
    avg_distance = 0.0
    if batch_metrics and len(batch_metrics) >= 1:
        last_distance = batch_metrics[-1]['mean_distance']
        # Filtrer les valeurs NaN ou invalides
        if last_distance == last_distance and last_distance is not None and last_distance >= -10:  # last_distance == last_distance détecte non-NaN
            avg_distance = last_distance
    
    # === LOGS AMÉLIORÉS ===
    # Détecter les changements de paliers
    prev_random = prev_cached[0] if prev_cached else None
    prev_steps = prev_cached[2] if prev_cached else None
    prev_phase = prev_cached[3] if prev_cached else None
    
    random_changed = prev_random is not None and random_percentage != prev_random
    steps_changed = prev_steps is not None and current_steps != prev_steps
    
    # Afficher l'état actuel du curriculum
    print(f"CURRICULUM STATE (Phase {current_phase}):")
    print(f"  Distance: {current_distance:.1f}m")
    print(f"  Random corridors: {random_percentage*100:.0f}% | Max steps: {current_steps} | Obstacles: holes + {int(bump_ratio*100)}% bumps")
    
    # Afficher les changements détectés
    if phase_changed:
        print(f"  ✓ PHASE CHANGE: Phase {prev_phase} → Phase {current_phase}")
        print(f"    Reason: Distance {current_distance:.1f}m reached phase threshold")
        print(f"    → Paliers reset (random & steps)")
    elif random_changed or steps_changed:
        print(f"  ✓ PALIER CHANGE:")
        if random_changed:
            print(f"    Random: {prev_random*100:.0f}% → {random_percentage*100:.0f}%")
        if steps_changed:
            print(f"    Max steps: {prev_steps} → {current_steps}")
        print(f"    Reason: Distance {current_distance:.1f}m reached new threshold")
    else:
        print(f"  → No change (distance not sufficient for next threshold)")
    
    print(f"{'='*70}")
    
    # Update max_steps for all environments
    if hasattr(envs, 'envs'):
        for env_wrapper in envs.envs:
            if hasattr(env_wrapper, 'env') and hasattr(env_wrapper.env, 'set_max_steps'):
                env_wrapper.env.set_max_steps(current_steps)
            elif hasattr(env_wrapper, 'set_max_steps'):
                env_wrapper.set_max_steps(current_steps)
    
    # Update curriculum parameters for environments that support it
    if hasattr(envs, 'call'):
        # Pour AsyncVectorEnv, utiliser la méthode call pour envoyer des commandes aux processus
        try:
            envs.call('update_curriculum_params', 
                     random_percentage=random_percentage,
                     bump_ratio=bump_ratio,
                     max_steps=current_steps)
        except Exception as e:
            # Fallback to old method
            if hasattr(envs, 'envs'):
                for i, env_wrapper in enumerate(envs.envs):
                    env = env_wrapper.env if hasattr(env_wrapper, 'env') else env_wrapper
                    if hasattr(env, 'update_curriculum_params'):
                        env.update_curriculum_params(
                            random_percentage=random_percentage,
                            bump_ratio=bump_ratio,
                            max_steps=current_steps
                        )
    elif hasattr(envs, 'envs'):
        for i, env_wrapper in enumerate(envs.envs):
            env = env_wrapper.env if hasattr(env_wrapper, 'env') else env_wrapper
            if hasattr(env, 'update_curriculum_params'):
                env.update_curriculum_params(
                    random_percentage=random_percentage,
                    bump_ratio=bump_ratio,
                    max_steps=current_steps
                )
    
    # Update debug environment too
    if hasattr(debug_env, 'set_max_steps'):
        debug_env.set_max_steps(current_steps)
    if hasattr(debug_env, 'update_curriculum_params'):
        debug_env.update_curriculum_params(
            random_percentage=random_percentage,
            bump_ratio=bump_ratio,
            max_steps=current_steps
        )
    
    return current_steps, random_percentage, bump_ratio, phase_changed


def debug_render_episode(agent, debug_env, device, max_steps=None, current_bump_ratio=None, show_vision=True):
    """Render un épisode de debug pour voir ce qui se passe."""
    print("\nDEBUG: Rendering episode visualization...")
    
    # Si un bump_ratio spécifique est fourni, créer un nouvel environnement temporaire
    if current_bump_ratio is not None:
        print(f"DEBUG: Creating temporary environment with bump_ratio={current_bump_ratio}")
        from corridor_env import CorridorEnv
        temp_debug_env = CorridorEnv(max_steps=max_steps or 3000, corridor_xml=None, obstacle_type="holes", use_fixed_seed=True, random_percentage=None)
        temp_debug_env.bump_ratio = current_bump_ratio
        env_to_use = temp_debug_env
    else:
        env_to_use = debug_env
    
    # Reset AVANT de créer le viewer (génère nouveau corridor + nouveau modèle)
    obs, _ = env_to_use.reset()
    
    # Maintenant utiliser le nouveau modèle/data
    m = env_to_use.model
    d = env_to_use.data
    robot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
    
    # Préparer la visualisation de la vision CNN si demandé
    vision_queue = None
    vision_window = None
    
    if show_vision:
        import queue
        import tkinter as tk
        from PIL import Image, ImageTk
        
        vision_queue = queue.Queue(maxsize=2)
        vision_window = VisionWindow(title='Vision CNN en temps réel', vision_queue=vision_queue)
    
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
                max_steps = env_to_use.max_steps
            
            # Log initial dans la fenêtre
            if show_vision and vision_queue:
                try:
                    vision_queue.put_nowait(f"Position initiale: x={env_to_use.data.qpos[0]:.2f}, y={env_to_use.data.qpos[1]:.2f}")
                    vision_queue.put_nowait(f"Max steps: {max_steps}")
                except queue.Full:
                    pass
            else:
                print(f"Position initiale: x={env_to_use.data.qpos[0]:.2f}, y={env_to_use.data.qpos[1]:.2f}")
                print(f"Max steps pour cet épisode: {max_steps}")
            
            while not done and v.is_running() and step < max_steps:
                # Action de l'agent
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    action, _, _, _ = agent.get_action_and_value(obs_t)
                    action = action.cpu().numpy()[0]
                
                obs, reward, term, trunc, info = env_to_use.step(action)
                done = term or trunc
                ep_return += reward
                step += 1
                
                # Mettre à jour la vision CNN
                if show_vision and vision_queue and step % 5 == 0:
                    try:
                        grid = obs[7+env_to_use.history_dim:].reshape(env_to_use.grid_rows, env_to_use.grid_cols, 2)
                        try:
                            vision_queue.put_nowait((grid, env_to_use))
                        except queue.Full:
                            pass
                        # Update tkinter
                        vision_window.update()
                    except Exception as e:
                        print(f"Warning: Vision error: {e}")
                
                # Afficher info
                if step % 25 == 0:
                    x = env_to_use.data.qpos[0]
                    stabilizing = " (STABILISATION)" if step < env_to_use.stabilization_steps else ""
                    log_msg = f"Step {step}: x={x:.2f}m, reward={reward:.3f}, return={ep_return:.1f}{stabilizing}"
                    
                    # Envoyer à la fenêtre ou au terminal
                    if show_vision and vision_queue:
                        try:
                            vision_queue.put_nowait(log_msg)
                        except queue.Full:
                            pass
                    else:
                        print(log_msg)
                
                v.sync()
                time.sleep(0.05)  # 20 FPS
            
            final_x = env_to_use.data.qpos[0]
            reason = info.get('reason', 'truncated')
            corridor_type = info.get('corridor_type', 'unknown')
            is_random = info.get('is_random', False)
            random_str = "random" if is_random else "fixed"
            final_msg = f"Episode ended: {reason:<9} | Steps: {step:>4} | Distance: {final_x:>5.2f}m | Reward: {ep_return:>5.1f} | Corridor: {corridor_type}-{random_str}"
            
            # Envoyer à la fenêtre ou au terminal
            if show_vision and vision_queue:
                try:
                    vision_queue.put_nowait("="*60)
                    vision_queue.put_nowait(final_msg)
                    vision_queue.put_nowait("="*60)
                except queue.Full:
                    pass
            else:
                print(final_msg)
            
            # Attendre un peu pour voir le résultat
            time.sleep(2.0)
            
    except Exception as e:
        print(f"Erreur render: {e}")
        print("Continuons sans render...")
    finally:
        # Fermer la fenêtre tkinter
        if vision_queue is not None:
            try:
                vision_queue.put_nowait(None)
            except:
                pass
        if vision_window is not None:
            try:
                vision_window.root.destroy()
            except:
                pass




def train(config_path="config.yaml", rollback=False, **kwargs):
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
    
    # Ouvrir fichier de log des épisodes
    episodes_log_file = open("episodes_log.txt", "a")
    episodes_log_file.write(f"\n{'='*70}\n")
    episodes_log_file.write(f"TRAINING SESSION - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    episodes_log_file.write(f"{'='*70}\n")
    episodes_log_file.flush()
    
    # Curriculum initial (distance 0)
    initial_random_percentage, initial_bump_ratio, initial_phase, _, _ = get_curriculum_state(config, [])
    
    # max_steps vient de environment config (fixe)
    initial_max_steps = params.get('max_steps', 7000)
    
    if initial_random_percentage is not None:
        print(f"CURRICULUM: Starting with {initial_random_percentage*100:.0f}% random corridors")
    print(f"CURRICULUM: Max steps per episode: {initial_max_steps}")
    if initial_bump_ratio is not None:
        print(f"CURRICULUM: Starting with holes + {int(initial_bump_ratio*100)}% bumps")
    
    # Environnements parallèles avec curriculum complet
    envs = gym.vector.AsyncVectorEnv([make_env(config, initial_random_percentage, initial_max_steps, initial_bump_ratio) for _ in range(num_envs)])
    
    # Environnement de debug pour visualisation
    if config and 'environment' in config:
        env_config = config['environment']
        debug_max_steps = initial_max_steps if initial_max_steps else env_config.get('max_steps', 1000)
        use_random = env_config.get('use_random_corridor', True)
        corridor_xml_file = env_config.get('corridor_xml', 'corridor_3x100_no_full_obstacles.xml')
        
        # Pour le debug, utiliser le curriculum aussi
        if use_random and initial_random_percentage is not None:
            debug_use_random = np.random.random() < initial_random_percentage
            debug_corridor_xml = None if debug_use_random else corridor_xml_file
        else:
            debug_corridor_xml = None if use_random else corridor_xml_file
    else:
        debug_max_steps = initial_max_steps if initial_max_steps else 1000
        debug_corridor_xml = None  # Génération aléatoire
    
    debug_bump_ratio = initial_bump_ratio if initial_bump_ratio is not None else 0.0
    debug_corridor_xml = None  # Toujours utiliser le générateur
    debug_use_fixed_seed = True  # Seed fixe pour debug
    
    debug_env = CorridorEnv(max_steps=debug_max_steps, corridor_xml=debug_corridor_xml, obstacle_type="holes", use_fixed_seed=debug_use_fixed_seed, random_percentage=initial_random_percentage)
    debug_env.bump_ratio = debug_bump_ratio
    
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]
    
    # Agent avec configuration
    agent = Agent(obs_dim, act_dim, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=optimizer_eps)
    
    # DÉTECTION ET CHARGEMENT DE MODÈLE EXISTANT
    start_iteration = 1
    global_step = 0
    last_batch_episode = 0  # Sera mis à jour si checkpoint trouvé
    
    # Chercher le dernier checkpoint
    checkpoint_info = find_latest_checkpoint("models")
    
    if checkpoint_info:
        checkpoint_path = checkpoint_info
        print(f"RESUME: Found checkpoint: {checkpoint_path}")
        
        # Charger le checkpoint
        loaded = load_checkpoint(checkpoint_path, agent, optimizer, device)
        
        if loaded:
            start_iteration = loaded['iteration'] + 1
            global_step = loaded['global_step']
            last_batch_episode = loaded.get('last_episode', 0)  # Récupérer le dernier épisode
            
            print(f"   Resuming from iteration {loaded['iteration']}, global_step {global_step:,}")
            
            # Nettoyer les fichiers temp (ils ont déjà été fusionnés au checkpoint précédent)
            temp_files_to_clean = [
                "models/temp_training_metrics.csv",
                "models/temp_episodes_log.txt"
            ]
            
            for file_path in temp_files_to_clean:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'w') as f:
                            pass
                        print(f"   Cleaned temp file: {file_path}")
                    except Exception as e:
                        print(f"   WARNING: Could not clean {file_path}: {e}")
    else:
        print("FRESH START: No checkpoint found")
        
        # Nettoyer les fichiers existants
        files_to_clean = [
            "models/temp_training_metrics.csv",
            "models/temp_episodes_log.txt",
            "models/iteration_summary.csv",
            "models/training_metrics.csv"
        ]
        
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                try:
                    # Tronquer le fichier (vider son contenu) au lieu de le supprimer
                    # Cela fonctionne même si le fichier est ouvert dans l'éditeur
                    with open(file_path, 'w') as f:
                        pass  # Juste ouvrir en mode 'w' vide le fichier
                    print(f"   Cleaned: {file_path}")
                except Exception as e:
                    print(f"   WARNING: Could not clean {file_path}: {e}")
    
    # Buffers GPU
    obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    terminateds = torch.zeros((num_steps, num_envs), device=device)  # Pour GAE correct
    values = torch.zeros((num_steps, num_envs), device=device)
    
    # Init
    if 'global_step' not in locals():
        global_step = 0  # Initialiser si pas déjà fait lors du chargement
    start_time = time.time()
    # Au redémarrage, ne pas utiliser le seed fixe pour avoir de la variété
    if start_iteration > 1:
        next_obs, _ = envs.reset()  # Pas de seed = aléatoire
    else:
        next_obs, _ = envs.reset(seed=seed)  # Premier démarrage = seed fixe
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)
    next_terminated = torch.zeros(num_envs, device=device)
    
    # Métriques par batch - Plus besoin de charger, on utilise les fichiers CSV directement
    # last_batch_episode est déjà initialisé lors du chargement du checkpoint
    
    # Calculer le dernier batch_num depuis les fichiers existants (une seule fois au démarrage)
    last_batch_num = get_last_batch_num()
    
    # Stats globales (pour best values)
    best_return = -float('inf')
    best_distance = 0.0
    successes = 0
    total_episodes = last_batch_episode  # Continue from last saved episode
    
    # Tracker pour les stats de l'itération courante (affichage)
    iteration_tracker = IterationTracker()
    
    # Listes pour les batches (calcul des métriques CSV par batch de 20)
    batch_episode_returns = []
    batch_episode_distances = []
    batch_episode_steps = []
    batch_episode_reasons = []
    
    os.makedirs("models", exist_ok=True)
    
    # Initialiser les variables du curriculum (seront mises à jour après chaque batch)
    current_max_steps = config['environment'].get('max_steps', 4000) if config and 'environment' in config else 4000
    
    # Initialiser avec les valeurs du curriculum de départ
    if config and 'curriculum' in config and config['curriculum'].get('enabled', False):
        curriculum_config = config['curriculum']
        # Phase 1 par défaut
        bump_ratio_schedule = curriculum_config.get('bump_ratio_schedule', [])
        if bump_ratio_schedule and len(bump_ratio_schedule) > 0:
            bump_ratio = bump_ratio_schedule[0].get('bump_ratio', 0.5)
        else:
            bump_ratio = 0.5
        random_percentage = 1.0  # 100% random au départ
    else:
        random_percentage = 1.0
        bump_ratio = 0.5
    
    current_phase = 1  # Phase initiale

    iteration = start_iteration
    while iteration <= num_iterations:
        # === UPDATE CURRICULUM (début d'itération) ===
        # Le curriculum est maintenant géré via les temp_metrics
        
        # === COLLECTE ROLLOUTS (parallèle sur num_envs) ===
        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            terminateds[step] = next_terminated
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()
            
            # Step tous les envs en parallèle
            next_obs_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            
            # IMPORTANT: Pour GAE, on doit distinguer terminated (vraie fin) et truncated (timeout)
            # - terminated: nextnonterminal = 0 (pas de bootstrap)
            # - truncated: nextnonterminal = 1 (bootstrap avec value function)
            next_done_np = np.logical_or(term, trunc)  # Pour reset des envs
            next_terminated_np = term  # Pour GAE (seulement les vraies terminaisons)
            
            rewards[step] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)
            next_terminated = torch.tensor(next_terminated_np, dtype=torch.float32, device=device)
            
            # Log épisodes terminés - LOGIQUE CORRIGÉE
            for i in range(num_envs):
                if next_done_np[i]:  # Cet environnement s'est terminé
                    # Récupérer infos de manière simple et robuste
                    try:
                        # Essayer d'abord le format simple (liste d'infos)
                        if isinstance(infos, list) and i < len(infos) and infos[i]:
                            info = infos[i]
                            reason = info.get('reason', 'truncated' if trunc[i] else 'unknown')
                            raw_dist = info.get('x', 0)
                            # Valider que la distance n'est pas NaN ou invalide
                            if raw_dist != raw_dist or raw_dist is None or raw_dist < -10:  # raw_dist != raw_dist détecte NaN
                                dist = 0.0
                                print(f"  WARNING: Invalid distance {raw_dist} for env {i}, using 0.0")
                            else:
                                dist = float(raw_dist)
                            ret = info.get('episode', {}).get('r', reward[i] if i < len(reward) else 0)
                            steps = info.get('step', 0)
                            corridor_type = info.get('corridor_type', 'unknown')
                            is_random = info.get('is_random', False)
                        
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
                                raw_dist = x_list[i]
                                # Valider que la distance n'est pas NaN ou invalide
                                if raw_dist != raw_dist or raw_dist is None or raw_dist < -10:  # raw_dist != raw_dist détecte NaN
                                    dist = 0.0
                                    print(f"  WARNING: Invalid distance {raw_dist} for env {i}, using 0.0")
                                else:
                                    dist = float(raw_dist)
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
                            
                            # Récupérer corridor type et random info
                            corridor_type_list = infos.get('corridor_type', [])
                            corridor_type_mask = infos.get('_corridor_type', [])
                            if i < len(corridor_type_list) and i < len(corridor_type_mask) and corridor_type_mask[i]:
                                corridor_type = corridor_type_list[i]
                            else:
                                corridor_type = 'unknown'
                            
                            is_random_list = infos.get('is_random', [])
                            is_random_mask = infos.get('_is_random', [])
                            if i < len(is_random_list) and i < len(is_random_mask) and is_random_mask[i]:
                                is_random = is_random_list[i]
                            else:
                                is_random = False
                            
                            # Récupérer corridor seed
                            corridor_seed_list = infos.get('corridor_seed', [])
                            corridor_seed_mask = infos.get('_corridor_seed', [])
                            if i < len(corridor_seed_list) and i < len(corridor_seed_mask) and corridor_seed_mask[i]:
                                corridor_seed = corridor_seed_list[i]
                            else:
                                corridor_seed = -1
                        
                        else:
                            # Fallback complet
                            reason = 'truncated' if trunc[i] else 'unknown'
                            dist = 0.0
                            ret = float(reward[i]) if i < len(reward) else 0.0
                            steps = 0
                            corridor_type = 'unknown'
                            is_random = False
                            corridor_seed = -1
                            
                    except (IndexError, KeyError, TypeError, AttributeError) as e:
                        # Fallback en cas d'erreur
                        reason = 'truncated' if trunc[i] else 'terminated'
                        dist = 0.0
                        ret = float(reward[i]) if i < len(reward) else 0.0
                        steps = 0
                        corridor_type = 'unknown'
                        is_random = False
                        corridor_seed = -1
                        print(f"  WARNING: Episode info extraction failed for env {i}: {e}")
                    
                    # Créer un dict info pour le log
                    info = {
                        'corridor_type': corridor_type,
                        'is_random': is_random,
                        'corridor_seed': corridor_seed
                    }
                    
                    # Ajouter l'épisode au tracker de l'itération (pour affichage)
                    iteration_tracker.add_episode(ret, dist, steps, reason)
                    
                    # Ajouter aux listes batch (pour calcul métriques CSV)
                    batch_episode_returns.append(ret)
                    batch_episode_distances.append(dist)
                    batch_episode_steps.append(steps)
                    batch_episode_reasons.append(reason)
                    
                    # Incrémenter AVANT de sauvegarder pour avoir le bon numéro
                    total_episodes += 1
                    episode_num = total_episodes
                    
                    # Sauvegarder dans le log temp
                    save_episode_to_temp_log(episode_num, ret, dist, steps, reason)
                    
                    # Log individuel pour chaque épisode avec type de corridor (console uniquement)
                    corridor_type = info.get('corridor_type', 'unknown')
                    is_random = info.get('is_random', False)
                    corridor_seed = info.get('corridor_seed', -1)
                    random_str = "random" if is_random else "fixed"
                    log_line = f"Episode {episode_num:>4}: {reason:<11} | Steps: {steps:>4} | Distance: {dist:>6.2f}m | Reward: {ret:>6.1f} | Corridor: {corridor_type}+{random_str} | Seed: {corridor_seed}"
                    print(log_line)
                    
                    if ret > best_return:
                        best_return = ret
                    if dist > best_distance:
                        best_distance = dist
                    
                    # Compter les succès globaux
                    if reason == "success":
                        successes += 1
        
        # === GAE ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgae = 0
            
            # GAE avec distinction terminated vs truncated
            # - terminated: nextnonterminal = 0 (pas de bootstrap, vraie fin)
            # - truncated: nextnonterminal = 1 (bootstrap avec value, timeout artificiel)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated  # Utiliser terminated, pas done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminateds[t + 1]  # Utiliser terminated, pas done
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
                
                # CORRECTION: Ajuster les indices pour les listes en mémoire
                # Les listes batch_episode_*, etc. commencent à 0
                # mais batch_start/batch_end sont des numéros d'épisodes absolus
                memory_start = len(batch_episode_returns) - (total_episodes - batch_start)
                memory_end = len(batch_episode_returns) - (total_episodes - batch_end)
                
                # Sécurité : s'assurer que les indices sont valides
                memory_start = max(0, memory_start)
                memory_end = min(len(batch_episode_returns), memory_end)
                
                # Calculer le numéro du batch depuis les temp metrics + dernier batch_num
                temp_metrics_count = len(load_temp_metrics())
                batch_num = last_batch_num + temp_metrics_count + 1
                
                if memory_end <= memory_start:
                    print(f"  ERROR: Invalid memory indices {memory_start}-{memory_end} for batch {batch_num}")
                    break
                
                # Calculer moyennes pour ce batch de 20 épisodes
                batch_returns = batch_episode_returns[memory_start:memory_end]
                batch_distances = batch_episode_distances[memory_start:memory_end]
                batch_steps_list = batch_episode_steps[memory_start:memory_end]
                batch_reasons = batch_episode_reasons[memory_start:memory_end]
                
                # Valider les distances avant calcul de la moyenne
                valid_distances = [d for d in batch_distances if d == d and d is not None and d >= -10]  # d == d détecte non-NaN
                if len(valid_distances) == 0:
                    mean_distance = 0.0
                    print(f"  WARNING: No valid distances in batch {batch_num}, using 0.0")
                else:
                    mean_distance = np.mean(valid_distances)
                    if len(valid_distances) < len(batch_distances):
                        invalid_count = len(batch_distances) - len(valid_distances)
                        print(f"  WARNING: {invalid_count} invalid distances filtered out in batch {batch_num}")
                
                # Valider les returns avant calcul de la moyenne
                valid_returns = [r for r in batch_returns if r == r and r is not None]  # r == r détecte non-NaN
                if len(valid_returns) == 0:
                    mean_return = 0.0
                    print(f"  WARNING: No valid returns in batch {batch_num}, using 0.0")
                else:
                    mean_return = np.mean(valid_returns)
                    if len(valid_returns) < len(batch_returns):
                        invalid_count = len(batch_returns) - len(valid_returns)
                        print(f"  WARNING: {invalid_count} invalid returns filtered out in batch {batch_num}")
                
                # Valider les steps avant calcul de la moyenne
                valid_steps = [s for s in batch_steps_list if s == s and s is not None and s >= 0]  # s == s détecte non-NaN
                if len(valid_steps) == 0:
                    mean_survival = 0.0
                    print(f"  WARNING: No valid steps in batch {batch_num}, using 0.0")
                else:
                    mean_survival = np.mean(valid_steps)
                    if len(valid_steps) < len(batch_steps_list):
                        invalid_count = len(batch_steps_list) - len(valid_steps)
                        print(f"  WARNING: {invalid_count} invalid steps filtered out in batch {batch_num}")
                
                # Compter succès dans ce batch basé sur les RAISONS, pas les distances
                batch_successes = sum(1 for reason in batch_reasons if reason == 'success')
                
                # Créer le nouveau batch metric avec les valeurs curriculum actuelles
                new_batch_metric = {
                    'batch_num': batch_num,
                    'episode_end': batch_end,  # Numéro du dernier épisode de ce batch
                    'episodes_range': f"{batch_start+1}-{batch_end}",
                    'global_step': global_step,
                    'mean_return': mean_return,
                    'mean_distance': mean_distance,
                    'mean_survival': mean_survival,
                    'success_rate': 100 * batch_successes / batch_size_metrics,
                    'current_phase': current_phase,
                    'random_percentage': random_percentage if random_percentage is not None else 1.0,
                    'bump_ratio': bump_ratio if bump_ratio is not None else 0.5,
                }
                
                # Sauvegarder immédiatement dans le CSV temporaire
                save_temp_batch_to_csv(new_batch_metric, "models/temp_training_metrics.csv")
                
                last_batch_episode = batch_end
                print(f"BATCH: Batch {batch_num} completed (episodes {batch_start+1}-{batch_end}) - saved to temp CSV")
            
            # Calculer la distance moyenne de l'itération courante pour le curriculum
            iteration_mean_distance = 0.0
            if iteration_tracker.has_episodes():
                stats = iteration_tracker.get_stats()
                iter_distances = stats['distances']
                if iter_distances:
                    iteration_mean_distance = np.mean(iter_distances)
            
            # Mettre à jour le curriculum basé sur la distance de l'itération
            # On crée un batch_metric temporaire avec la distance de l'itération
            temp_batch_for_curriculum = [{
                'mean_distance': iteration_mean_distance,
                'batch_num': 0  # Pas utilisé pour le curriculum
            }]
            
            random_percentage, bump_ratio, current_phase, current_distance, phase_changed = get_curriculum_state(config, temp_batch_for_curriculum)
            
            # Log de vérification du curriculum
            print(f"\nCURRICULUM CHECK:")
            print(f"  Iteration mean distance: {iteration_mean_distance:.2f}m")
            print(f"  Current phase: {current_phase}")
            
            # Obtenir le seuil pour la phase actuelle
            bump_ratio_schedule = config['curriculum'].get('bump_ratio_schedule', [])
            phase_threshold = None
            for phase_config in bump_ratio_schedule:
                if phase_config['phase'] == current_phase:
                    phase_threshold = phase_config.get('distance_threshold', None)
                    break
            
            if phase_threshold is not None:
                print(f"  Phase {current_phase} threshold: {phase_threshold}m")
                if iteration_mean_distance >= phase_threshold:
                    print(f"  → Distance {iteration_mean_distance:.2f}m >= threshold {phase_threshold}m (ready for next phase)")
                else:
                    print(f"  → Distance {iteration_mean_distance:.2f}m < threshold {phase_threshold}m (need {phase_threshold - iteration_mean_distance:.2f}m more)")
            else:
                print(f"  → Phase {current_phase} is final phase (no threshold)")
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps:,} | Time: {elapsed:.0f}s")
            if current_phase is not None:
                print(f"Curriculum Phase: {current_phase}")
            print(f"Max Steps: {current_max_steps}")
            if random_percentage is not None:
                print(f"Random Corridors: {random_percentage*100:.0f}%")
            if bump_ratio is not None:
                print(f"Obstacles: holes + {int(bump_ratio*100)}% bumps")
            print(f"{'='*70}")
            
            # Afficher les stats de l'itération courante
            if iteration_tracker.has_episodes():
                stats = iteration_tracker.get_stats()
                iter_returns = stats['returns']
                iter_distances = stats['distances']
                iter_steps = stats['steps']
                iter_terminations = stats['termination_counts']
                
                success_rate = 100 * successes / max(1, total_episodes)
                print(f"EPISODES: {total_episodes} total | Success: {successes} ({success_rate:.1f}%)")
                
                # Afficher le nombre de batches depuis les temp metrics
                temp_metrics = load_temp_metrics()
                if temp_metrics:
                    print(f"BATCHES: {len(temp_metrics)} batches of {batch_size_metrics} episodes in temp")
                
                # Calculer min/max de l'itération
                iter_best_return = np.max(iter_returns)
                iter_best_distance = np.max(iter_distances)
                
                print(f"REWARD  : Mean {np.mean(iter_returns):>7.1f} +/- {np.std(iter_returns):>5.1f} | Max {iter_best_return:>7.1f}")
                print(f"DISTANCE: Mean {np.mean(iter_distances):>7.1f}m +/- {np.std(iter_distances):>5.1f}m | Max {iter_best_distance:>7.1f}m")
                print(f"SURVIVAL: Mean {np.mean(iter_steps):>7.0f} steps +/- {np.std(iter_steps):>5.0f}")
                
                # Récapitulatif terminaisons (seulement si > 0)
                active_reasons = {k: v for k, v in iter_terminations.items() if v > 0}
                if active_reasons:
                    reasons_str = " | ".join([f"{k}:{v}" for k, v in active_reasons.items()])
                    print(f"TERMINATIONS: {reasons_str}")
            else:
                print(f"WARNING: No episodes completed in this iteration")
            
            print(f"{'='*70}")
            
            # Réinitialiser le tracker pour la prochaine itération
            iteration_tracker.reset()
        
        # Sauvegarde périodique
        if iteration % save_interval == 0:
            print(f"\n{'='*70}")
            print(f"SAVE CHECK - Iteration {iteration}")
            print(f"{'='*70}")
            
            # Comparer distances
            current_distance = get_mean_distance_from_temp()
            last_distance = load_last_iteration_summary()
            
            print(f"Current distance: {current_distance:.2f}m" if current_distance else "Current distance: None")
            print(f"Last saved distance: {last_distance:.2f}m" if last_distance else "Last saved distance: None (first save)")
            
            # Vérifier si on doit sauvegarder
            should_save = last_distance is None or (current_distance is not None and current_distance >= last_distance)
            
            if not should_save:
                print(f"\nREJECTED: Current distance ({current_distance:.2f}m) < Last saved ({last_distance:.2f}m)")
                
                if rollback:
                    print(f"ROLLBACK: Loading last checkpoint and continuing training...")
                    
                    # Charger le dernier checkpoint
                    checkpoint_path = find_latest_checkpoint()
                    if checkpoint_path:
                        checkpoint = load_checkpoint(agent, optimizer, checkpoint_path)
                        iteration = checkpoint['iteration']
                        global_step = checkpoint['global_step']
                        total_episodes = checkpoint['total_episodes']
                        
                        # Nettoyer les fichiers temp
                        temp_metrics_path = os.path.join(models_dir, "temp_training_metrics.csv")
                        temp_episodes_path = os.path.join(models_dir, "temp_episodes_log.txt")
                        
                        if os.path.exists(temp_metrics_path):
                            with open(temp_metrics_path, 'w') as f:
                                pass
                            print(f"Cleaned temp file: {temp_metrics_path}")
                        
                        if os.path.exists(temp_episodes_path):
                            with open(temp_episodes_path, 'w') as f:
                                pass
                            print(f"Cleaned temp file: {temp_episodes_path}")
                        
                        # Recharger last_batch_num
                        last_batch_num = get_last_batch_num()
                        
                        print(f"Rolled back to iteration {iteration}, continuing training...")
                        print(f"{'='*70}\n")
                        
                        # Continuer la boucle sans sauvegarder
                        iteration += 1
                        continue
                    else:
                        print(f"WARNING: No checkpoint found for rollback, continuing without save...")
                        print(f"{'='*70}\n")
                        iteration += 1
                        continue
                else:
                    print(f"Continuing training without saving...")
                    print(f"{'='*70}\n")
                    iteration += 1
                    continue
            
            print(f"\nACCEPTED: Saving checkpoint...")
            
            # 1. Sauvegarder iteration summary
            temp_metrics = load_temp_metrics()
            if temp_metrics:
                last_episode_from_batch = temp_metrics[-1]['episode_end']
            else:
                last_episode_from_batch = total_episodes
            
            save_iteration_summary(iteration, global_step, last_episode_from_batch)
            
            # 2. Flush temp vers main
            flush_temp_to_main_metrics()
            
            # 2b. Flush temp episode logs
            flush_temp_episode_logs()
            
            # 3. Mettre à jour last_batch_num après le flush
            last_batch_num = get_last_batch_num()
            
            # 4. Plot training curves
            plot_training_curves(iteration=iteration)
            
            # 5. Save checkpoint (utiliser total_episodes pour avoir le vrai nombre)
            checkpoint_metrics = compute_checkpoint_metrics(temp_metrics) if temp_metrics else None
            
            # Récupérer l'état du curriculum
            curriculum_state = None
            if temp_metrics and len(temp_metrics) > 0:
                last_batch = temp_metrics[-1]
                curriculum_state = {
                    'current_phase': last_batch.get('current_phase', 1),
                    'random_percentage': last_batch.get('random_percentage', 0.0),
                    'bump_ratio': last_batch.get('bump_ratio', 0.0)
                }
            
            save_checkpoint(agent, optimizer, iteration, global_step, total_episodes, 
                          checkpoint_metrics, curriculum_state)
            
            print(f"{'='*70}\n")
        
        # Debug render
        if iteration % render_interval == 0 or iteration == 1:
            debug_render_episode(agent, debug_env, device, current_max_steps, bump_ratio)
        
        # Incrémenter l'itération pour la boucle while
        iteration += 1
    
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
    
    # Sauvegarder métriques finales
    flush_temp_to_main_metrics()
    plot_training_curves()  # Sans numéro d'itération pour le plot final
    
    # Le dernier checkpoint sauvegardé est déjà le modèle final
    last_checkpoint = find_latest_checkpoint("models")
    if last_checkpoint:
        print(f"\nModèle final: {last_checkpoint}")
    else:
        print(f"\nAucun checkpoint sauvegardé")
    
    print(f"{'='*70}\n")
    
    envs.close()
    return last_checkpoint


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
    parser.add_argument("--rollback", action="store_true", help="Activer le rollback automatique en cas de régression")
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
    
    train(config_path=args.config, rollback=args.rollback, **kwargs)
