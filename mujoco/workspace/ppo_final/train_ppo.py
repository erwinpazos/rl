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


def load_existing_metrics(metrics_file="models/training_metrics.csv"):
    """Charge les métriques existantes depuis le CSV si disponible."""
    import csv
    import os
    
    if not os.path.exists(metrics_file):
        return [], 0
    
    try:
        batch_metrics = []
        last_batch_episode = 0
        
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertir les valeurs en types appropriés avec validation
                try:
                    mean_distance = float(row['mean_distance'])
                    # Valider que la distance n'est pas NaN
                    if mean_distance != mean_distance or mean_distance is None:  # mean_distance != mean_distance détecte NaN
                        print(f"WARNING: Invalid mean_distance {row['mean_distance']} in CSV, using 0.0")
                        mean_distance = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_distance {row['mean_distance']} in CSV, using 0.0")
                    mean_distance = 0.0
                
                metrics = {
                    'batch_num': int(row['batch_num']),
                    'episode_end': int(row['episode_end']),
                    'episodes_range': row['episodes_range'],
                    'global_step': int(row['global_step']),
                    'mean_return': float(row['mean_return']),
                    'mean_distance': mean_distance,
                    'mean_survival': float(row['mean_survival']),
                    'success_rate': float(row['success_rate'])
                }
                batch_metrics.append(metrics)
                last_batch_episode = max(last_batch_episode, metrics['episode_end'])
        
        print(f"RESUME: Loaded {len(batch_metrics)} existing metric batches from {metrics_file}")
        print(f"   Last batch episode: {last_batch_episode}")
        return batch_metrics, last_batch_episode
        
    except Exception as e:
        print(f"WARNING: Failed to load existing metrics: {e}")
        return [], 0


def save_metrics_to_csv(batch_metrics, metrics_file="models/training_metrics.csv"):
    """Sauvegarde les métriques dans le CSV."""
    import csv
    
    if not batch_metrics:
        return
    
    # NOUVEAU: Éliminer les doublons basés sur batch_num avant sauvegarde
    seen_batch_nums = set()
    unique_metrics = []
    duplicates_found = 0
    
    for metrics in batch_metrics:
        batch_num = metrics['batch_num']
        if batch_num not in seen_batch_nums:
            seen_batch_nums.add(batch_num)
            unique_metrics.append(metrics)
        else:
            duplicates_found += 1
    
    if duplicates_found > 0:
        print(f"WARNING: Removed {duplicates_found} duplicate batches before saving CSV")
    
    # Trier par batch_num pour assurer l'ordre
    unique_metrics.sort(key=lambda x: x['batch_num'])
    
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                               'global_step', 'mean_return', 'mean_distance', 
                                               'mean_survival', 'success_rate'])
        writer.writeheader()
        
        for metrics in unique_metrics:
            writer.writerow(metrics)


def save_temp_batch_to_csv(batch_metric, temp_metrics_file="models/temp_training_metrics.csv"):
    """Sauvegarde un nouveau batch dans le CSV temporaire."""
    import csv
    import os
    
    if not batch_metric:
        return
    
    # Si le fichier temp n'existe pas, créer avec header
    file_exists = os.path.exists(temp_metrics_file)
    
    with open(temp_metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                               'global_step', 'mean_return', 'mean_distance', 
                                               'mean_survival', 'success_rate'])
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(batch_metric)


def load_temp_metrics(temp_metrics_file="models/temp_training_metrics.csv"):
    """Charge les métriques temporaires depuis le CSV temp."""
    import csv
    import os
    
    if not os.path.exists(temp_metrics_file):
        return []
    
    try:
        temp_metrics = []
        with open(temp_metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertir les valeurs en types appropriés avec validation
                try:
                    mean_distance = float(row['mean_distance'])
                    # Valider que la distance n'est pas NaN
                    if mean_distance != mean_distance or mean_distance is None:  # mean_distance != mean_distance détecte NaN
                        print(f"WARNING: Invalid mean_distance {row['mean_distance']} in temp CSV, using 0.0")
                        mean_distance = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_distance {row['mean_distance']} in temp CSV, using 0.0")
                    mean_distance = 0.0
                
                try:
                    mean_return = float(row['mean_return'])
                    # Valider que le return n'est pas NaN
                    if mean_return != mean_return or mean_return is None:  # mean_return != mean_return détecte NaN
                        print(f"WARNING: Invalid mean_return {row['mean_return']} in temp CSV, using 0.0")
                        mean_return = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_return {row['mean_return']} in temp CSV, using 0.0")
                    mean_return = 0.0
                
                try:
                    mean_survival = float(row['mean_survival'])
                    # Valider que la survival n'est pas NaN
                    if mean_survival != mean_survival or mean_survival is None:  # mean_survival != mean_survival détecte NaN
                        print(f"WARNING: Invalid mean_survival {row['mean_survival']} in temp CSV, using 0.0")
                        mean_survival = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_survival {row['mean_survival']} in temp CSV, using 0.0")
                    mean_survival = 0.0
                
                metrics = {
                    'batch_num': int(row['batch_num']),
                    'episode_end': int(row['episode_end']),
                    'episodes_range': row['episodes_range'],
                    'global_step': int(row['global_step']),
                    'mean_return': mean_return,
                    'mean_distance': mean_distance,
                    'mean_survival': mean_survival,
                    'success_rate': float(row['success_rate'])
                }
                temp_metrics.append(metrics)
        
        return temp_metrics
        
    except Exception as e:
        print(f"WARNING: Failed to load temp metrics: {e}")
        return []


def merge_and_sync_metrics(batch_metrics, temp_metrics_file="models/temp_training_metrics.csv", main_metrics_file="models/training_metrics.csv"):
    """Fusionne les métriques temp avec les principales et reset le temp."""
    import os
    
    # Charger les métriques temp
    temp_metrics = load_temp_metrics(temp_metrics_file)
    
    if temp_metrics:
        # NOUVEAU: Éviter les doublons en vérifiant les batch_num existants
        existing_batch_nums = set(batch['batch_num'] for batch in batch_metrics)
        
        # Ajouter seulement les métriques temp qui ne sont pas déjà présentes
        new_temp_metrics = []
        for temp_metric in temp_metrics:
            if temp_metric['batch_num'] not in existing_batch_nums:
                new_temp_metrics.append(temp_metric)
            else:
                print(f"WARNING: Skipping duplicate batch {temp_metric['batch_num']} from temp CSV")
        
        if new_temp_metrics:
            # Ajouter les nouvelles métriques temp aux principales
            batch_metrics.extend(new_temp_metrics)
            print(f"SYNC: Merged {len(new_temp_metrics)} new temp batches into main metrics")
        else:
            print(f"SYNC: No new temp batches to merge (all were duplicates)")
        
        # Sauvegarder tout dans le CSV principal
        save_metrics_to_csv(batch_metrics, main_metrics_file)
        
        # Supprimer le fichier temp
        try:
            os.remove(temp_metrics_file)
            print(f"SYNC: Temp metrics file cleared")
        except Exception as e:
            print(f"WARNING: Could not remove temp file: {e}")
    
    return batch_metrics


def get_combined_metrics(batch_metrics, temp_metrics_file="models/temp_training_metrics.csv"):
    """Obtient les métriques combinées (principales + temp) pour le curriculum."""
    temp_metrics = load_temp_metrics(temp_metrics_file)
    combined_metrics = batch_metrics + temp_metrics
    return combined_metrics


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
    
    # Obtenir les paramètres basés sur la distance avec paliers irréversibles
    random_percentage = get_curriculum_value_by_distance(
        curriculum_config.get('random_corridor_schedule', []), 
        current_distance, 
        'random_percentage', 
        0.2,
        get_curriculum_state.max_distance_achieved  # Palier irréversible
    )
    
    return random_percentage, current_bump_ratio, current_phase, current_distance, phase_changed


def get_curriculum_value_by_distance(schedule, distance, value_key, default_value, current_max_achieved=None):
    """Obtenir une valeur du curriculum basée sur la distance avec paliers irréversibles."""
    if not schedule:
        return default_value
    
    # Trouver le palier le plus élevé atteint
    max_achieved_value = default_value
    if current_max_achieved is not None:
        for stage in schedule:
            if current_max_achieved >= stage['distance']:
                max_achieved_value = stage[value_key]
    
    # Trouver le palier actuel basé sur la distance
    current_value = default_value
    for stage in schedule:
        if distance >= stage['distance']:
            current_value = stage[value_key]
        else:
            break
    
    # Retourner le maximum entre le palier actuel et le palier max déjà atteint (pas de régression)
    if current_max_achieved is not None:
        # Pour les valeurs qui augmentent (random_percentage, max_steps)
        if value_key in ['random_percentage', 'max_steps']:
            return max(current_value, max_achieved_value)
        else:
            return current_value
    else:
        return current_value


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
        
        class VisionWindow:
            def __init__(self):
                self.root = tk.Tk()
                self.root.title('Vision CNN en temps réel')
                self.root.geometry('1200x650')
                
                # Frame principal
                main_frame = tk.Frame(self.root)
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # Frame pour les 3 vues (en haut)
                vision_frame = tk.Frame(main_frame)
                vision_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
                
                # 3 colonnes pour les 3 vues
                self.frames = []
                self.labels = []
                self.titles = ['Canal 0 - Obstacles', 'Canal 1 - Trous', 'Vue Combinée']
                
                for i, title in enumerate(self.titles):
                    frame = tk.Frame(vision_frame)
                    frame.grid(row=0, column=i, padx=10, pady=10)
                    
                    title_label = tk.Label(frame, text=title, font=('Arial', 12, 'bold'))
                    title_label.pack()
                    
                    img_label = tk.Label(frame)
                    img_label.pack()
                    
                    self.frames.append(frame)
                    self.labels.append(img_label)
                
                # Frame pour les logs (en bas)
                log_frame = tk.Frame(main_frame)
                log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                log_title = tk.Label(log_frame, text='Episode Progress', font=('Arial', 12, 'bold'))
                log_title.pack()
                
                # Zone de texte avec scrollbar
                scrollbar = tk.Scrollbar(log_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                self.log_text = tk.Text(log_frame, height=8, yscrollcommand=scrollbar.set, 
                                       font=('Courier', 9), bg='black', fg='lime')
                self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=self.log_text.yview)
                
                # Timer pour checker la queue
                self.check_queue()
            
            def add_log(self, message):
                """Ajoute un message au log."""
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)  # Auto-scroll vers le bas
            
            def check_queue(self):
                try:
                    data = vision_queue.get_nowait()
                    if data is None:
                        self.root.quit()
                        return
                    
                    # Distinguer entre vision data et log message
                    if isinstance(data, str):
                        # C'est un message de log
                        self.add_log(data)
                    else:
                        # C'est des données de vision
                        grid, env_data = data
                        self.display_grid(grid, env_data)
                except queue.Empty:
                    pass
                finally:
                    self.root.after(50, self.check_queue)  # Check toutes les 50ms
            
            def display_grid(self, grid, env_data):
                rows, cols = grid.shape[0], grid.shape[1]
                robot_row = env_data.robot_row_in_grid
                robot_col = env_data.robot_col_in_grid
                
                # Canal 0 - Obstacles (rouge)
                img0 = self.grid_to_image(grid[:, :, 0], robot_row, robot_col, (255, 0, 0))
                photo0 = ImageTk.PhotoImage(img0.resize((380, 380), Image.NEAREST))
                self.labels[0].config(image=photo0)
                self.labels[0].image = photo0  # Garder référence
                
                # Canal 1 - Trous (bleu)
                img1 = self.grid_to_image(grid[:, :, 1], robot_row, robot_col, (0, 0, 255))
                photo1 = ImageTk.PhotoImage(img1.resize((380, 380), Image.NEAREST))
                self.labels[1].config(image=photo1)
                self.labels[1].image = photo1
                
                # Vue combinée
                img2 = self.grid_combined_to_image(grid, robot_row, robot_col)
                photo2 = ImageTk.PhotoImage(img2.resize((380, 380), Image.NEAREST))
                self.labels[2].config(image=photo2)
                self.labels[2].image = photo2
            
            def grid_to_image(self, channel, robot_row, robot_col, color):
                rows, cols = channel.shape
                img_data = np.zeros((rows, cols, 3), dtype=np.uint8)
                
                # Obstacles en couleur
                mask = channel > 0.5
                img_data[mask] = color
                img_data[~mask] = [255, 255, 255]
                
                # Robot en vert
                if 0 <= robot_row < rows and 0 <= robot_col < cols:
                    img_data[robot_row, robot_col] = [0, 255, 0]
                
                return Image.fromarray(img_data, 'RGB')
            
            def grid_combined_to_image(self, grid, robot_row, robot_col):
                rows, cols = grid.shape[0], grid.shape[1]
                img_data = np.ones((rows, cols, 3), dtype=np.uint8) * 255
                
                for i in range(rows):
                    for j in range(cols):
                        obstacle = grid[i, j, 0]
                        hole = grid[i, j, 1]
                        
                        if obstacle > 0.5 and hole > 0.5:
                            img_data[i, j] = [128, 0, 128]  # Purple
                        elif obstacle > 0.5:
                            img_data[i, j] = [255, 0, 0]  # Rouge
                        elif hole > 0.5:
                            img_data[i, j] = [0, 0, 255]  # Bleu
                
                # Robot en vert
                if 0 <= robot_row < rows and 0 <= robot_col < cols:
                    img_data[robot_row, robot_col] = [0, 255, 0]
                
                return Image.fromarray(img_data, 'RGB')
            
            def update(self):
                self.root.update()
        
        vision_window = VisionWindow()
    
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


def update_vision_display(axes, grid, env):
    """Met à jour l'affichage de la vision CNN."""
    robot_row = env.robot_row_in_grid
    robot_col = env.robot_col_in_grid
    
    # Canal 0 - Obstacles
    axes[0].clear()
    axes[0].imshow(grid[:, :, 0], cmap='Greys_r', interpolation='nearest', vmin=0, vmax=1)
    axes[0].plot(robot_col, robot_row, 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
    axes[0].arrow(robot_col, robot_row, 0, 3, head_width=1.5, head_length=1, fc='yellow', ec='orange', linewidth=2)
    axes[0].set_title('Canal 0 - Obstacles')
    axes[0].grid(True, alpha=0.3)
    
    # Canal 1 - Trous
    axes[1].clear()
    axes[1].imshow(grid[:, :, 1], cmap='Greys_r', interpolation='nearest', vmin=0, vmax=1)
    axes[1].plot(robot_col, robot_row, 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
    axes[1].arrow(robot_col, robot_row, 0, 3, head_width=1.5, head_length=1, fc='yellow', ec='orange', linewidth=2)
    axes[1].set_title('Canal 1 - Trous')
    axes[1].grid(True, alpha=0.3)
    
    # Vue combinée
    axes[2].clear()
    rows, cols = grid.shape[0], grid.shape[1]
    img = np.ones((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            obstacle = grid[i, j, 0]
            hole = grid[i, j, 1]
            
            if obstacle > 0.5 and hole > 0.5:
                img[i, j] = [0.5, 0, 0.5]  # Purple
            elif obstacle > 0.5:
                img[i, j] = [1, 0, 0]  # Rouge
            elif hole > 0.5:
                img[i, j] = [0, 0, 1]  # Bleu
            else:
                img[i, j] = [1, 1, 1]  # Blanc
    
    axes[2].imshow(img, interpolation='nearest')
    axes[2].plot(robot_col, robot_row, 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2)
    axes[2].arrow(robot_col, robot_row, 0, 3, head_width=1.5, head_length=1, fc='yellow', ec='orange', linewidth=2)
    axes[2].set_title('Vue Combinée')
    axes[2].grid(True, alpha=0.3)


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
            checkpoint = torch.load(model_path, map_location=device)
            
            # Nouveau format avec métadonnées
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
                
                # Restaurer l'état de l'optimiseur si disponible
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"   OK: Optimizer state restored!")
                else:
                    print(f"   WARNING: No optimizer state found (old checkpoint format)")
                
                saved_iteration = checkpoint.get('iteration', (latest_step // batch_size) + 1)
                saved_global_step = checkpoint.get('global_step', latest_step)
                saved_total_episodes = checkpoint.get('total_episodes', 0)
                
                # Restaurer l'état du curriculum si disponible
                if 'curriculum_state' in checkpoint and checkpoint['curriculum_state'] is not None:
                    curr_state = checkpoint['curriculum_state']
                    print(f"\nRESTORING CURRICULUM STATE:")
                    print(f"  Phase: {curr_state.get('current_phase', 1)}")
                    print(f"  Distance: {curr_state.get('current_distance', 0):.1f}m")
                    print(f"  Random: {curr_state.get('random_percentage', 0.2)*100:.0f}%")
                    print(f"  Max steps: {curr_state.get('max_steps', 3000)}")
                    print(f"  Bump ratio: {curr_state.get('bump_ratio', 0)*100:.0f}%")
                    
                    # Restaurer les attributs de get_curriculum_state
                    if 'phase_distance_history' in curr_state:
                        get_curriculum_state.phase_distance_history = curr_state['phase_distance_history']
                    if 'max_distance_achieved' in curr_state:
                        get_curriculum_state.max_distance_achieved = curr_state['max_distance_achieved']
                    if 'phase_transition_done' in curr_state:
                        get_curriculum_state.phase_transition_done = set(curr_state['phase_transition_done'])
                    
                    # Restaurer le cache de update_curriculum
                    update_curriculum.cached_curriculum_values = (
                        curr_state.get('random_percentage'),
                        curr_state.get('bump_ratio'),
                        curr_state.get('max_steps'),
                        curr_state.get('current_phase'),
                        curr_state.get('current_distance'),
                        False  # phase_changed = False au démarrage
                    )
                    print(f"{'='*70}\n")
                
                start_iteration = saved_iteration + 1  # Reprendre à l'itération suivante
                print(f"   OK: Model loaded with metadata! Resuming at iteration {start_iteration}")
                print(f"   SAVED: iteration={saved_iteration}, global_step={saved_global_step:,}, episodes={saved_total_episodes}")
                
                # NOUVEAU: Initialiser global_step avec la valeur sauvegardée
                global_step = saved_global_step
                
                # Ajuster total_episodes si disponible
                if saved_total_episodes > 0:
                    total_episodes = saved_total_episodes
                    
            # Ancien format (juste state_dict)
            else:
                agent.load_state_dict(checkpoint)
                start_iteration = (latest_step // batch_size) + 1
                global_step = latest_step  # NOUVEAU: Initialiser global_step
                print(f"   OK: Model loaded (legacy format)! Resuming at iteration {start_iteration}")
                
            print(f"   STATS: Steps completed: {latest_step:,}")
            print(f"   TARGET: Steps remaining: {total_timesteps - latest_step:,}")
            
            # NOUVEAU: Nettoyer le CSV temp au redémarrage pour éviter les données obsolètes
            temp_csv_path = "models/temp_training_metrics.csv"
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                print(f"   CLEANUP: Removed obsolete temp CSV")
            
            # NOUVEAU: Nettoyer les anciens graphiques PNG d'itérations futures
            import glob
            png_files = glob.glob("models/training_progress_iter_*.png")
            cleaned_count = 0
            for png_file in png_files:
                try:
                    # Extraire le numéro d'itération du nom de fichier
                    filename = os.path.basename(png_file)
                    iter_num = int(filename.replace("training_progress_iter_", "").replace(".png", ""))
                    
                    # Supprimer si l'itération est >= à l'itération de reprise
                    if iter_num >= start_iteration:
                        os.remove(png_file)
                        cleaned_count += 1
                except (ValueError, OSError):
                    continue
            
            if cleaned_count > 0:
                print(f"   CLEANUP: Removed {cleaned_count} obsolete PNG files (iter >= {start_iteration})")
                
        except Exception as e:
            print(f"   ERROR: Loading failed: {e}")
            print(f"   RESTART: Starting from scratch...")
            start_iteration = 1
            global_step = 0  # NOUVEAU: Initialiser global_step à 0 si échec
    else:
        print("NEW: New model - Starting from scratch")
    
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
    
    # Métriques par batch (configurable) - Charger existantes si disponibles
    batch_metrics, last_batch_episode = load_existing_metrics("models/training_metrics.csv")
    
    # Stats
    episode_returns = []
    episode_distances = []
    episode_steps = []  # Nouveau : durée des épisodes
    episode_reasons = []  # Nouveau : raisons de terminaison pour calcul succès
    best_return = -float('inf')
    best_distance = 0.0
    successes = 0
    total_episodes = last_batch_episode  # FIXED: Continue from last saved episode instead of 0
    
    # Compteur raisons de terminaison
    termination_reasons = {
        'success': 0,
        'fell': 0,
        'flipped': 0,
        'collision': 0,
        'out_of_bounds': 0,
        'stuck': 0,
        'no_progress': 0,
        'truncated': 0,
        'terminated': 0,
    }
    
    os.makedirs("models", exist_ok=True)
    
    # Initialiser les variables du curriculum (seront mises à jour après chaque batch)
    current_max_steps = config['environment'].get('max_steps', 4000) if config and 'environment' in config else 4000
    random_percentage = None
    bump_ratio = None

    for iteration in range(start_iteration, num_iterations + 1):
        # === UPDATE CURRICULUM (début d'itération) ===
        # Vérifier si un nouveau batch a été complété et mettre à jour le curriculum
        if batch_metrics:
            # Lire le dernier batch complété
            last_batch = batch_metrics[-1]
            last_batch_distance = last_batch['mean_distance']
            
            print(f"\n{'='*70}")
            print(f"CURRICULUM CHECK (Iteration {iteration})")
            print(f"{'='*70}")
            print(f"Last batch: #{last_batch['batch_num']} | Episodes {last_batch['episodes_range']} | Distance: {last_batch_distance:.1f}m")
            print()
            
            # Mettre à jour le curriculum basé sur le dernier batch
            combined_metrics = get_combined_metrics(batch_metrics, "models/temp_training_metrics.csv")
            current_max_steps, random_percentage, bump_ratio, phase_changed = update_curriculum(envs, debug_env, iteration, num_iterations, config, combined_metrics)
            
            # Render immédiatement si changement de phase
            if phase_changed:
                print(f"\n{'='*70}")
                print("PHASE CHANGE DETECTED - Rendering episode with new obstacles")
                print(f"{'='*70}\n")
                debug_render_episode(agent, debug_env, device, current_max_steps, bump_ratio)
                
                # IMPORTANT: Reset phase_changed dans le cache pour éviter de re-render
                if hasattr(update_curriculum, 'cached_curriculum_values'):
                    cached = list(update_curriculum.cached_curriculum_values)
                    cached[5] = False  # phase_changed = False
                    update_curriculum.cached_curriculum_values = tuple(cached)
        else:
            print(f"\n{'='*70}")
            print(f"CURRICULUM CHECK (Iteration {iteration})")
            print(f"{'='*70}")
            print("No batch metrics available yet")
            print(f"{'='*70}\n")
        
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
                    
                    episode_returns.append(ret)
                    episode_distances.append(dist)
                    episode_steps.append(steps)
                    episode_reasons.append(reason)  # Stocker la raison
                    total_episodes += 1
                    
                    # Log individuel pour chaque épisode avec type de corridor
                    corridor_type = info.get('corridor_type', 'unknown')
                    is_random = info.get('is_random', False)
                    corridor_seed = info.get('corridor_seed', -1)
                    random_str = "random" if is_random else "fixed"
                    log_line = f"Episode {total_episodes:>4}: {reason:<11} | Steps: {steps:>4} | Distance: {dist:>6.2f}m | Reward: {ret:>6.1f} | Corridor: {corridor_type}+{random_str} | Seed: {corridor_seed}"
                    print(log_line)
                    
                    # Écrire dans le fichier
                    episodes_log_file.write(log_line + "\n")
                    episodes_log_file.flush()  # Force l'écriture immédiate
                    
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
                # Les listes episode_returns, episode_distances, etc. commencent à 0
                # mais batch_start/batch_end sont des numéros d'épisodes absolus
                memory_start = len(episode_returns) - (total_episodes - batch_start)
                memory_end = len(episode_returns) - (total_episodes - batch_end)
                
                # Sécurité : s'assurer que les indices sont valides
                memory_start = max(0, memory_start)
                memory_end = min(len(episode_returns), memory_end)
                
                if memory_end <= memory_start:
                    print(f"  ERROR: Invalid memory indices {memory_start}-{memory_end} for batch {len(batch_metrics) + 1}")
                    break
                
                # Calculer moyennes pour ce batch de 20 épisodes
                batch_returns = episode_returns[memory_start:memory_end]
                batch_distances = episode_distances[memory_start:memory_end]
                batch_steps_list = episode_steps[memory_start:memory_end]
                batch_reasons = episode_reasons[memory_start:memory_end]
                
                # Valider les distances avant calcul de la moyenne
                valid_distances = [d for d in batch_distances if d == d and d is not None and d >= -10]  # d == d détecte non-NaN
                if len(valid_distances) == 0:
                    mean_distance = 0.0
                    print(f"  WARNING: No valid distances in batch {len(batch_metrics) + 1}, using 0.0")
                else:
                    mean_distance = np.mean(valid_distances)
                    if len(valid_distances) < len(batch_distances):
                        invalid_count = len(batch_distances) - len(valid_distances)
                        print(f"  WARNING: {invalid_count} invalid distances filtered out in batch {len(batch_metrics) + 1}")
                
                # Valider les returns avant calcul de la moyenne
                valid_returns = [r for r in batch_returns if r == r and r is not None]  # r == r détecte non-NaN
                if len(valid_returns) == 0:
                    mean_return = 0.0
                    print(f"  WARNING: No valid returns in batch {len(batch_metrics) + 1}, using 0.0")
                else:
                    mean_return = np.mean(valid_returns)
                    if len(valid_returns) < len(batch_returns):
                        invalid_count = len(batch_returns) - len(valid_returns)
                        print(f"  WARNING: {invalid_count} invalid returns filtered out in batch {len(batch_metrics) + 1}")
                
                # Valider les steps avant calcul de la moyenne
                valid_steps = [s for s in batch_steps_list if s == s and s is not None and s >= 0]  # s == s détecte non-NaN
                if len(valid_steps) == 0:
                    mean_survival = 0.0
                    print(f"  WARNING: No valid steps in batch {len(batch_metrics) + 1}, using 0.0")
                else:
                    mean_survival = np.mean(valid_steps)
                    if len(valid_steps) < len(batch_steps_list):
                        invalid_count = len(batch_steps_list) - len(valid_steps)
                        print(f"  WARNING: {invalid_count} invalid steps filtered out in batch {len(batch_metrics) + 1}")
                
                # Compter succès dans ce batch basé sur les RAISONS, pas les distances
                batch_successes = sum(1 for reason in batch_reasons if reason == 'success')
                
                # Calculer le prochain batch_num basé sur le max existant (pas sur len)
                next_batch_num = max([b['batch_num'] for b in batch_metrics], default=0) + 1
                
                # Créer le nouveau batch metric
                new_batch_metric = {
                    'batch_num': next_batch_num,
                    'episode_end': batch_end,  # Numéro du dernier épisode de ce batch
                    'episodes_range': f"{batch_start+1}-{batch_end}",
                    'global_step': global_step,
                    'mean_return': mean_return,
                    'mean_distance': mean_distance,
                    'mean_survival': mean_survival,
                    'success_rate': 100 * batch_successes / batch_size_metrics,
                }
                
                # Ajouter aux métriques en mémoire
                batch_metrics.append(new_batch_metric)
                
                # NOUVEAU: Sauvegarder immédiatement dans le CSV temporaire
                save_temp_batch_to_csv(new_batch_metric, "models/temp_training_metrics.csv")
                
                last_batch_episode = batch_end
                print(f"BATCH: Batch {len(batch_metrics)} completed (episodes {batch_start+1}-{batch_end}) - saved to temp CSV")
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_iterations} | Steps: {global_step:,} | SPS: {sps:,} | Time: {elapsed:.0f}s")
            print(f"Max Steps Curriculum: {current_max_steps}")
            if random_percentage is not None:
                print(f"Random Corridors: {random_percentage*100:.0f}%")
            if bump_ratio is not None:
                print(f"Obstacles: holes + {int(bump_ratio*100)}% bumps")
            print(f"{'='*70}")
            
            if episode_returns:
                recent_ret = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
                recent_dist = episode_distances[-100:] if len(episode_distances) >= 100 else episode_distances
                recent_steps = episode_steps[-100:] if len(episode_steps) >= 100 else episode_steps
                
                success_rate = 100 * successes / max(1, total_episodes)
                print(f"EPISODES: {total_episodes} total | Success: {successes} ({success_rate:.1f}%)")
                
                # NOUVEAU: Utiliser les métriques combinées pour l'affichage
                combined_metrics = get_combined_metrics(batch_metrics, "models/temp_training_metrics.csv")
                print(f"BATCHES: {len(combined_metrics)} batches of {batch_size_metrics} episodes completed (including temp)")
                
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
            # NOUVEAU: Fusionner les métriques temp avec les principales AVANT de sauvegarder le modèle
            batch_metrics = merge_and_sync_metrics(batch_metrics, "models/temp_training_metrics.csv", "models/training_metrics.csv")
            print(f"METRICS: All metrics synchronized to models/training_metrics.csv")
            
            model_path = f"models/ppo_corridor_{global_step}.pth"
            
            # Récupérer l'état du curriculum depuis le cache
            curriculum_state = None
            if hasattr(update_curriculum, 'cached_curriculum_values'):
                cached = update_curriculum.cached_curriculum_values
                curriculum_state = {
                    'random_percentage': cached[0],
                    'bump_ratio': cached[1],
                    'max_steps': cached[2],
                    'current_phase': cached[3],
                    'current_distance': cached[4],
                    'phase_distance_history': getattr(get_curriculum_state, 'phase_distance_history', []),
                    'max_distance_achieved': getattr(get_curriculum_state, 'max_distance_achieved', 0.0),
                    'phase_transition_done': list(getattr(get_curriculum_state, 'phase_transition_done', set()))
                }
            
            # NOUVEAU: Sauvegarder avec métadonnées (itération, global_step, etc.)
            save_data = {
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # Sauvegarder l'état de l'optimiseur
                'iteration': iteration,
                'global_step': global_step,
                'total_episodes': total_episodes,
                'batch_size': batch_size,
                'config_path': config_path if 'config_path' in locals() else 'config.yaml',
                'curriculum_state': curriculum_state
            }
            torch.save(save_data, model_path)
            print(f"SAVE: Model saved to {model_path} (iteration {iteration})")
        
        # Afficher graphiques avec toutes les métriques (principales + temp)
        if iteration % plot_interval == 0 and iteration > 0 and batch_metrics:
            combined_metrics = get_combined_metrics(batch_metrics, "models/temp_training_metrics.csv")
            plot_training_progress(combined_metrics, iteration)
        
        # Debug render
        if iteration % render_interval == 0 or iteration == 1:
            debug_render_episode(agent, debug_env, device, current_max_steps, bump_ratio)
    
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
    
    # Sauvegarder métriques finales (fusionner temp avec principales)
    batch_metrics = merge_and_sync_metrics(batch_metrics, "models/temp_training_metrics.csv", "models/training_metrics.csv")
    print(f"\nMETRICS: All metrics synchronized to models/training_metrics.csv")
    
    # Générer graphique final avec toutes les métriques
    if batch_metrics:
        plot_training_progress(batch_metrics, num_iterations)
    
    # Sauvegarde finale
    final_path = "models/ppo_corridor_final.pth"
    
    # Récupérer l'état du curriculum pour la sauvegarde finale
    curriculum_state = None
    if hasattr(update_curriculum, 'cached_curriculum_values'):
        cached = update_curriculum.cached_curriculum_values
        curriculum_state = {
            'random_percentage': cached[0],
            'bump_ratio': cached[1],
            'max_steps': cached[2],
            'current_phase': cached[3],
            'current_distance': cached[4],
            'phase_distance_history': getattr(get_curriculum_state, 'phase_distance_history', []),
            'max_distance_achieved': getattr(get_curriculum_state, 'max_distance_achieved', 0.0),
            'phase_transition_done': list(getattr(get_curriculum_state, 'phase_transition_done', set()))
        }
    
    final_save_data = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # Sauvegarder l'état de l'optimiseur
        'iteration': num_iterations,
        'global_step': total_timesteps,
        'total_episodes': total_episodes,
        'batch_size': batch_size,
        'config_path': config_path if 'config_path' in locals() else 'config.yaml',
        'curriculum_state': curriculum_state,
        'training_completed': True
    }
    torch.save(final_save_data, final_path)
    print(f"\nModèle sauvegardé: {final_path} (training completed)")
    print(f"{'='*70}\n")
    
    episodes_log_file.close()
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
