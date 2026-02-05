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
    """Obtenir l'état actuel du curriculum basé sur la distance moyenne des 2 derniers batches."""
    if not config or 'curriculum' not in config:
        return None, None, None, 1
    
    curriculum_config = config['curriculum']
    if not curriculum_config.get('enabled', False):
        return None, None, None, 1
    
    # Calculer la distance moyenne des 2 derniers batches
    if len(batch_metrics) < 2:
        avg_distance = 0.0
    else:
        recent_distances = [batch['mean_distance'] for batch in batch_metrics[-2:]]
        
        # Filtrer les valeurs NaN ou invalides
        valid_distances = [d for d in recent_distances if d == d and d is not None and d >= -10]  # d == d détecte NaN
        if len(valid_distances) == 0:
            avg_distance = 0.0
            print(f"WARNING: All recent distances are invalid, using 0.0m for curriculum")
        else:
            avg_distance = sum(valid_distances) / len(valid_distances)
            
        # Debug seulement si problème
        if avg_distance == 0.0 and len(batch_metrics) >= 2:
            print(f"DEBUG: avg_distance=0! recent_distances={recent_distances}, valid_distances={valid_distances}")
            print(f"DEBUG: Last 2 batches: batch_nums=[{batch_metrics[-2]['batch_num']}, {batch_metrics[-1]['batch_num']}]")
    
    # Déterminer la phase actuelle et la distance locale
    phase_distance_history = getattr(get_curriculum_state, 'phase_distance_history', [])
    current_phase = len(phase_distance_history) + 1
    if current_phase > 3:
        current_phase = 3
    
    # Obtenir le seuil de distance pour passer à la phase suivante depuis le config
    bump_ratio_schedule = curriculum_config.get('bump_ratio_schedule', [])
    phase_threshold = 100  # Valeur par défaut
    for phase_config in bump_ratio_schedule:
        if phase_config['phase'] == current_phase:
            phase_threshold = phase_config.get('distance_threshold', 100)
            break
    
    # Calculer la distance "locale" pour la phase actuelle
    if current_phase == 1:
        # Phase 1: utiliser la distance globale
        local_distance = avg_distance
    else:
        # Phase 2 ou 3: calculer la distance depuis le début de la phase
        if len(phase_distance_history) >= current_phase - 1:
            # Distance depuis la transition vers cette phase
            phase_start_distance = phase_distance_history[current_phase - 2] if current_phase > 1 else 0
            local_distance = max(0, avg_distance - phase_start_distance)
        else:
            local_distance = avg_distance
    
    # Vérifier si on doit passer à la phase suivante
    phase_changed = False
    if current_phase < 3 and phase_threshold is not None and local_distance >= phase_threshold:
        # Vérifier si on n'a pas déjà fait la transition pour cette phase
        if not hasattr(get_curriculum_state, 'phase_transition_done'):
            get_curriculum_state.phase_transition_done = set()
        
        # Calculer la nouvelle phase
        new_phase = len(phase_distance_history) + 2  # +2 car on va ajouter à l'historique
        if new_phase > 3:
            new_phase = 3
        
        # Vérifier si cette transition n'a pas déjà été faite
        if new_phase not in get_curriculum_state.phase_transition_done:
            # Transition vers la phase suivante
            phase_distance_history.append(avg_distance)
            current_phase = len(phase_distance_history) + 1
            if current_phase > 3:
                current_phase = 3
            local_distance = 0.0  # Reset de la distance locale pour la nouvelle phase
            phase_changed = True
            get_curriculum_state.phase_transition_done.add(current_phase)  # Marquer cette phase comme traitée
            print(f"CURRICULUM: Phase transition! Moving to phase {current_phase} (local_distance: {local_distance:.1f}m ≥ {phase_threshold}m)")
    
    # Sauvegarder l'historique et tracker la distance max
    get_curriculum_state.phase_distance_history = phase_distance_history
    
    # NOUVEAU: Tracker la distance maximale globale atteinte (irréversible)
    if not hasattr(get_curriculum_state, 'max_distance_achieved'):
        get_curriculum_state.max_distance_achieved = 0.0
    
    # Mettre à jour la distance max si on progresse
    if avg_distance > get_curriculum_state.max_distance_achieved:
        get_curriculum_state.max_distance_achieved = avg_distance
        print(f"CURRICULUM: New max distance achieved: {avg_distance:.1f}m")
    
    max_distance_achieved = get_curriculum_state.max_distance_achieved
    
    # Obtenir le ratio de bumps pour la phase actuelle
    current_bump_ratio = 0.0  # Défaut
    for phase_config in bump_ratio_schedule:
        if phase_config['phase'] == current_phase:
            current_bump_ratio = phase_config['bump_ratio']
            break
    
    # Pour chaque phase, calculer la distance "locale" (reset à 0 à chaque phase)
    # Obtenir les paramètres basés sur la distance locale (reset à chaque phase)
    random_percentage = get_curriculum_value_by_distance(
        curriculum_config.get('random_corridor_schedule', []), 
        local_distance, 
        'random_percentage', 
        0.2,
        None  # Pas de palier irréversible, on reset à chaque phase
    )
    
    max_steps = get_curriculum_value_by_distance(
        curriculum_config.get('max_steps_schedule', []), 
        local_distance, 
        'max_steps', 
        3000,
        None  # Pas de palier irréversible, on reset à chaque phase
    )
    
    return random_percentage, current_bump_ratio, max_steps, current_phase, local_distance, phase_changed


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
    random_percentage, _, _, _, _, _ = get_curriculum_state(config, batch_metrics)
    return random_percentage


def get_curriculum_bump_ratio(config, iteration, batch_metrics=None):
    """Obtenir le ratio de bumps selon le curriculum."""
    if batch_metrics is None:
        return None
    _, bump_ratio, _, _, _, _ = get_curriculum_state(config, batch_metrics)
    return bump_ratio


def get_curriculum_max_steps(config, iteration, batch_metrics=None):
    """Obtenir le nombre de steps max selon le curriculum (compatibilité)."""
    if batch_metrics is None:
        return None
    _, _, max_steps, _, _, _ = get_curriculum_state(config, batch_metrics)
    return max_steps


def update_curriculum(envs, debug_env, iteration, num_iterations, config=None, batch_metrics=None):
    """Curriculum learning basé sur la distance moyenne des 2 derniers batches."""
    
    # Vérifier si les batch_metrics ont changé depuis la dernière fois
    current_batch_count = len(batch_metrics) if batch_metrics else 0
    last_batch_count = getattr(update_curriculum, 'last_batch_count', -1)
    
    # Ne mettre à jour le curriculum que si de nouveaux batches sont disponibles
    if current_batch_count <= last_batch_count:
        # Pas de nouveaux batches, utiliser les dernières valeurs calculées
        cached_values = getattr(update_curriculum, 'cached_curriculum_values', (None, None, None, None, None, False))
        random_percentage, bump_ratio, current_steps, current_phase, local_distance, phase_changed = cached_values
        
        # Valeurs par défaut si pas de cache
        if current_steps is None:
            if config and 'environment' in config:
                current_steps = config['environment'].get('max_steps', 4000)
            else:
                current_steps = 4000
        
        # Pas d'affichage si pas de changement
        return current_steps, random_percentage, bump_ratio, phase_changed
    
    # Nouveaux batches disponibles, recalculer le curriculum
    if iteration % 5 == 0:  # Afficher seulement toutes les 5 itérations pour réduire le spam
        print(f"CURRICULUM: New batch detected ({current_batch_count} vs {last_batch_count}), updating curriculum...")
    update_curriculum.last_batch_count = current_batch_count
    
    # Obtenir l'état du curriculum basé sur la distance
    random_percentage, bump_ratio, current_steps, current_phase, local_distance, phase_changed = get_curriculum_state(config, batch_metrics or [])
    
    # Sauvegarder les valeurs calculées
    update_curriculum.cached_curriculum_values = (random_percentage, bump_ratio, current_steps, current_phase, local_distance, phase_changed)
    
    # Valeurs par défaut si pas de curriculum
    if current_steps is None:
        if config and 'environment' in config:
            current_steps = config['environment'].get('max_steps', 4000)
        else:
            current_steps = 4000
    
    # Calculer la distance moyenne pour l'affichage
    avg_distance = 0.0
    if batch_metrics and len(batch_metrics) >= 2:
        recent_distances = [batch['mean_distance'] for batch in batch_metrics[-2:]]
        # Filtrer les valeurs NaN ou invalides
        valid_distances = [d for d in recent_distances if d == d and d is not None and d >= -10]  # d == d détecte non-NaN
        if len(valid_distances) > 0:
            avg_distance = sum(valid_distances) / len(valid_distances)
    
    if random_percentage is not None or current_steps != 4000 or bump_ratio is not None:
        # Afficher le curriculum actuel seulement si c'est un nouveau batch ou toutes les 10 itérations
        if current_batch_count > last_batch_count or iteration % 10 == 0:
            curriculum_info = f"CURRICULUM (Phase {current_phase}, local_dist: {local_distance:.1f}m, global_dist: {avg_distance:.1f}m):"
            if random_percentage is not None:
                curriculum_info += f" Random corridors: {random_percentage*100:.0f}%"
            curriculum_info += f" Max steps: {current_steps}"
            if bump_ratio is not None:
                curriculum_info += f" Obstacles: holes + {int(bump_ratio*100)}% bumps"
            print(curriculum_info)
    
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


def debug_render_episode(agent, debug_env, device, max_steps=None, current_bump_ratio=None):
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
                
                # Afficher info + vision
                if step % 25 == 0:
                    x = env_to_use.data.qpos[0]
                    stabilizing = " (STABILISATION)" if step < env_to_use.stabilization_steps else ""
                    print(f"Step {step}: x={x:.2f}m, reward={reward:.3f}, return={ep_return:.1f}{stabilizing}")
                    
                    # Décoder l'observation SIMPLIFIÉE AVEC HISTORIQUE RÉDUIT
                    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
                    # history_simplified = obs[7:7+env_to_use.history_dim].reshape(env_to_use.history_length, 6)  # frames × 6 valeurs
                    # grid = obs[7+env_to_use.history_dim:].reshape(env_to_use.grid_rows, env_to_use.grid_cols, 2)  # Grille dynamique×2
                    
                    print(f"  Robot: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f}), vel=({robot_state[3]:.2f}, {robot_state[4]:.2f}, {robot_state[5]:.2f}), angle={robot_state[6]:.2f}rad ({np.degrees(robot_state[6]):.1f}°)")
                    
                    # # Afficher historique simplifié (dernière frame)
                    # last_frame = history_simplified[-1]  # 6 valeurs: pos(3) + vel(3)
                    # last_pos = last_frame[:3]
                    # last_vel = last_frame[3:]
                    # print(f"  Historique (4 frames): dernière pos=({last_pos[0]:+.2f}, {last_pos[1]:+.2f}, {last_pos[2]:+.2f}), vel=({last_vel[0]:+.2f}, {last_vel[1]:+.2f}, {last_vel[2]:+.2f})")
                    
                    # # Afficher grille (20 lignes × TOUTE la largeur 30 colonnes) - Canal 0 (obstacles)
                    # obstacles_grid = grid[:, :, 0]  # Canal obstacles
                    # trous_grid = grid[:, :, 1]      # Canal trous
                    
                    # print(f"  GRILLE (lignes 0-{env_to_use.grid_rows-1}, EGO-CENTRIQUE - tourne avec robot):")
                    # for i in range(min(env_to_use.grid_rows, 20)):  # Limiter l'affichage à 20 lignes max
                    #     line = "    "
                    #     for j in range(min(env_to_use.grid_cols, 40)):  # Limiter l'affichage à 40 colonnes max
                    #         if obstacles_grid[i, j] > 0.5:
                    #             line += '#'  # Obstacle (bump)
                    #         elif trous_grid[i, j] > 0.5:
                    #             line += '.'  # Trou
                    #         else:
                    #             line += '/'  # Sol
                    #     relative_dist = (i - env_to_use.robot_row_in_grid) * env_to_use.cell_size  # Distance relative au robot
                    #     print(f"    {relative_dist:+.1f}m: {line}")
                    # print("    (/=floor, #=obstacle/bump, .=hole)")
                    # print("    (EGO-CENTRIC grid: rotates with robot, 'forward' = always up)")
                    # print(f"    (Vision: {env_to_use.vision_length}m x {env_to_use.vision_width}m, {env_to_use.cell_size}m cells)")
                
                v.sync()
                time.sleep(0.05)  # 20 FPS
            
            final_x = env_to_use.data.qpos[0]
            reason = info.get('reason', 'truncated')
            corridor_type = info.get('corridor_type', 'unknown')
            is_random = info.get('is_random', False)
            random_str = "random" if is_random else "fixed"
            print(f"Episode ended: {reason:<9} | Steps: {step:>4} | Distance: {final_x:>5.2f}m | Reward: {ep_return:>5.1f} | Corridor: {corridor_type}-{random_str}")
            
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
    
    # Curriculum initial (distance 0)
    initial_random_percentage, initial_bump_ratio, initial_max_steps, initial_phase, _, _ = get_curriculum_state(config, [])
    
    if initial_random_percentage is not None:
        print(f"CURRICULUM: Starting with {initial_random_percentage*100:.0f}% random corridors")
    if initial_max_steps is not None:
        print(f"CURRICULUM: Starting with {initial_max_steps} max steps per episode")
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
                saved_iteration = checkpoint.get('iteration', (latest_step // batch_size) + 1)
                saved_global_step = checkpoint.get('global_step', latest_step)
                saved_total_episodes = checkpoint.get('total_episodes', 0)
                
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
    values = torch.zeros((num_steps, num_envs), device=device)
    
    # Init
    if 'global_step' not in locals():
        global_step = 0  # Initialiser si pas déjà fait lors du chargement
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)
    
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
            
            print(f"{'='*70}\n")
        
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
                        
                        else:
                            # Fallback complet
                            reason = 'truncated' if trunc[i] else 'unknown'
                            dist = 0.0
                            ret = float(reward[i]) if i < len(reward) else 0.0
                            steps = 0
                            corridor_type = 'unknown'
                            is_random = False
                            
                    except (IndexError, KeyError, TypeError, AttributeError) as e:
                        # Fallback en cas d'erreur
                        reason = 'truncated' if trunc[i] else 'terminated'
                        dist = 0.0
                        ret = float(reward[i]) if i < len(reward) else 0.0
                        steps = 0
                        corridor_type = 'unknown'
                        is_random = False
                        print(f"  WARNING: Episode info extraction failed for env {i}: {e}")
                    
                    # Créer un dict info pour le log
                    info = {
                        'corridor_type': corridor_type,
                        'is_random': is_random
                    }
                    
                    episode_returns.append(ret)
                    episode_distances.append(dist)
                    episode_steps.append(steps)
                    episode_reasons.append(reason)  # Stocker la raison
                    total_episodes += 1
                    
                    # Log individuel pour chaque épisode avec type de corridor
                    corridor_type = info.get('corridor_type', 'unknown')
                    is_random = info.get('is_random', False)
                    random_str = "random" if is_random else "fixed"
                    print(f"Episode {total_episodes:>3}: {reason:<9} | Steps: {steps:>4} | Distance: {dist:>5.2f}m | Reward: {ret:>5.1f} | Corridor: {corridor_type}-{random_str}")
                    
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
                
                # Créer le nouveau batch metric
                new_batch_metric = {
                    'batch_num': len(batch_metrics) + 1,
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
            model_path = f"models/ppo_corridor_{global_step}.pth"
            
            # NOUVEAU: Sauvegarder avec métadonnées (itération, global_step, etc.)
            save_data = {
                'model_state_dict': agent.state_dict(),
                'iteration': iteration,
                'global_step': global_step,
                'total_episodes': total_episodes,
                'batch_size': batch_size,
                'config_path': config_path if 'config_path' in locals() else 'config.yaml'
            }
            torch.save(save_data, model_path)
            print(f"SAVE: Model saved to {model_path} (iteration {iteration})")
            
            # NOUVEAU: Fusionner les métriques temp avec les principales et sauvegarder
            batch_metrics = merge_and_sync_metrics(batch_metrics, "models/temp_training_metrics.csv", "models/training_metrics.csv")
            print(f"METRICS: All metrics synchronized to models/training_metrics.csv")
        
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
    final_save_data = {
        'model_state_dict': agent.state_dict(),
        'iteration': num_iterations,
        'global_step': total_timesteps,
        'total_episodes': total_episodes,
        'batch_size': batch_size,
        'config_path': config_path if 'config_path' in locals() else 'config.yaml',
        'training_completed': True
    }
    torch.save(final_save_data, final_path)
    print(f"\nModèle sauvegardé: {final_path} (training completed)")
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
