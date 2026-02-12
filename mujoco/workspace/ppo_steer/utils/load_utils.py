"""
Utilitaires pour le chargement des checkpoints et métriques d'entraînement.
"""
import os
import re
import torch


def find_latest_checkpoint(models_dir="models"):
    """Trouve le checkpoint le plus récent dans le dossier models (basé sur le numéro d'itération dans le nom)."""
    if not os.path.exists(models_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(models_dir):
        if file.startswith("ppo_corridor_") and file.endswith(".pth") and file != "ppo_corridor_best.pth":
            # Extraire le numéro d'itération du nom de fichier
            match = re.search(r'ppo_corridor_(\d+)\.pth', file)
            if match:
                iteration_num = int(match.group(1))
                checkpoint_files.append((iteration_num, file))
    
    if not checkpoint_files:
        return None
    
    # Trier par numéro d'itération et prendre le plus récent
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    _, latest_file = checkpoint_files[0]
    
    return os.path.join(models_dir, latest_file)


def load_checkpoint(checkpoint_path, agent, optimizer=None, device='cpu'):
    """Charge un checkpoint et retourne les informations d'état."""
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Charger les poids du modèle
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        # Charger l'optimizer si fourni
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Extraire les informations d'état
        iteration = checkpoint.get('iteration', 1)
        global_step = checkpoint.get('global_step', 0)
        last_episode = checkpoint.get('last_episode', 0)
        
        # Extraire les métriques si disponibles
        mean_return = checkpoint.get('mean_return', None)
        mean_distance = checkpoint.get('mean_distance', None)
        mean_survival = checkpoint.get('mean_survival', None)
        success_rate = checkpoint.get('success_rate', None)
        
        # Extraire l'état du curriculum si disponible
        current_phase = checkpoint.get('current_phase', None)
        random_percentage = checkpoint.get('random_percentage', None)
        bump_ratio = checkpoint.get('bump_ratio', None)
        
        print(f"RESUME: Loaded checkpoint from {checkpoint_path}")
        print(f"   Iteration: {iteration}")
        print(f"   Global step: {global_step:,}")
        print(f"   Last episode: {last_episode}")
        
        if mean_return is not None:
            print(f"   Last checkpoint metrics:")
            print(f"      Return: {mean_return:.1f}")
            print(f"      Distance: {mean_distance:.2f}m")
            print(f"      Survival: {mean_survival:.0f} steps")
            print(f"      Success rate: {success_rate:.1f}%")
        
        if current_phase is not None:
            print(f"   Curriculum state:")
            print(f"      Phase: {current_phase}")
            print(f"      Random: {random_percentage*100:.0f}%")
            print(f"      Bumps: {bump_ratio*100:.0f}%")
        
        return {
            'iteration': iteration,
            'global_step': global_step,
            'last_episode': last_episode,
            'mean_return': mean_return,
            'mean_distance': mean_distance,
            'mean_survival': mean_survival,
            'success_rate': success_rate,
            'current_phase': current_phase,
            'random_percentage': random_percentage,
            'bump_ratio': bump_ratio,
            'checkpoint': checkpoint
        }
        
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return None


def load_last_iteration_summary(summary_file="models/iteration_summary.csv"):
    """Charge la mean_distance de la dernière itération sauvegardée.
    
    Returns:
        float: mean_distance de la dernière itération, ou None si pas de fichier
    """
    import csv
    
    if not os.path.exists(summary_file):
        return None
    
    try:
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            
            last_row = rows[-1]
            mean_distance = float(last_row['mean_distance'])
            
            print(f"RESUME: Loaded last iteration summary from {summary_file}")
            print(f"   Mean distance: {mean_distance:.2f}m")
            
            return mean_distance
            
    except Exception as e:
        print(f"WARNING: Could not load last iteration summary: {e}")
        return None


def load_temp_metrics(temp_metrics_file="models/temp_training_metrics.csv"):
    """Charge les métriques temporaires depuis le CSV temp."""
    import csv
    
    if not os.path.exists(temp_metrics_file):
        return []
    
    try:
        temp_metrics = []
        with open(temp_metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Valider mean_distance
                try:
                    mean_distance = float(row['mean_distance'])
                    if mean_distance != mean_distance or mean_distance is None:
                        print(f"WARNING: Invalid mean_distance {row['mean_distance']} in temp CSV, using 0.0")
                        mean_distance = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_distance {row['mean_distance']} in temp CSV, using 0.0")
                    mean_distance = 0.0
                
                # Valider mean_return
                try:
                    mean_return = float(row['mean_return'])
                    if mean_return != mean_return or mean_return is None:
                        print(f"WARNING: Invalid mean_return {row['mean_return']} in temp CSV, using 0.0")
                        mean_return = 0.0
                except (ValueError, TypeError):
                    print(f"WARNING: Could not parse mean_return {row['mean_return']} in temp CSV, using 0.0")
                    mean_return = 0.0
                
                # Valider mean_survival
                try:
                    mean_survival = float(row['mean_survival'])
                    if mean_survival != mean_survival or mean_survival is None:
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
                
                # Ajouter les infos curriculum si disponibles
                if 'current_phase' in row:
                    metrics['current_phase'] = int(row['current_phase'])
                if 'random_percentage' in row:
                    metrics['random_percentage'] = float(row['random_percentage'])
                if 'bump_ratio' in row:
                    metrics['bump_ratio'] = float(row['bump_ratio'])
                
                temp_metrics.append(metrics)
        
        return temp_metrics
        
    except Exception as e:
        print(f"WARNING: Failed to load temp metrics: {e}")
        return []


def get_mean_distance_from_temp(temp_metrics_file="models/temp_training_metrics.csv"):
    """Calcule la moyenne des distances depuis les métriques temporaires.
    
    Returns:
        float: Moyenne des distances, ou None si pas de métriques
    """
    temp_metrics = load_temp_metrics(temp_metrics_file)
    
    if not temp_metrics:
        return None
    
    mean_distance = sum(m['mean_distance'] for m in temp_metrics) / len(temp_metrics)
    return mean_distance


def load_metrics(metrics_file="models/training_metrics.csv"):
    """Charge toutes les métriques depuis le CSV principal.
    
    Returns:
        list: Liste de tous les batches de métriques, ou [] si pas de fichier
    """
    import csv
    
    if not os.path.exists(metrics_file):
        return []
    
    try:
        all_metrics = []
        
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertir les valeurs en types appropriés avec validation
                try:
                    mean_distance = float(row['mean_distance'])
                    if mean_distance != mean_distance or mean_distance is None:
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
                
                # Ajouter les infos curriculum si disponibles
                if 'current_phase' in row:
                    metrics['current_phase'] = int(row['current_phase'])
                if 'random_percentage' in row:
                    metrics['random_percentage'] = float(row['random_percentage'])
                if 'bump_ratio' in row:
                    metrics['bump_ratio'] = float(row['bump_ratio'])
                
                all_metrics.append(metrics)
        
        # Log seulement si des métriques ont été chargées (utile pour plot)
        if all_metrics:
            print(f"LOAD: Loaded {len(all_metrics)} metric batches from {metrics_file}")
        return all_metrics
        
    except Exception as e:
        print(f"WARNING: Failed to load metrics: {e}")
        return []


def get_last_batch_num(metrics_file="models/training_metrics.csv"):
    """Récupère le numéro du dernier batch depuis le CSV principal.
    
    Returns:
        int: Numéro du dernier batch, ou 0 si pas de fichier
    """
    import csv
    
    if not os.path.exists(metrics_file):
        return 0
    
    try:
        with open(metrics_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 0
            
            last_batch_num = int(rows[-1]['batch_num'])
            return last_batch_num
            
    except Exception as e:
        print(f"WARNING: Could not read last batch num: {e}")
        return 0


def load_temp_episode_logs(temp_log_file="models/temp_episodes_log.txt"):
    """Charge les logs d'épisodes temporaires.
    
    Returns:
        str: Contenu du fichier temp, ou chaîne vide si pas de fichier
    """
    if not os.path.exists(temp_log_file):
        return ""
    
    try:
        with open(temp_log_file, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"WARNING: Could not load temp episode logs: {e}")
        return ""
