"""
Utilitaires pour la sauvegarde des checkpoints et métriques d'entraînement.
"""
import os
import csv
import torch


def save_checkpoint(agent, optimizer, iteration, global_step, last_episode=None, 
                   checkpoint_metrics=None, curriculum_state=None, 
                   metrics_file="models/training_metrics.csv", checkpoint_file=None):
    """Sauvegarde un checkpoint du modèle et de l'optimizer.
    
    Args:
        checkpoint_metrics: Dict avec les moyennes depuis le dernier checkpoint:
            - mean_return: Moyenne des returns
            - mean_distance: Moyenne des distances
            - mean_survival: Moyenne des durées de survie
            - success_rate: Taux de succès
        curriculum_state: Dict avec l'état du curriculum:
            - current_phase: Phase actuelle
            - random_percentage: Pourcentage de corridors aléatoires
            - bump_ratio: Ratio de bumps
    """
    os.makedirs("models", exist_ok=True)
    
    if checkpoint_file is None:
        checkpoint_file = f"models/ppo_corridor_{iteration}.pth"
    
    checkpoint = {
        'iteration': iteration,
        'global_step': global_step,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if last_episode is not None:
        checkpoint['last_episode'] = last_episode
    
    # Ajouter les métriques moyennes depuis le dernier checkpoint
    if checkpoint_metrics is not None:
        checkpoint['mean_return'] = checkpoint_metrics.get('mean_return', 0.0)
        checkpoint['mean_distance'] = checkpoint_metrics.get('mean_distance', 0.0)
        checkpoint['mean_survival'] = checkpoint_metrics.get('mean_survival', 0.0)
        checkpoint['success_rate'] = checkpoint_metrics.get('success_rate', 0.0)
    
    # Ajouter l'état du curriculum
    if curriculum_state is not None:
        checkpoint['current_phase'] = curriculum_state.get('current_phase', 1)
        checkpoint['random_percentage'] = curriculum_state.get('random_percentage', 0.0)
        checkpoint['bump_ratio'] = curriculum_state.get('bump_ratio', 0.0)
        checkpoint['phase_distance_history'] = curriculum_state.get('phase_distance_history', [])
    
    torch.save(checkpoint, checkpoint_file)
    print(f"CHECKPOINT: Saved to {checkpoint_file}")
    if checkpoint_metrics:
        print(f"   Metrics: Return={checkpoint_metrics.get('mean_return', 0):.1f}, "
              f"Distance={checkpoint_metrics.get('mean_distance', 0):.2f}m, "
              f"Success={checkpoint_metrics.get('success_rate', 0):.1f}%")
    if curriculum_state:
        print(f"   Curriculum: Phase={curriculum_state.get('current_phase', 1)}, "
              f"Random={curriculum_state.get('random_percentage', 0)*100:.0f}%, "
              f"Bumps={curriculum_state.get('bump_ratio', 0)*100:.0f}%")
    
    return checkpoint_file


def save_metrics_to_csv(batch_metrics, metrics_file="models/training_metrics.csv"):
    """Sauvegarde les métriques dans le CSV principal."""
    if not batch_metrics:
        return
    
    # Éliminer les doublons basés sur batch_num avant sauvegarde
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
                                               'mean_survival', 'success_rate',
                                               'current_phase', 'random_percentage', 'bump_ratio'])
        writer.writeheader()
        writer.writerows(unique_metrics)


def save_temp_batch_to_csv(batch_metric, temp_metrics_file="models/temp_training_metrics.csv"):
    """Sauvegarde un nouveau batch dans le CSV temporaire."""
    if not batch_metric:
        return
    
    try:
        # Si le fichier n'existe pas OU est vide, créer avec header
        file_exists = os.path.exists(temp_metrics_file)
        file_is_empty = not file_exists or os.path.getsize(temp_metrics_file) == 0
        
        with open(temp_metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                                   'global_step', 'mean_return', 'mean_distance', 
                                                   'mean_survival', 'success_rate', 
                                                   'current_phase', 'random_percentage', 'bump_ratio'])
            if file_is_empty:
                writer.writeheader()
            
            writer.writerow(batch_metric)
    except PermissionError:
        # Fichier ouvert dans l'éditeur - ignorer silencieusement
        if not hasattr(save_temp_batch_to_csv, '_permission_warning_shown'):
            print(f"WARNING: Cannot write to {temp_metrics_file} (file in use by editor)")
            save_temp_batch_to_csv._permission_warning_shown = True
    except Exception as e:
        print(f"WARNING: Could not write to temp metrics CSV: {e}")


def save_iteration_summary(iteration, global_step, last_episode, summary_file="models/iteration_summary.csv"):
    """Sauvegarde un résumé de l'itération avec la moyenne des métriques temp.
    
    Args:
        iteration: Numéro d'itération
        global_step: Nombre total de steps
        last_episode: Numéro du dernier épisode terminé
        summary_file: Fichier CSV de sortie
    """
    from .load_utils import load_temp_metrics
    
    # Charger les métriques temp
    temp_metrics = load_temp_metrics()
    
    if not temp_metrics:
        print("WARNING: No temp metrics to save in iteration summary")
        return
    
    # Calculer moyennes des métriques temp
    mean_return = sum(m['mean_return'] for m in temp_metrics) / len(temp_metrics)
    mean_distance = sum(m['mean_distance'] for m in temp_metrics) / len(temp_metrics)
    mean_survival = sum(m['mean_survival'] for m in temp_metrics) / len(temp_metrics)
    mean_success_rate = sum(m['success_rate'] for m in temp_metrics) / len(temp_metrics)
    
    # Prendre les dernières valeurs du curriculum (dernier batch)
    last_batch = temp_metrics[-1]
    current_phase = last_batch.get('current_phase', 1)
    random_percentage = last_batch.get('random_percentage', 0.0)
    bump_ratio = last_batch.get('bump_ratio', 0.0)
    
    # Créer l'entrée de résumé
    summary = {
        'iteration': iteration,
        'global_step': global_step,
        'last_episode': last_episode,
        'num_batches': len(temp_metrics),
        'mean_return': mean_return,
        'mean_distance': mean_distance,
        'mean_survival': mean_survival,
        'mean_success_rate': mean_success_rate,
        'current_phase': current_phase,
        'random_percentage': random_percentage,
        'bump_ratio': bump_ratio
    }
    
    # Écrire dans le CSV
    file_exists = os.path.exists(summary_file)
    file_is_empty = not file_exists or os.path.getsize(summary_file) == 0
    
    with open(summary_file, 'a', newline='') as f:
        fieldnames = ['iteration', 'global_step', 'last_episode', 'num_batches', 
                     'mean_return', 'mean_distance', 'mean_survival', 'mean_success_rate',
                     'current_phase', 'random_percentage', 'bump_ratio']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if file_is_empty:
            writer.writeheader()
        
        writer.writerow(summary)
        f.flush()  # Force l'écriture sur disque
        os.fsync(f.fileno())  # Sync avec le système de fichiers
    
    print(f"SUMMARY: Iteration {iteration} summary saved to {summary_file}")


def flush_temp_to_main_metrics(temp_metrics_file="models/temp_training_metrics.csv", 
                                main_metrics_file="models/training_metrics.csv"):
    """Charge les temp metrics, les ajoute au CSV principal, puis supprime le temp.
    
    Args:
        temp_metrics_file: Fichier CSV temporaire
        main_metrics_file: Fichier CSV principal
    """
    from .load_utils import load_temp_metrics
    
    # Charger les temp metrics
    temp_metrics = load_temp_metrics(temp_metrics_file)
    
    if not temp_metrics:
        print("FLUSH: No temp metrics to flush")
        return
    
    # Vérifier si le fichier principal existe et est vide
    file_exists = os.path.exists(main_metrics_file)
    file_is_empty = not file_exists or os.path.getsize(main_metrics_file) == 0
    
    # Ajouter les métriques à la suite du fichier principal
    with open(main_metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_num', 'episode_end', 'episodes_range', 
                                               'global_step', 'mean_return', 'mean_distance', 
                                               'mean_survival', 'success_rate',
                                               'current_phase', 'random_percentage', 'bump_ratio'])
        
        # Écrire le header seulement si le fichier est vide
        if file_is_empty:
            writer.writeheader()
        
        # Écrire toutes les métriques temp
        writer.writerows(temp_metrics)
    
    print(f"FLUSH: Added {len(temp_metrics)} batches to main metrics")
    
    # Supprimer le fichier temp
    try:
        if os.path.exists(temp_metrics_file):
            os.remove(temp_metrics_file)
            print(f"FLUSH: Temp metrics file deleted")
    except Exception as e:
        print(f"WARNING: Could not remove temp file: {e}")


def plot_training_curves(iteration=None, metrics_file="models/training_metrics.csv", output_file=None):
    """Génère et sauvegarde les graphiques de progression de l'entraînement.
    
    Args:
        iteration: Numéro d'itération pour le nom du fichier (optionnel)
        metrics_file: Fichier CSV contenant les métriques
        output_file: Fichier de sortie pour les graphiques (si None, utilise iteration dans le nom)
    """
    from .load_utils import load_metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Générer le nom du fichier avec l'itération si fournie
    if output_file is None:
        if iteration is not None:
            output_file = f"models/training_curves_{iteration}.png"
        else:
            output_file = "models/training_curves.png"
    import numpy as np
    
    # Charger toutes les métriques
    metrics = load_metrics(metrics_file)
    
    if not metrics:
        print("WARNING: No metrics to plot")
        return
    
    # Extraire les données
    episodes = [m['episode_end'] for m in metrics]
    returns = [m['mean_return'] for m in metrics]
    distances = [m['mean_distance'] for m in metrics]
    success_rates = [m['success_rate'] for m in metrics]
    survivals = [m['mean_survival'] for m in metrics]
    
    # Extraire les infos curriculum si disponibles
    has_curriculum = 'current_phase' in metrics[0]
    if has_curriculum:
        phases = [m.get('current_phase', 1) for m in metrics]
        random_pcts = [m.get('random_percentage', 0) for m in metrics]
        bump_ratios = [m.get('bump_ratio', 0) for m in metrics]
    
    # Créer figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Progression entraînement - {len(episodes)} batches', fontsize=14)
    axes = axes.flatten()
    
    # Définir les couleurs par phase si curriculum disponible
    if has_curriculum:
        # Créer un colormap basé sur les phases
        unique_phases = sorted(set(phases))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_phases)))
        phase_colors = {phase: colors[i] for i, phase in enumerate(unique_phases)}
        
        # Détecter les changements de phase pour les lignes verticales
        phase_changes = []
        for i in range(1, len(phases)):
            if phases[i] != phases[i-1]:
                phase_changes.append(episodes[i])
    
    # 1. Return moyen
    ax = axes[0]
    if has_curriculum:
        for i in range(len(episodes)):
            color = phase_colors[phases[i]]
            ax.plot(episodes[i:i+2] if i < len(episodes)-1 else [episodes[i]], 
                   returns[i:i+2] if i < len(returns)-1 else [returns[i]], 
                   color=color, linewidth=1.5, marker='o', markersize=3)
        for pc in phase_changes:
            ax.axvline(x=pc, color='red', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.plot(episodes, returns, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Return moyen')
    ax.set_title('Return moyen par batch')
    ax.grid(True, alpha=0.3)
    
    # 2. Distance moyenne
    ax = axes[1]
    if has_curriculum:
        for i in range(len(episodes)):
            color = phase_colors[phases[i]]
            ax.plot(episodes[i:i+2] if i < len(episodes)-1 else [episodes[i]], 
                   distances[i:i+2] if i < len(distances)-1 else [distances[i]], 
                   color=color, linewidth=1.5, marker='o', markersize=3)
        for pc in phase_changes:
            ax.axvline(x=pc, color='red', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.plot(episodes, distances, 'g-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance moyenne par batch')
    ax.grid(True, alpha=0.3)
    
    # 3. Taux de succès
    ax = axes[2]
    if has_curriculum:
        for i in range(len(episodes)):
            color = phase_colors[phases[i]]
            ax.plot(episodes[i:i+2] if i < len(episodes)-1 else [episodes[i]], 
                   success_rates[i:i+2] if i < len(success_rates)-1 else [success_rates[i]], 
                   color=color, linewidth=1.5, marker='o', markersize=3)
        for pc in phase_changes:
            ax.axvline(x=pc, color='red', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.plot(episodes, success_rates, 'r-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Taux de succès (%)')
    ax.set_title('Taux de succès par batch')
    ax.grid(True, alpha=0.3)
    
    # 4. Survie moyenne
    ax = axes[3]
    if has_curriculum:
        for i in range(len(episodes)):
            color = phase_colors[phases[i]]
            ax.plot(episodes[i:i+2] if i < len(episodes)-1 else [episodes[i]], 
                   survivals[i:i+2] if i < len(survivals)-1 else [survivals[i]], 
                   color=color, linewidth=1.5, marker='o', markersize=3)
        for pc in phase_changes:
            ax.axvline(x=pc, color='red', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.plot(episodes, survivals, 'm-', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Épisodes')
    ax.set_ylabel('Steps de survie')
    ax.set_title('Durée moyenne par batch')
    ax.grid(True, alpha=0.3)
    
    # Ajouter une légende pour les phases si curriculum
    if has_curriculum:
        legend_elements = [plt.Line2D([0], [0], color=phase_colors[phase], lw=2, label=f'Phase {phase}') 
                          for phase in unique_phases]
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=1, label='Changement de phase'))
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"PLOT: Training curves saved to {output_file}")



def save_episode_to_temp_log(episode_num, return_val, distance, survival, reason, 
                              temp_log_file="models/temp_episodes_log.txt"):
    """Sauvegarde un épisode dans le log temporaire.
    
    Args:
        episode_num: Numéro de l'épisode
        return_val: Return de l'épisode
        distance: Distance parcourue
        survival: Nombre de steps
        reason: Raison de fin (success, collision, timeout, etc.)
        temp_log_file: Fichier de log temporaire
    """
    log_line = f"Episode {episode_num:>6} | Return: {return_val:>7.1f} | Distance: {distance:>6.2f}m | Steps: {survival:>4} | {reason}\n"
    
    try:
        with open(temp_log_file, 'a') as f:
            f.write(log_line)
    except PermissionError:
        # Fichier ouvert dans l'éditeur - ignorer silencieusement
        # Afficher le warning seulement la première fois
        if not hasattr(save_episode_to_temp_log, '_permission_warning_shown'):
            print(f"WARNING: Cannot write to {temp_log_file} (file in use by editor)")
            save_episode_to_temp_log._permission_warning_shown = True
    except Exception as e:
        print(f"WARNING: Could not write to temp episode log: {e}")


def flush_temp_episode_logs(temp_log_file="models/temp_episodes_log.txt", 
                            main_log_file="models/episodes_log.txt"):
    """Fusionne les logs d'épisodes temp avec le log principal, puis supprime le temp.
    
    Args:
        temp_log_file: Fichier de log temporaire
        main_log_file: Fichier de log principal
    """
    from .load_utils import load_temp_episode_logs
    
    # Charger le contenu temp
    temp_content = load_temp_episode_logs(temp_log_file)
    
    if not temp_content:
        print("FLUSH LOGS: No temp episode logs to flush")
        return
    
    # Ajouter au fichier principal
    try:
        os.makedirs(os.path.dirname(main_log_file), exist_ok=True)
        with open(main_log_file, 'a') as f:
            f.write(temp_content)
        
        print(f"FLUSH LOGS: Episode logs flushed to {main_log_file}")
        
        # Supprimer le fichier temp
        if os.path.exists(temp_log_file):
            os.remove(temp_log_file)
            print(f"FLUSH LOGS: Temp episode log deleted")
            
    except Exception as e:
        print(f"WARNING: Could not flush temp episode logs: {e}")
