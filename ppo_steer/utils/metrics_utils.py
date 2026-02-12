"""
Utilitaires pour la gestion des métriques d'entraînement.
"""
import csv
import os


class IterationTracker:
    """Tracker pour les statistiques de l'itération courante."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise les stats pour une nouvelle itération."""
        self.episode_returns = []
        self.episode_distances = []
        self.episode_steps = []
        self.episode_reasons = []
        self.termination_reasons = {
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
    
    def add_episode(self, return_val, distance, steps, reason):
        """Ajoute un épisode aux stats de l'itération."""
        self.episode_returns.append(return_val)
        self.episode_distances.append(distance)
        self.episode_steps.append(steps)
        self.episode_reasons.append(reason)
        
        # Incrémenter le compteur de raison
        if reason in self.termination_reasons:
            self.termination_reasons[reason] += 1
    
    def get_stats(self):
        """Retourne les stats de l'itération courante.
        
        Returns:
            dict avec: returns, distances, steps, reasons, termination_counts
        """
        return {
            'returns': self.episode_returns,
            'distances': self.episode_distances,
            'steps': self.episode_steps,
            'reasons': self.episode_reasons,
            'termination_counts': self.termination_reasons.copy()
        }
    
    def has_episodes(self):
        """Vérifie si des épisodes ont été enregistrés."""
        return len(self.episode_returns) > 0


def clear_temp_metrics(temp_metrics_file="models/temp_training_metrics.csv"):
    """Supprime le fichier de métriques temporaires sans les fusionner."""
    try:
        if os.path.exists(temp_metrics_file):
            os.remove(temp_metrics_file)
            print(f"ROLLBACK: Temp metrics cleared without saving")
    except Exception as e:
        print(f"WARNING: Could not remove temp file: {e}")


def save_checkpoint_summary_to_log(iteration, global_step, temp_metrics, curriculum_state=None, 
                                   log_file="episodes_log.txt"):
    """Sauvegarde un résumé du checkpoint dans le fichier de log des épisodes."""
    if not temp_metrics:
        return
    
    # Calculer moyennes
    mean_return = sum(m['mean_return'] for m in temp_metrics) / len(temp_metrics)
    mean_distance = sum(m['mean_distance'] for m in temp_metrics) / len(temp_metrics)
    mean_survival = sum(m['mean_survival'] for m in temp_metrics) / len(temp_metrics)
    mean_success_rate = sum(m['success_rate'] for m in temp_metrics) / len(temp_metrics)
    
    # Créer le message de résumé
    summary_lines = [
        f"\n{'='*70}",
        f"CHECKPOINT SAVED - Iteration {iteration} | Global Step {global_step:,}",
        f"{'='*70}",
        f"Batches: {len(temp_metrics)} | Avg Return: {mean_return:.1f} | Avg Distance: {mean_distance:.2f}m",
        f"Avg Survival: {mean_survival:.0f} steps | Success Rate: {mean_success_rate:.1f}%"
    ]
    
    if curriculum_state:
        summary_lines.append(
            f"Curriculum: Random {curriculum_state.get('random_percentage', 0)*100:.0f}% | "
            f"Bumps {curriculum_state.get('bump_ratio', 0)*100:.0f}% | "
            f"Max Steps {curriculum_state.get('max_steps', 0)}"
        )
    
    summary_lines.append(f"{'='*70}\n")
    
    # Écrire dans le fichier
    try:
        with open(log_file, 'a') as f:
            f.write('\n'.join(summary_lines))
            f.flush()
    except Exception as e:
        print(f"WARNING: Could not write checkpoint summary to log: {e}")


def compute_checkpoint_metrics(temp_metrics):
    """Calcule les métriques moyennes depuis les temp_metrics pour le checkpoint.
    
    Args:
        temp_metrics: Liste des métriques de batches depuis le dernier checkpoint
        
    Returns:
        Dict avec les moyennes: mean_return, mean_distance, mean_survival, success_rate
    """
    if not temp_metrics:
        return None
    
    mean_return = sum(m['mean_return'] for m in temp_metrics) / len(temp_metrics)
    mean_distance = sum(m['mean_distance'] for m in temp_metrics) / len(temp_metrics)
    mean_survival = sum(m['mean_survival'] for m in temp_metrics) / len(temp_metrics)
    success_rate = sum(m['success_rate'] for m in temp_metrics) / len(temp_metrics)
    
    return {
        'mean_return': mean_return,
        'mean_distance': mean_distance,
        'mean_survival': mean_survival,
        'success_rate': success_rate
    }
