"""
Package utils pour l'entraînement PPO.
"""
from .load_utils import (
    find_latest_checkpoint, 
    load_checkpoint, 
    load_last_iteration_summary, 
    load_temp_metrics,
    get_mean_distance_from_temp,
    load_metrics,
    load_temp_episode_logs
)
from .save_utils import (
    save_checkpoint, 
    save_metrics_to_csv, 
    save_temp_batch_to_csv, 
    save_iteration_summary,
    flush_temp_to_main_metrics,
    plot_training_curves,
    save_episode_to_temp_log,
    flush_temp_episode_logs
)
from .metrics_utils import (
    IterationTracker,
    clear_temp_metrics, 
    save_checkpoint_summary_to_log,
    compute_checkpoint_metrics
)
from .display_utils import (
    check_and_install_display_dependencies,
    display_vision,
    VisionWindow
)

__all__ = [
    # Checkpoint loading
    'find_latest_checkpoint',
    'load_checkpoint',
    # Checkpoint saving
    'save_checkpoint',
    # Metrics
    'save_metrics_to_csv',
    'save_temp_batch_to_csv',
    'load_temp_metrics',
    'merge_and_sync_metrics',
    'save_iteration_summary',
    'get_last_iteration_distance',
    'clear_temp_metrics',
    'save_checkpoint_summary_to_log',
    # Display
    'display_vision',
    'VisionWindow',
]
