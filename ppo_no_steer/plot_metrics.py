"""
Script pour visualiser les métriques d'entraînement.
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_training_metrics(csv_file="models/training_metrics.csv"):
    """Affiche les courbes d'entraînement."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Fichier {csv_file} non trouvé!")
        return
    
    if df.empty:
        print("Pas de données dans le fichier!")
        return
    
    # Créer figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métriques d\'entraînement PPO', fontsize=16)
    
    # 1. Return moyen
    ax = axes[0, 0]
    ax.plot(df['global_step'], df['mean_return'], 'b-', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Return moyen')
    ax.set_title('Return moyen (derniers 100 épisodes)')
    ax.grid(True, alpha=0.3)
    
    # 2. Distance moyenne
    ax = axes[0, 1]
    ax.plot(df['global_step'], df['mean_distance'], 'g-', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance moyenne (derniers 100 épisodes)')
    ax.grid(True, alpha=0.3)
    
    # 3. Taux de succès
    ax = axes[1, 0]
    ax.plot(df['global_step'], df['success_rate'], 'r-', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Taux de succès (%)')
    ax.set_title('Taux de succès')
    ax.grid(True, alpha=0.3)
    
    # 4. Survie moyenne
    ax = axes[1, 1]
    ax.plot(df['global_step'], df['mean_survival'], 'm-', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Steps de survie')
    ax.set_title('Durée moyenne des épisodes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "models/training_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Courbes sauvegardées: {output_file}")
    
    # Afficher
    plt.show()
    
    # Afficher stats finales
    print("\n" + "="*50)
    print("STATISTIQUES FINALES")
    print("="*50)
    print(f"Total steps: {df['global_step'].iloc[-1]:,}")
    print(f"Total épisodes: {df['total_episodes'].iloc[-1]}")
    print(f"Return final: {df['mean_return'].iloc[-1]:.1f}")
    print(f"Distance finale: {df['mean_distance'].iloc[-1]:.1f}m")
    print(f"Taux de succès final: {df['success_rate'].iloc[-1]:.1f}%")
    print(f"Survie finale: {df['mean_survival'].iloc[-1]:.0f} steps")
    print("="*50)

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "models/training_metrics.csv"
    plot_training_metrics(csv_file)
