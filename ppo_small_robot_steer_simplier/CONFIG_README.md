# Configuration System - PPO Robot Training

Ce système utilise des fichiers JSON pour configurer tous les paramètres d'entraînement, permettant une gestion flexible et reproductible des expériences.

## Fichiers de configuration

### `config.json` - Configuration par défaut
Fichier de configuration principal utilisé par défaut par `train_ppo.py`.

### `config_example.json` - Configuration avec commentaires
Version documentée avec des commentaires expliquant chaque paramètre.

## Structure de la configuration

```json
{
  "training": {
    "total_timesteps": 8000000,    // Nombre total de steps d'entraînement
    "num_envs": 32,                // Nombre d'environnements parallèles
    "num_steps": 1024,             // Steps par rollout par environnement
    "num_minibatches": 32,         // Nombre de minibatches pour l'update
    "update_epochs": 10,           // Epochs par update PPO
    "seed": 1                      // Seed pour reproductibilité
  },
  
  "ppo": {
    "lr": 0.0005,                  // Learning rate
    "gamma": 0.995,                // Discount factor
    "gae_lambda": 0.98,            // GAE lambda
    "clip_coef": 0.2,              // PPO clip coefficient
    "ent_coef": 0.05,              // Entropy coefficient
    "vf_coef": 0.5,                // Value function coefficient
    "max_grad_norm": 0.5           // Gradient clipping
  },
  
  "environment": {
    "max_steps": 1000,             // Steps max par épisode
    "corridor_xml": null,          // Corridor fixe (null = aléatoire)
    "use_random_corridor": true    // Utiliser génération aléatoire
  },
  
  "network": {
    "robot_net_hidden": [64, 64],          // Couches cachées MLP robot
    "history_net_hidden": [128, 64, 32],   // Couches cachées MLP historique
    "cnn_channels": [32, 64, 128],         // Canaux CNN
    "backbone_hidden": [128, 64]           // Couches finales
  },
  
  "logging": {
    "log_interval": 2,             // Intervalle de logging (itérations)
    "save_interval": 10,           // Intervalle de sauvegarde
    "render_interval": 5,          // Intervalle de rendu debug
    "plot_interval": 5,            // Intervalle de génération graphiques
    "batch_size_metrics": 20       // Taille batch pour métriques
  }
}
```

## Utilisation

### Entraînement avec configuration par défaut
```bash
python train_ppo.py
```

### Entraînement avec configuration personnalisée
```bash
python train_ppo.py --config ma_config.json
```

### Override de paramètres spécifiques
```bash
python train_ppo.py --config ma_config.json --lr 0.001 --num-envs 64
```

### Validation d'une configuration
```bash
python validate_config.py config.json
```

### Comparaison de configurations
```bash
python validate_config.py config1.json config2.json
```

## Exemples de configurations

### Configuration rapide (debug)
```json
{
  "training": {
    "total_timesteps": 100000,
    "num_envs": 4,
    "num_steps": 256,
    "seed": 42
  },
  "ppo": {
    "lr": 0.001
  },
  "logging": {
    "log_interval": 1,
    "save_interval": 5,
    "render_interval": 2
  }
}
```

### Configuration longue (production)
```json
{
  "training": {
    "total_timesteps": 20000000,
    "num_envs": 64,
    "num_steps": 2048,
    "seed": 1
  },
  "ppo": {
    "lr": 0.0003,
    "gamma": 0.999
  },
  "logging": {
    "save_interval": 20,
    "render_interval": 10
  }
}
```

### Configuration avec corridor fixe
```json
{
  "environment": {
    "corridor_xml": "corridor_100.xml",
    "use_random_corridor": false,
    "max_steps": 3000
  }
}
```

## Validation automatique

Le script `validate_config.py` vérifie :

### Erreurs (bloquantes)
- Structure JSON valide
- Sections requises présentes
- Types de données corrects
- Paramètres obligatoires

### Avertissements (informatifs)
- Valeurs hors plages recommandées
- Paramètres potentiellement problématiques
- Incohérences entre paramètres

## Bonnes pratiques

### 1. Nommage des configurations
```
config_baseline.json          # Configuration de référence
config_high_lr.json          # Expérience learning rate élevé
config_small_network.json    # Réseau plus petit
config_long_episodes.json    # Épisodes plus longs
```

### 2. Versioning
- Garder les configurations d'expériences réussies
- Documenter les changements dans des commentaires
- Utiliser git pour tracker les modifications

### 3. Reproductibilité
- Toujours spécifier un seed fixe
- Sauvegarder la configuration avec les modèles
- Noter la version du code utilisée

### 4. Expérimentation
```bash
# Test rapide d'une nouvelle config
python validate_config.py nouvelle_config.json

# Comparaison avec baseline
python validate_config.py config_baseline.json nouvelle_config.json

# Entraînement avec override ponctuel
python train_ppo.py --config nouvelle_config.json --seed 42
```

## Paramètres importants

### Performance
- `num_envs`: Plus = plus rapide mais plus de RAM
- `num_steps`: Plus = plus stable mais updates moins fréquents
- `num_minibatches`: Équilibre vitesse/stabilité

### Apprentissage
- `lr`: 3e-4 à 1e-3 généralement bon
- `gamma`: 0.99-0.999 pour tâches longues
- `gae_lambda`: 0.95-0.98 pour bon credit assignment

### Stabilité
- `clip_coef`: 0.1-0.3, plus bas = plus conservateur
- `max_grad_norm`: 0.5-1.0 pour éviter explosions
- `ent_coef`: 0.01-0.1 pour exploration

## Dépannage

### Erreur "Section manquante"
Ajouter la section requise dans le JSON.

### Erreur "JSON invalide"
Vérifier la syntaxe avec `validate_config.py`.

### Performance dégradée
- Réduire `num_envs` si manque de RAM
- Ajuster `num_steps` selon la longueur des épisodes
- Vérifier que `lr` n'est pas trop élevé

### Pas de convergence
- Réduire `lr`
- Augmenter `gamma` pour tâches longues
- Ajuster `ent_coef` pour plus d'exploration