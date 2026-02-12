# PPO Robot Navigation - Corridor Environment

Entraînement d'un robot à 4 roues pour naviguer dans un corridor avec obstacles (trous et bosses) en utilisant l'algorithme PPO (Proximal Policy Optimization).

## 📋 Table des matières

- [Dépendances](#dépendances)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [Différences entre ppo_no_steer et ppo_steer](#différences-entre-ppo_no_steer-et-ppo_steer)

---

## 🔧 Dépendances

### Dépendances Python principales

```bash
# Deep Learning
torch>=2.0.0
numpy>=1.24.0

# Reinforcement Learning
gymnasium>=0.29.0

# Simulation physique
mujoco>=3.0.0

# Visualisation
matplotlib>=3.7.0
pillow>=10.0.0
tkinter  # Généralement inclus avec Python

# Configuration et données
pyyaml>=6.0
pandas>=2.0.0
```

### Installation automatique des dépendances d'affichage

Les scripts vérifient automatiquement la présence de `tkinter` et `PIL` au démarrage. Si ces dépendances sont manquantes, ils tentent de les installer automatiquement avec:

```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk
pip install pillow
```

---

## 📁 Structure du projet

```
workspace/
├── README.md                          # Ce fichier
│
├── ppo_no_steer/                      # Version avec 4 roues indépendantes
│   ├── README.md                      # Documentation détaillée
│   ├── config.yaml                    # Configuration complète
│   ├── train_ppo.py                   # Script d'entraînement
│   ├── test_ppo.py                    # Script de test
│   ├── corridor_env.py                # Environnement Gymnasium
│   ├── corridor_generator_similar.py  # Générateur de corridors
│   ├── visualize_corridor_map.py      # Visualisation CNN
│   ├── plot_metrics.py                # Graphiques de métriques
│   ├── four_wheel_robot.xml           # Modèle MuJoCo du robot
│   ├── corridor_*.xml                 # Corridors prédéfinis
│   ├── models/                        # Checkpoints et métriques
│   │   ├── ppo_corridor_*.pth         # Checkpoints sauvegardés
│   │   ├── training_metrics.csv       # Métriques d'entraînement
│   │   ├── training_curves_*.png      # Graphiques par itération
│   │   └── episodes_log.txt           # Log détaillé des épisodes
│   └── utils/                         # Modules utilitaires
│       ├── load_utils.py              # Chargement checkpoints/métriques
│       ├── save_utils.py              # Sauvegarde checkpoints/métriques
│       ├── metrics_utils.py           # Tracking des métriques
│       └── display_utils.py           # Affichage vision CNN
│
├── ppo_steer/                         # Version avec contrôle par volant
│   ├── README.md                      # Documentation détaillée
│   ├── config.yaml                    # Configuration complète
│   ├── train_ppo.py                   # Script d'entraînement
│   ├── test_ppo.py                    # Script de test
│   ├── corridor_env.py                # Environnement Gymnasium (steering)
│   ├── corridor_generator_similar.py  # Générateur de corridors
│   ├── visualize_corridor_map.py      # Visualisation CNN
│   ├── plot_metrics.py                # Graphiques de métriques
│   ├── four_wheel_robot.xml           # Modèle MuJoCo du robot
│   ├── corridor_*.xml                 # Corridors prédéfinis
│   ├── models/                        # Checkpoints et métriques
│   └── utils/                         # Modules utilitaires
│
├── ppo_final/                         # Version finale (référence)
├── corridor_creation/                 # Outils de création de corridors
└── notebooks/                         # Notebooks d'expérimentation
```

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone <repository-url>
cd workspace
```

### 2. Installer les dépendances Python

```bash
pip install torch numpy gymnasium mujoco matplotlib pillow pyyaml pandas
```

### 3. Vérifier l'installation

```bash
python -c "import torch; import mujoco; import gymnasium; print('✅ Installation OK')"
```

---

## ⚡ Utilisation rapide

### Entraînement (ppo_no_steer)

```bash
cd ppo_no_steer
python train_ppo.py
```

### Test d'un modèle entraîné

```bash
cd ppo_no_steer
python test_ppo.py --render --show-vision --num-episodes 5
```

### Visualisation des métriques

```bash
cd ppo_no_steer
python plot_metrics.py
```

---

## 🔄 Différences entre ppo_no_steer et ppo_steer

| Caractéristique | ppo_no_steer | ppo_steer |
|----------------|--------------|-----------|
| **Contrôle** | 4 roues indépendantes | Volant + vitesse (steering) |
| **Action space** | `Box(-1, 1, (4,))` | `Box(-1, 1, (2,))` |
| **Actions** | `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]` | `[steering_angle, speed]` |
| **Conversion** | Directe vers vitesses roues | `steer_angle_to_wheel_speeds()` |
| **Complexité** | Plus de liberté, plus difficile | Plus naturel, plus simple |
| **Usage** | Contrôle bas niveau | Contrôle haut niveau (comme une voiture) |

### Exemple d'actions

**ppo_no_steer:**
```python
action = [0.8, 0.8, 0.8, 0.8]  # Avancer tout droit
action = [0.5, 0.8, 0.5, 0.8]  # Tourner à gauche
```

**ppo_steer:**
```python
action = [0.0, 0.8]   # Avancer tout droit (angle=0°, vitesse=0.8)
action = [0.5, 0.8]   # Tourner à gauche (angle=15°, vitesse=0.8)
action = [-1.0, 0.5]  # Tourner à droite max (angle=-30°, vitesse=0.5)
```

---

## 📚 Documentation détaillée

Pour plus d'informations sur chaque version:

- **ppo_no_steer**: Voir [ppo_no_steer/README.md](ppo_no_steer/README.md)
- **ppo_steer**: Voir [ppo_steer/README.md](ppo_steer/README.md)

Chaque README contient:
- Description détaillée de tous les scripts
- Tous les arguments disponibles
- Pipeline complet d'entraînement avec diagrammes
- Exemples d'utilisation avancés
- Architecture du réseau de neurones
- Système de curriculum learning

---

## 🎯 Objectif

Le robot doit apprendre à naviguer dans un corridor de 100m avec:
- **Trous** (holes): Zones à éviter (chute = échec)
- **Bosses** (bumps): Obstacles qui ralentissent et pénalisent
- **Murs latéraux**: Limites du corridor (3m de large)

Le robot utilise:
- **Vision ego-centrique**: Grille CNN 2 canaux (obstacles, trous)
- **Historique de positions**: 8 frames passées pour anticipation
- **État du robot**: Position, vitesse, orientation

---

## 📊 Métriques d'entraînement

Les métriques suivantes sont trackées:
- **Return moyen**: Récompense cumulée par épisode
- **Distance moyenne**: Distance parcourue (objectif: 100m)
- **Taux de succès**: % d'épisodes atteignant 100m
- **Survie moyenne**: Nombre de steps avant terminaison
- **Raisons de terminaison**: fell, flipped, no_progress, success

---

## 🎓 Curriculum Learning

L'entraînement utilise un curriculum progressif:

1. **Phase 1**: Trous + 50% bosses (seuil: 10m)
2. **Phase 2**: Trous + 65% bosses (seuil: 12m)
3. **Phase 3**: Trous + 75% bosses (seuil: 65m)
4. **Phase 4**: Trous + 100% bosses (pas de seuil)

Le passage à la phase suivante se fait automatiquement quand la distance moyenne de l'itération dépasse le seuil.

---

## 🛠️ Configuration

Tous les paramètres sont configurables via `config.yaml`:
- Hyperparamètres PPO (learning rate, gamma, etc.)
- Architecture du réseau (CNN, MLP)
- Paramètres d'environnement (max_steps, vision)
- Curriculum learning (phases, seuils)
- Système de récompenses

---

## 📝 Licence

[Votre licence ici]

---

## 👥 Auteurs

[Vos noms ici]
