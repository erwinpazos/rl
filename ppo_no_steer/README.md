# PPO No Steer - Robot Navigation avec Contrôle Direct des Roues

Entraînement PPO pour un robot 4 roues naviguant dans un corridor avec obstacles (bumps) et trous, utilisant une vision CNN ego-centrique.

**DIFFÉRENCE AVEC ppo_final**: Contrôle DIRECT des 4 roues indépendantes au lieu d'un volant (steering + speed).

---

## 🔧 Prérequis

### Installation système (Ubuntu/Debian)

Pour la visualisation CNN avec tkinter:
```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk
```

### Packages Python

```bash
pip install numpy torch gymnasium mujoco pyyaml matplotlib pillow python3-pil.imagetk
```

**Note**: 
- tkinter est inclus avec Python mais nécessite `python3-tk` au niveau système
- ImageTk nécessite `python3-pil.imagetk` pour la visualisation CNN

---

## 📁 Structure du Projet

### 🔧 Fichiers de Configuration

#### `config.yaml`
Configuration centrale du projet. Contient tous les hyperparamètres:
- **PPO**: learning rate, gamma, GAE lambda, epochs, clip epsilon
- **Training**: nombre d'environnements parallèles, steps par rollout, batch sizes
- **Network**: architecture CNN et MLP (couches, tailles)
- **Vision**: dimensions de la grille, cell_size, distances de vision
- **Robot**: paramètres de contrôle (spawn angle) - PAS de steering/speed car contrôle direct des roues
- **Corridor**: longueur, largeur, distance de succès
- **Rewards**: récompenses/pénalités pour succès, échec, progression, collision, no-progress
- **Curriculum**: progression automatique basée sur la distance moyenne
  - `random_corridor_schedule`: % de corridors aléatoires vs fixes
  - `bump_ratio_schedule`: ratio de bumps ajoutés aux trous par phase

---

### 🤖 Environnement

#### `corridor_env.py`
Environnement Gymnasium pour le robot dans le corridor.

**CONTRÔLE**: 4 actions continues (vitesses angulaires des roues) au lieu de 2 (steering + speed)

**Caractéristiques:**
- **Observation**: 
  - État robot: position (x,y,z), vitesse (vx,vy,vz), angle (θ)
  - Historique: 8 frames de vitesses passées (48 valeurs)
  - Grille CNN: 35×20×2 canaux (obstacles, trous) - vision ego-centrique
- **Action**: 
  - `steering_angle`: angle de volant normalisé [-1, 1] → [-30°, +30°]
  - `speed`: vitesse normalisée [-1, 1] → [-1, +1] m/s
- **Récompenses**:
  - Progression en X: `progress_multiplier × distance`
  - Collision avec bump: `collision_penalty` par step
  - Succès (100m): `success_reward`
  - Échec (tombé/renversé): `failure_penalty`
  - No-progress: `no_progress_penalty` si < 0.5m en 500 steps
- **Terminaisons**:
  - `fell`: robot tombé (z < 0.15m)
  - `flipped`: robot renversé (angle > 60°)
  - `no_progress`: pas de progression suffisante
  - `success`: distance ≥ 100m (truncated, pas terminated)

**Méthodes importantes:**
- `_build_ego_centric_grid()`: Construit la grille de vision centrée et orientée selon le robot
- `_build_cell_map_from_xml()`: Parse le XML MuJoCo pour créer la carte des cellules
- `_build_model_from_new_generator()`: Génère un corridor aléatoire avec le générateur
- `_compute_reward()`: Calcule la récompense basée sur progression et collisions

**Détection des types de cellules:**
- Trous: géométries avec "hole" dans le nom OU absence de géométrie
- Bumps: géométries avec "bump" dans le nom
- Sol: géométries avec "flat", "floor", ou "cell" dans le nom

---

#### `corridor_generator_similar.py`
Générateur procédural de corridors avec trous et bumps.

**Fonctionnalités:**
- Génère des patterns de trous espacés de ~4m
- Ajoute des bumps entre les trous selon un ratio configurable
- Évite les répétitions de positions Y consécutives
- Crée de vrais trous en supprimant les tuiles de sol
- Génère des tuiles de 0.5m × 0.5m

**Méthodes principales:**
- `generate_hole_pattern(length, seed)`: Génère positions des trous
- `generate_bump_pattern()`: Génère positions des bumps entre trous
- `generate_corridor_xml(length, width, seed, obstacle_type, bump_ratio)`: Génère XML complet
- `save_corridor(filename, ...)`: Sauvegarde le corridor en fichier XML

**Paramètres:**
- Trous: 1 tuile (0.5m) en X, 2 tuiles (0.5m) en Y
- Bumps: 1 tuile (0.5m) cubique
- Positions Y: 6 niveaux pour bumps, 4 pour trous (symétriques)

---

### 🎓 Entraînement

#### `train_ppo.py`
Script principal d'entraînement PPO avec environnements parallèles.

**Architecture:**
- **Agent**: CNN + MLP avec 3 branches
  - Robot net: MLP pour état robot (7 → 32)
  - History net: MLP pour historique (48 → 32)
  - CNN: 2 couches conv pour grille (35×20×2 → 64)
  - Backbone: Fusion des 3 branches (128 → 64)
  - Actor/Critic: Têtes séparées
- **Optimisation**: Adam avec learning rate configurable
- **Parallélisation**: 30 environnements asynchrones (AsyncVectorEnv)
- **Batch processing**: Gros batches (30,720 steps) pour GPU

**Fonctionnalités:**
- Sauvegarde automatique tous les 10 batches
- Sauvegarde de l'état de l'optimizer pour reprise stable
- Curriculum learning automatique basé sur distance moyenne
- Logging des épisodes dans `episodes_log.txt`
- Métriques CSV par batch de 20 épisodes
- Graphiques de progression (return, distance, succès, survie)

**Curriculum:**
- Phase 1: 100% corridors aléatoires, 100% bumps
- Phase 2: Débloquée à 50m de distance moyenne
- Phase 3: Débloquée à 70m de distance moyenne
- Progression irréversible (pas de régression)

**Commandes:**
```bash
# Nouvel entraînement
python3 train_ppo.py

# Reprendre entraînement (détection auto du dernier modèle)
python3 train_ppo.py
```

---

#### `test_ppo.py`
Test d'un agent entraîné avec visualisation.

**Options:**
- `--model PATH`: Chemin du modèle (auto-détection si omis)
- `--episodes N`: Nombre d'épisodes à tester (défaut: 5)
- `--render`: Afficher la simulation MuJoCo
- `--show-vision`: Afficher la fenêtre tkinter avec vision CNN
- `--bump RATIO`: Ratio de bumps (0.0-1.0)
- `--corridor FILE`: Utiliser un corridor XML spécifique

**Fenêtre Vision CNN:**
- 3 vues: Canal 0 (obstacles), Canal 1 (trous), Vue combinée
- Zone de logs avec progression de l'épisode
- Mise à jour en temps réel toutes les 5 steps

**Commandes:**
```bash
# Test avec render et vision
python3 test_ppo.py --render --show-vision --episodes 3

# Test avec bump ratio spécifique
python3 test_ppo.py --render --bump 0.5

# Test avec corridor fixe
python3 test_ppo.py --render --corridor corridor_yguel.xml
```

---

### 🎮 Contrôle Manuel

#### `manual_control.py`
Contrôle manuel du robot avec les flèches du clavier.

**Contrôles:**
- `↑`: Accélérer (toggle ON/OFF)
- `↓`: Freiner/Reculer (toggle ON/OFF)
- `←`: Tourner à gauche (toggle ON/OFF)
- `→`: Tourner à droite (toggle ON/OFF)
- `ESPACE`: Arrêt d'urgence (reset tous les états)
- `R`: Reset environnement
- `ESC`: Quitter

**Options:**
- `--seed N`: Utiliser un seed spécifique pour le corridor
- `--fixed`: Utiliser un corridor fixe au lieu d'aléatoire
- `--bump RATIO`: Ratio de bumps (0.0-1.0)

**Fenêtre Vision CNN:**
- Même interface que test_ppo.py
- Affiche la vision du robot en temps réel
- Logs des commandes et statut

**Commandes:**
```bash
# Corridor aléatoire avec vision
python3 manual_control.py

# Corridor avec seed spécifique
python3 manual_control.py --seed 9371 --bump 1.0

# Corridor fixe
python3 manual_control.py --fixed --bump 0.3
```

---

### 📊 Visualisation

#### `plot_metrics.py`
Génère des graphiques de progression depuis les métriques CSV.

**Graphiques:**
- Return moyen par batch
- Distance moyenne par batch
- Taux de succès par batch
- Durée moyenne (steps) par batch

**Commande:**
```bash
python3 plot_metrics.py
```

Génère: `models/training_progress_latest.png`

---

#### `visualize_corridor_map.py`
Visualise la carte des cellules d'un corridor.

**Fonctionnalités:**
- Affiche la cell_map avec couleurs (sol, bumps, trous)
- Montre la position du robot
- Utile pour débugger la détection des obstacles

**Commande:**
```bash
python3 visualize_corridor_map.py --corridor corridor_yguel.xml
```

---

### 🧪 Utilitaires de Test

#### `generate_test_corridor.py`
Génère un corridor de test avec le générateur pour analyse.

**Utilité:**
- Vérifier que le générateur fonctionne correctement
- Analyser les positions des trous et bumps
- Tester avec des seeds spécifiques

**Commande:**
```bash
python3 generate_test_corridor.py
```

Génère: `corridor_test_generated.xml`

---

### 🗺️ Fichiers XML

#### `four_wheel_robot.xml`
Définition du robot 4 roues pour MuJoCo.

**Caractéristiques:**
- Empattement: 0.50m
- Voie: 0.40m
- Rayon des roues: 0.15m
- 4 roues avec joints de rotation
- Contrôle par steering angle + speed

---

#### `corridor_yguel.xml`
Corridor de test avec trous et bumps.

**Spécificités:**
- Trous représentés par `floor_hole_tile` avec `contype="0"` (pas de collision)
- Bumps représentés par `floor_bump`
- Tuiles de sol: `floor_flat`
- Longueur: ~100m

---

#### `corridor_erwin.xml`
Autre corridor de test (variante).

---

### 📂 Dossiers

#### `models/`
Contient les modèles sauvegardés et métriques.

**Fichiers:**
- `ppo_corridor_XXXXXX.pth`: Checkpoints du modèle (tous les 10 batches)
- `training_metrics.csv`: Métriques par batch (20 épisodes)
- `temp_training_metrics.csv`: Métriques temporaires (fusionnées à la sauvegarde)
- `training_progress_iter_XXX.png`: Graphiques de progression

**Format du checkpoint (.pth):**
```python
{
    'model_state_dict': ...,      # Poids du réseau
    'optimizer_state_dict': ...,  # État de l'optimizer (Adam)
    'iteration': N,               # Numéro d'itération
    'global_step': XXXXX,         # Steps totaux
    'total_episodes': XXX,        # Épisodes totaux
    'curriculum_state': {         # État du curriculum
        'phase': N,
        'distance': X.X,
        'random_percentage': X.X,
        'bump_ratio': X.X
    }
}
```

---

#### `final_working_model/`
Sauvegarde du meilleur modèle final.

---

#### `__pycache__/`
Cache Python (généré automatiquement).

---

### 📝 Fichiers de Logs

#### `episodes_log.txt`
Log de tous les épisodes pendant l'entraînement.

**Format:**
```
Episode XXX: reason | Steps: XXXX | Distance: XX.XXm | Reward: XX.X | Corridor: type | Seed: XXXX
```

**Raisons de terminaison:**
- `fell`: Tombé dans un trou
- `flipped`: Renversé
- `no_progress`: Pas de progression
- `success`: Objectif atteint (100m)

---

## 🚀 Workflow Complet

### 1. Nouvel Entraînement
```bash
# Configurer les paramètres dans config.yaml
nano config.yaml

# Lancer l'entraînement
python3 train_ppo.py

# Suivre la progression
tail -f episodes_log.txt
```

### 2. Reprendre un Entraînement
```bash
# Le script détecte automatiquement le dernier modèle
python3 train_ppo.py

# Ou spécifier un modèle
python3 train_ppo.py --model models/ppo_corridor_921600.pth
```

### 3. Tester un Modèle
```bash
# Test avec visualisation
python3 test_ppo.py --render --show-vision --episodes 5

# Test avec différents niveaux de difficulté
python3 test_ppo.py --render --bump 0.0  # Facile (seulement trous)
python3 test_ppo.py --render --bump 0.5  # Moyen
python3 test_ppo.py --render --bump 1.0  # Difficile (100% bumps)
```

### 4. Contrôle Manuel
```bash
# Tester manuellement avec vision CNN
python3 manual_control.py

# Tester un corridor spécifique vu pendant l'entraînement
python3 manual_control.py --seed 9371 --bump 1.0
```

### 5. Analyser les Résultats
```bash
# Générer les graphiques
python3 plot_metrics.py

# Visualiser un corridor
python3 visualize_corridor_map.py --corridor corridor_yguel.xml
```

---

## 🔍 Détails Techniques

### Vision CNN Ego-Centrique
- **Grille**: 35 lignes × 20 colonnes × 2 canaux
- **Cell size**: 0.2m × 0.2m
- **Vision**: 7m devant, 2m derrière, 2m de chaque côté
- **Robot**: Toujours à la ligne 10, colonne 10 (centre)
- **Rotation**: La grille tourne avec le robot (ego-centrique)

**Canal 0 - Obstacles:**
- 1.0 = Bump OU mur latéral
- 0.0 = Navigable

**Canal 1 - Trous:**
- 1.0 = Trou OU extérieur avant/arrière
- 0.0 = Navigable

**Sol navigable:** Les deux canaux à 0.0

---

### Curriculum Learning
Le curriculum progresse automatiquement basé sur la distance moyenne:

**Phase 1** (début):
- Random corridors: 100%
- Bump ratio: 100%
- Seuil: 0m

**Phase 2** (distance ≥ 50m):
- Random corridors: 100%
- Bump ratio: 50%
- Seuil: 50m

**Phase 3** (distance ≥ 70m):
- Random corridors: 100%
- Bump ratio: 30%
- Seuil: 70m

La progression est **irréversible** - une fois une phase atteinte, on ne régresse jamais.

---

### Optimisations
- **Environnements parallèles**: 30 envs asynchrones
- **Gros batches**: 30,720 steps par rollout
- **GPU**: Calculs sur CUDA si disponible
- **Minibatches**: 960 steps par minibatch (32 minibatches)
- **Epochs**: 10 epochs par batch
- **Clip epsilon**: 0.2 pour stabilité

---

### Sauvegarde et Reprise
- **Sauvegarde automatique**: Tous les 10 batches
- **Optimizer state**: Sauvegardé pour reprise stable
- **Métriques CSV**: Fusionnées avant sauvegarde pour synchronisation
- **Curriculum state**: Restauré à la reprise

**Important:** L'optimizer state est crucial! Sans lui, les premières itérations après reprise sont catastrophiques.

---

## 🐛 Debugging

### Problème: Trous non détectés
**Cause:** Les géométries de trous doivent avoir "hole" dans le nom OU être absentes.

**Solution:** Dans le XML, utiliser:
```xml
<geom name="floor_hole_tile_X" contype="0" conaffinity="0" .../>
```

### Problème: Performance catastrophique après reload
**Cause:** Optimizer state non sauvegardé/restauré.

**Solution:** Vérifier que le checkpoint contient `optimizer_state_dict`.

### Problème: Corridors identiques dans tous les envs
**Cause:** Utilisation de `np.random.randint()` au lieu de `self.env_random.randint()`.

**Solution:** Toujours utiliser le générateur aléatoire indépendant de chaque env.

---

## 🎯 Objectifs du Projet

1. ✅ Robot navigue dans un corridor avec vision CNN
2. ✅ Détection des trous et bumps
3. ✅ Curriculum learning automatique
4. ✅ Entraînement parallèle efficace
5. ✅ Sauvegarde/reprise stable
6. ✅ Visualisation en temps réel
7. ✅ Contrôle manuel pour tests

---

## 📈 Résultats Attendus

- **Phase 1** (100% bumps): ~40m de distance moyenne
- **Phase 2** (50% bumps): ~60m de distance moyenne
- **Phase 3** (30% bumps): ~80m+ de distance moyenne
- **Succès**: Atteindre 100m régulièrement

---

## 🔧 Configuration Recommandée

**Pour entraînement rapide:**
- 30 environnements parallèles
- Batch size: 30,720
- Learning rate: 0.0004
- GPU: CUDA recommandé

**Pour tests:**
- 1 environnement
- Render activé
- Vision CNN activée

---

## 📞 Support

Pour toute question ou problème, vérifier:
1. Les logs dans `episodes_log.txt`
2. Les métriques dans `models/training_metrics.csv`
3. Les graphiques générés par `plot_metrics.py`
4. La visualisation du corridor avec `visualize_corridor_map.py`
