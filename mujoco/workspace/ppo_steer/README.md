# PPO Steer - Contrôle par volant et vitesse

Version avec contrôle haut niveau par volant et vitesse (action space: 2 dimensions).

**Emplacement**: `mujoco/workspace/ppo_steer/`

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Scripts disponibles](#scripts-disponibles)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Architecture du réseau](#architecture-du-réseau)
- [Configuration](#configuration)

---

## Vue d'ensemble

Cette version contrôle le robot comme une voiture:
- **Action space**: `Box(-1.0, 1.0, (2,))` 
- **Actions**: `[steering_angle, speed]`
  - `steering_angle`: Angle de volant normalisé [-1, 1] → [-30°, +30°]
  - `speed`: Vitesse normalisée [-1, 1] → [-max_speed, +max_speed]
- Conversion automatique vers vitesses des 4 roues via `steer_angle_to_wheel_speeds()`
- Plus naturel et plus simple à apprendre

**Prérequis**: Environnement Docker lancé (voir [README principal](../../../README.md))

---

## Conversion steering → wheel speeds

La fonction `steer_angle_to_wheel_speeds()` convertit les commandes haut niveau en vitesses de roues:

```python
def steer_angle_to_wheel_speeds(steering_angle, speed, wheelbase, track_width):
    """
    steering_angle: angle en radians [-π/6, +π/6] (±30°)
    speed: vitesse linéaire en m/s
    wheelbase: distance entre essieux (m)
    track_width: largeur entre roues (m)
    
    Returns: [v_FL, v_FR, v_RL, v_RR] (vitesses angulaires rad/s)
    """
```

**Principe:**
1. Calculer rayon de braquage: `R = wheelbase / tan(steering_angle)`
2. Calculer vitesses linéaires des roues gauche/droite
3. Convertir en vitesses angulaires: `ω = v / wheel_radius`

---

## Scripts disponibles

### 1. train_ppo.py - Entraînement

Script principal d'entraînement avec PPO et contrôle par steering.

**Usage:**
```bash
# Dans l'environnement Docker (http://localhost:6080)
cd ~/workspace/ppo_steer
python train_ppo.py [OPTIONS]
```

**Arguments:**
- `--config PATH`: Fichier de configuration YAML (défaut: `config.yaml`)
- `--timesteps N`: Override total timesteps (défaut: 8,000,000)
- `--num-envs N`: Override nombre d'environnements parallèles (défaut: 30)
- `--num-steps N`: Override steps par rollout (défaut: 1024)
- `--lr FLOAT`: Override learning rate (défaut: 0.0004)
- `--seed N`: Override seed pour reproductibilité (défaut: 1)
- `--fresh-start`: Forcer nouveau démarrage (ignorer checkpoints existants)
- `--rollback`: Activer rollback automatique si performance régresse

**Exemples:**
```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Entraînement standard avec rollback
python train_ppo.py --rollback

# Nouveau démarrage avec 16 environnements
python train_ppo.py --fresh-start --num-envs 16

# Override learning rate et seed
python train_ppo.py --lr 0.0003 --seed 42
```

**Fonctionnalités:**
- Entraînement parallèle avec AsyncVectorEnv
- Sauvegarde automatique tous les N itérations
- Curriculum learning progressif (4 phases)
- Rollback automatique en cas de régression (avec `--rollback`)
- Génération de graphiques de métriques
- Logs détaillés par itération et par épisode
- Resume automatique depuis dernier checkpoint
- Action space 2D: `[steering_angle, speed]`

**Fichiers générés:**
- `models/ppo_corridor_{iteration}.pth`: Checkpoints sauvegardés
- `models/training_metrics.csv`: Métriques d'entraînement
- `models/training_curves_{iteration}.png`: Graphiques par itération
- `models/training_curves.png`: Graphique final
- `models/episodes_log.txt`: Log détaillé de tous les épisodes
- `models/iteration_summary.json`: Résumé de la dernière itération sauvegardée

---

### 2. test_ppo.py - Test d'un modèle

Teste un modèle entraîné sur N épisodes avec contrôle steering.

**Usage:**
```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer
python test_ppo.py [OPTIONS]
```

**Arguments:**
- `--model PATH`: Chemin vers le checkpoint (défaut: dernier checkpoint trouvé)
- `--config PATH`: Fichier de configuration YAML (défaut: `config.yaml`)
- `--num-episodes N`: Nombre d'épisodes à tester (défaut: 10)
- `--render`: Afficher le rendu 3D MuJoCo
- `--show-vision`: Afficher la vision CNN en temps réel
- `--corridor PATH`: Utiliser un corridor spécifique (XML)
- `--bump-ratio FLOAT`: Ratio de bosses (0.0 à 1.0, défaut: depuis config)

**Exemples:**
```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Test simple (10 épisodes, pas de rendu)
python test_ppo.py

# Test avec rendu 3D et vision CNN
python test_ppo.py --render --show-vision --num-episodes 5

# Test sur corridor spécifique
python test_ppo.py --render --corridor corridor_yguel.xml

# Test avec 100% de bosses
python test_ppo.py --render --bump-ratio 1.0 --num-episodes 3

# Test d'un checkpoint spécifique
python test_ppo.py --model models/ppo_corridor_50.pth --render
```

**Affichage:**
- Statistiques par épisode (reward, distance, raison de terminaison)
- Résumé final (moyenne ± std, meilleure distance)
- Vision CNN en temps réel (si `--show-vision`)
- Rendu 3D MuJoCo (si `--render`)
- Actions steering affichées: `[angle, speed]`

---

### 3. visualize_corridor_map.py - Visualisation CNN

Visualise exactement ce que le CNN reçoit en entrée (grille 2 canaux).

**Usage:**
```bash
python visualize_corridor_map.py [OPTIONS]
```

**Arguments:**
- `--corridor PATH`: Fichier XML du corridor (défaut: génération aléatoire)
- `--x FLOAT`: Position X du robot (défaut: aléatoire)
- `--y FLOAT`: Position Y du robot (défaut: aléatoire)
- `--angle FLOAT`: Angle du robot en degrés (défaut: aléatoire)
- `--seed N`: Seed pour génération aléatoire
- `--bump-ratio FLOAT`: Ratio de bosses (0.0 à 1.0, défaut: 0.5)
- `--render`: Ouvrir rendu 3D MuJoCo après visualisation

**Exemples:**
```bash
# Visualisation avec corridor aléatoire
python visualize_corridor_map.py

# Visualisation d'un corridor spécifique
python visualize_corridor_map.py --corridor corridor_yguel.xml

# Position spécifique du robot
python visualize_corridor_map.py --x 50.0 --y 0.5 --angle 15

# Avec rendu 3D
python visualize_corridor_map.py --render

# Corridor aléatoire avec seed fixe
python visualize_corridor_map.py --seed 42 --bump-ratio 0.8
```

**Affichage:**
- État du robot (position, vitesse, angle)
- Historique simplifié (8 frames × 6 valeurs)
- Canal 0: Obstacles (bumps + murs latéraux)
- Canal 1: Trous (holes + extérieur)
- Vue combinée (rouge=obstacle, bleu=trou, blanc=sol)
- Statistiques des canaux

---

### 4. corridor_env.py - Environnement Gymnasium

Environnement personnalisé pour le robot dans le corridor avec contrôle steering.

**Caractéristiques:**
- Compatible Gymnasium (gym.Env)
- Observation: état robot + historique + grille CNN (2 canaux)
- Action: 2 valeurs `[steering_angle, speed]` dans `[-1, 1]`
- Récompenses: progression, collision, succès/échec
- Terminaison: fell, flipped, no_progress, success

**Action space et conversion:**

```python
# Action space
self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

# Dans step()
def step(self, action):
    steering_angle_normalized = action[0]  # [-1, 1]
    speed_normalized = action[1]           # [-1, 1]
    
    # Dénormaliser
    steering_angle = steering_angle_normalized * self.max_steering_angle_rad
    speed = speed_normalized * self.max_speed
    
    # Convertir en vitesses de roues
    wheel_speeds = steer_angle_to_wheel_speeds(
        steering_angle, speed, 
        self.wheelbase_length, self.track_width
    )
    
    # Appliquer aux actuateurs MuJoCo
    self.data.ctrl[:] = wheel_speeds
```

### 5. corridor_generator_similar.py - Générateur de corridors

Génère des corridors aléatoires avec trous et bosses.

**Fonctionnalités:**
- Génération procédurale avec seed
- Contrôle du ratio de bosses (bump_ratio)
- Toujours 100% de trous + X% de bosses
- Sauvegarde en XML MuJoCo

---

## Pipeline d'entraînement complet

### Diagramme du pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         INITIALISATION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ load_config()    │ ← config.yaml
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Créer Agent      │ ← Actor: 64→2 (steering+speed)
                    │ + Optimizer      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ find_latest_     │
                    │ checkpoint()     │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
            ┌──────────────┐   ┌──────────────┐
            │ Checkpoint   │   │ Pas de       │
            │ trouvé       │   │ checkpoint   │
            └──────┬───────┘   └──────┬───────┘
                   │                  │
                   └──────────┬───────┘
                              │
                              ▼
         ┌─────────────────────────────────┐
         │ Créer environnements parallèles │
         │ AsyncVectorEnv (30 envs)        │
         │ Action space: Box(-1,1,(2,))    │
         └─────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BOUCLE D'ENTRAÎNEMENT                       │
│                    (260 itérations × 30,720 steps)               │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ COLLECTE ROLLOUT            │
         │ (30 envs × 1024 steps)      │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Pour chaque step:           │
         │ 1. agent.get_action(obs)    │ ← Forward: obs → [steering, speed]
         │ 2. envs.step(actions)       │ ← Conversion: [s,v] → [4 wheels]
         │ 3. Stocker transitions      │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ CALCUL ADVANTAGES           │
         │ compute_gae()               │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ UPDATE PPO                  │
         │ (10 epochs × 32 minibatches)│
         │ Actor: 2 actions            │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ MÉTRIQUES + SAUVEGARDE      │
         │ (identique à no_steer)      │
         └─────────────────────────────┘
```

### Détails du pipeline

**Initialisation:**
1. Charger configuration depuis `config.yaml`
2. Créer agent avec architecture CNN+MLP (actor head: 64→2)
3. Chercher dernier checkpoint pour resume automatique
4. Créer 30 environnements parallèles avec action space 2D

**Boucle d'entraînement (260 itérations):**
1. Reset stats d'itération
2. Déterminer phase curriculum (bump_ratio, max_steps)
3. Collecter rollout (30 envs × 1024 steps):
   - Forward pass: obs → `[steering, speed]`
   - Conversion automatique: `[steering, speed]` → 4 vitesses de roues
   - Simulation MuJoCo
   - Stocker transitions
4. Calculer advantages avec GAE
5. Update PPO (10 epochs × 32 minibatches)
6. Calculer métriques batch
7. Sauvegarder dans temp CSV
8. Tous les N itérations:
   - Vérifier performance (rollback si régression)
   - Sauvegarder checkpoint + métriques
   - Générer graphiques
   - Flush temp → main files

**Caractéristiques clés:**
- Action space 2D simplifie l'apprentissage
- Conversion steering→wheels dans l'environnement
- Curriculum progressif (4 phases)
- Rollback automatique optionnel

---

## Architecture du réseau

### Vue d'ensemble

```
OBSERVATION (7 + history_dim + grid_dim valeurs)
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Robot   │    │ History  │    │ Grid     │    │ Grid         │
│ State   │    │ (48 val) │    │ Canal 0  │    │ Canal 1      │
│ (7 val) │    │          │    │(obstacles│    │ (trous)      │
└────┬────┘    └─────┬────┘    └─────┬────┘    └──────┬───────┘
     │               │               │                 │
     ▼               ▼               └────────┬────────┘
┌─────────┐    ┌──────────┐                  │
│ MLP     │    │ MLP      │                  ▼
│ 7→32    │    │ 48→64→32 │         ┌─────────────────┐
└────┬────┘    └─────┬────┘         │ CNN 2 canaux    │
     │               │              │ Conv 2→32       │
     │               │              │ Conv 32→64      │
     │               │              │ Flatten         │
     │               │              │ Linear→64       │
     │               │              └────────┬────────┘
     │               │                       │
     └───────┬───────┴───────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Concatenate     │
    │ (32+32+64=128)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Backbone MLP    │
    │ 128→64          │
    └────────┬────────┘
             │
             ├─────────────────┬─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Actor Mean  │   │ Actor LogStd│   │ Critic      │
    │ 64→2        │   │ (learnable) │   │ 64→1        │
    │ [steer,spd] │   │             │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                  │
           └────────┬────────┘                  │
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐         ┌─────────────┐
           │ Normal(μ, σ)    │         │ Value       │
           │ Sample action   │         │ Estimate    │
           │ [steering,speed]│         │             │
           └─────────────────┘         └─────────────┘
```

**Différence avec no_steer:**
- Actor Mean: `Linear(64, 2)` au lieu de `Linear(64, 4)`
- Actor LogStd: `Parameter(zeros(1, 2))` au lieu de `(1, 4)`
- Output: `[steering_angle, speed]` au lieu de `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]`

Le reste de l'architecture est identique.

---

## Configuration (config.yaml)

### Structure complète

```yaml
training:
  total_timesteps: 8000000  # Total steps d'entraînement
  num_envs: 30             # Environnements parallèles
  num_steps: 1024          # Steps par rollout
  num_minibatches: 32      # Minibatches pour update
  update_epochs: 10        # Époques d'optimisation
  seed: 1                  # Seed reproductibilité

ppo:
  lr: 0.0004              # Learning rate
  gamma: 0.995            # Discount factor
  gae_lambda: 0.98        # GAE lambda
  clip_coef: 0.2          # PPO clip coefficient
  ent_coef: 0.01          # Entropy coefficient
  vf_coef: 0.5            # Value function coefficient
  max_grad_norm: 0.5      # Gradient clipping

optimizer:
  eps: 0.00001            # Adam epsilon

environment:
  max_steps: 7000                    # Steps max par épisode
  use_random_corridor: true          # Corridors aléatoires
  corridor_xml: "corridor_yguel.xml" # Si use_random=false

network:
  robot_net_hidden: [32]           # MLP robot: 7→32
  history_net_hidden: [64, 32]     # MLP history: 48→64→32
  cnn_channels: [32, 64]           # CNN: 2→32→64
  cnn_kernel_size: 3               # Kernel 3×3
  cnn_stride: 2                    # Stride 2
  backbone_hidden: [64]            # Backbone: 128→64

logging:
  log_interval: 1          # Log toutes les N itérations
  save_interval: 5         # Save tous les N itérations
  render_interval: 10      # Render tous les N itérations
  batch_size_metrics: 20   # Taille batch pour métriques

curriculum:
  enabled: true            # Activer curriculum
  stabilization_steps: 20  # Steps stabilisation début
  
  bump_ratio_schedule:
    - phase: 1
      bump_ratio: 0.5
      distance_threshold: 10
    - phase: 2
      bump_ratio: 0.65
      distance_threshold: 12
    - phase: 3
      bump_ratio: 0.75
      distance_threshold: 65
    - phase: 4
      bump_ratio: 1.0
      distance_threshold: null

robot:
  max_steering_angle: 30.0  # Angle volant max (°)
  max_speed: 1.0           # Vitesse max (m/s)
  spawn_angle_max: 30.0    # Angle spawn max (°)

vision:
  cell_size: 0.2           # Taille cellule (m)
  vision_front: 5          # Vision devant (m)
  vision_behind: 2         # Vision derrière (m)
  vision_left: 2           # Vision gauche (m)
  vision_right: 2          # Vision droite (m)

history:
  history_interval: 20     # Sauver position tous les N steps
  history_length: 8        # Nombre positions passées

corridor:
  corridor_length: 200.0   # Longueur corridor (m)
  corridor_width: 3.0      # Largeur corridor (m)
  success_distance: 100.0  # Distance succès (m)

rewards:
  success_reward: 50.0      # Récompense succès
  failure_penalty: -5.0     # Pénalité échec
  progress_multiplier: 5.0  # Multiplicateur progression
  collision_penalty: -0.01  # Pénalité collision
  fell_threshold: 0.15      # Seuil chute (m)
  no_progress_check_interval: 750  # Check progrès (steps)
  no_progress_min_distance: 0.3    # Distance min (m)
  no_progress_penalty: -4.0        # Pénalité no progress
```

### Explication des paramètres

#### Section training
- `total_timesteps`: Nombre total de steps d'entraînement (8M = ~260 itérations avec 30 envs et 1024 steps)
- `num_envs`: Nombre d'environnements parallèles (plus = plus rapide mais plus de RAM/VRAM)
- `num_steps`: Steps par rollout avant update PPO (plus = plus stable mais moins d'updates)
- `num_minibatches`: Division du batch pour l'optimisation (32 = 960 steps par minibatch)
- `update_epochs`: Nombre de passes sur les données collectées (10 = bon compromis)
- `seed`: Seed pour reproductibilité (change pour varier l'entraînement)

#### Section ppo
- `lr`: Learning rate (0.0004 = bon départ, réduire si instable)
- `gamma`: Discount factor (0.995 = horizon long terme, proche de 1 = plus patient)
- `gae_lambda`: GAE lambda pour estimation des advantages (0.98 = bon compromis biais/variance)
- `clip_coef`: PPO clip coefficient (0.2 = standard, limite les changements de policy)
- `ent_coef`: Entropy coefficient (0.01 = encourage exploration, augmenter si bloqué)
- `vf_coef`: Value function coefficient (0.5 = poids de la loss du critic)
- `max_grad_norm`: Gradient clipping (0.5 = évite les explosions de gradients)

#### Section optimizer
- `eps`: Adam epsilon (1e-5 = stabilité numérique)

#### Section environment
- `max_steps`: Steps maximum par épisode (7000 = ~70s à 100Hz)
- `use_random_corridor`: true = génération aléatoire, false = utiliser corridor_xml
- `corridor_xml`: Fichier XML si use_random_corridor=false

#### Section network
- `robot_net_hidden`: Couches MLP pour état robot [32] = 7→32
- `history_net_hidden`: Couches MLP pour historique [64,32] = 48→64→32
- `cnn_channels`: Canaux CNN [32,64] = 2→32→64
- `cnn_kernel_size`: Taille kernel convolution (3 = 3×3)
- `cnn_stride`: Stride convolution (2 = réduit dimensions par 2)
- `backbone_hidden`: Couches MLP fusion [64] = 128→64

#### Section logging
- `log_interval`: Afficher logs toutes les N itérations (1 = chaque itération)
- `save_interval`: Sauvegarder checkpoint tous les N itérations (5 = tous les 5)
- `render_interval`: Render tous les N itérations (10 = rarement, coûteux)
- `batch_size_metrics`: Taille batch pour calcul métriques (20 épisodes)

#### Section curriculum
- `enabled`: Activer curriculum learning (true recommandé)
- `stabilization_steps`: Steps avant première vérification curriculum (20 itérations)
- `bump_ratio_schedule`: Liste des phases avec:
  - `phase`: Numéro de phase (1, 2, 3, 4)
  - `bump_ratio`: Ratio de bosses (0.5 = 50%, 1.0 = 100%)
  - `distance_threshold`: Distance moyenne pour passer à la phase suivante (null = phase finale)

#### Section robot
- `max_steering_angle`: Angle volant maximum en degrés (30° = réaliste pour voiture)
- `max_speed`: Vitesse maximum en m/s (1.0 = ~3.6 km/h, lent mais stable)
- `spawn_angle_max`: Angle spawn maximum en degrés (30° = variation initiale)

#### Section vision
- `cell_size`: Taille d'une cellule de la grille en mètres (0.2 = 20cm)
- `vision_front`: Distance vision devant en mètres (5 = voir loin devant)
- `vision_behind`: Distance vision derrière en mètres (2 = contexte arrière)
- `vision_left`: Distance vision gauche en mètres (2 = détecter murs)
- `vision_right`: Distance vision droite en mètres (2 = détecter murs)

#### Section history
- `history_interval`: Sauver position tous les N steps (20 = ~0.2s à 100Hz)
- `history_length`: Nombre de positions passées (8 = 8 frames × 6 valeurs = 48)

#### Section corridor
- `corridor_length`: Longueur corridor en mètres (200 = long)
- `corridor_width`: Largeur corridor en mètres (3 = étroit)
- `success_distance`: Distance pour succès en mètres (100 = objectif)

#### Section rewards
- `success_reward`: Récompense succès (50 = grosse récompense)
- `failure_penalty`: Pénalité échec (-5 = pénalité modérée)
- `progress_multiplier`: Multiplicateur progression (5.0 = encourage avancer)
- `collision_penalty`: Pénalité collision par step (-0.01 = petite pénalité continue)
- `fell_threshold`: Seuil chute en mètres (0.15 = robot tombé si z < 0.15m)
- `no_progress_check_interval`: Vérifier progrès tous les N steps (750 = ~7.5s)
- `no_progress_min_distance`: Distance minimum requise en mètres (0.3 = doit avancer)
- `no_progress_penalty`: Pénalité no progress (-4.0 = pénalité forte)

### Conseils de tuning

**Pour accélérer l'entraînement:**
- Augmenter `num_envs` (si RAM/VRAM suffisante)
- Augmenter `num_steps` (plus stable mais moins d'updates)
- Réduire `update_epochs` (moins de passes sur les données)

**Si l'entraînement est instable:**
- Réduire `lr` (0.0003 ou 0.0002)
- Augmenter `ent_coef` (0.02 ou 0.03 pour plus d'exploration)
- Réduire `num_steps` (updates plus fréquents)

**Si le robot n'explore pas assez:**
- Augmenter `ent_coef` (0.02 ou plus)
- Réduire `clip_coef` (0.1 pour plus de changements)
- Ajuster curriculum (commencer plus facile)

**Si le robot est trop prudent:**
- Réduire `collision_penalty` (moins pénalisant)
- Augmenter `progress_multiplier` (encourage vitesse)
- Augmenter `max_speed` (permet d'aller plus vite)

**Si le robot est trop agressif:**
- Augmenter `collision_penalty` (plus pénalisant)
- Réduire `progress_multiplier` (moins pressé)
- Augmenter `failure_penalty` (plus peur d'échouer)

**Spécifique au contrôle steering:**
- Ajuster `max_steering_angle` si le robot tourne trop/pas assez (20-45° typique)
- Vérifier paramètres physiques (wheelbase, track_width) dans corridor_env.py
- Le contrôle steering converge généralement plus vite que 4 roues indépendantes

---

## Interprétation des actions

### Action space

```python
action = [steering_angle_normalized, speed_normalized]
# Chaque valeur dans [-1, 1]
```

### Exemples d'actions

| Action | Interprétation | Résultat |
|--------|---------------|----------|
| `[0.0, 0.8]` | Tout droit, vitesse 80% | Avancer droit |
| `[1.0, 0.8]` | Gauche max (+30°), vitesse 80% | Tourner à gauche |
| `[-1.0, 0.8]` | Droite max (-30°), vitesse 80% | Tourner à droite |
| `[0.5, 0.5]` | Gauche 15°, vitesse 50% | Virage doux gauche |
| `[0.0, -0.5]` | Tout droit, reculer 50% | Reculer droit |
| `[0.0, 0.0]` | Pas de mouvement | Arrêt |

### Conversion en vitesses de roues

```python
# Exemple: action = [0.5, 0.8]
steering_angle = 0.5 * 30° = 15° = 0.262 rad
speed = 0.8 * 1.0 m/s = 0.8 m/s

# Calcul rayon de braquage
R = wheelbase / tan(steering_angle)
R = 0.5 / tan(0.262) = 1.88 m

# Vitesses linéaires
v_left = speed * (R - track_width/2) / R
v_right = speed * (R + track_width/2) / R

# Vitesses angulaires (rad/s)
ω_left = v_left / wheel_radius
ω_right = v_right / wheel_radius

# Résultat: [ω_FL, ω_FR, ω_RL, ω_RR]
```

---

## Métriques trackées

### Par batch (training_metrics.csv)

- `batch_num`: Numéro du batch
- `episode_end`: Dernier épisode du batch
- `episodes_range`: Range d'épisodes (ex: "1-20")
- `global_step`: Steps totaux depuis début
- `mean_return`: Return moyen du batch
- `mean_distance`: Distance moyenne du batch
- `mean_survival`: Survie moyenne du batch
- `success_rate`: Taux de succès du batch
- `current_phase`: Phase curriculum actuelle
- `random_percentage`: % corridors aléatoires (fixe à 1.0)
- `bump_ratio`: Ratio de bosses actuel

### Par itération (logs console)

- Return: Recent (mean ± std) | Max
- Distance: Recent (mean ± std) | Max
- Survival: Recent (mean ± std)
- Terminations: fell, flipped, no_progress counts

### Par épisode (episodes_log.txt)

```
Episode 123: fell | Steps: 456 | Distance: 12.34m | Reward: 45.6 | Corridor: holes+50%bumps+random | Seed: 7890
```

---

## Curriculum Learning

### Progression des phases

```
Phase 1: holes + 50% bumps
  ↓ (distance >= 10m)
Phase 2: holes + 65% bumps
  ↓ (distance >= 12m)
Phase 3: holes + 75% bumps
  ↓ (distance >= 65m)
Phase 4: holes + 100% bumps
  (pas de seuil, phase finale)
```

### Vérification à chaque itération

```
CURRICULUM CHECK
Iteration mean distance: 11.5m
Current phase: 1 (threshold: 10.0m)
✓ Distance >= threshold → Ready for next phase
```

### Rollback automatique (--rollback)

Si la performance régresse (distance < dernière sauvegarde):
1. Charger le dernier checkpoint
2. Nettoyer les fichiers temp
3. Continuer l'entraînement

Sans `--rollback`: continue sans sauvegarder.

---

## Exemples d'utilisation

### Entraînement

```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Entraînement standard avec rollback
python train_ppo.py --rollback

# Nouveau démarrage
python train_ppo.py --fresh-start

# Override paramètres
python train_ppo.py --lr 0.0003 --num-envs 16 --seed 42
```

### Test

```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Test avec visualisation complète
python test_ppo.py --render --show-vision --num-episodes 5

# Test sur corridor difficile
python test_ppo.py --render --bump-ratio 1.0

# Test checkpoint spécifique
python test_ppo.py --model models/ppo_corridor_50.pth --render
```

### Visualisation

```bash
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Visualiser vision CNN
python visualize_corridor_map.py --render

# Position spécifique
python visualize_corridor_map.py --x 50 --y 0.5 --angle 15
```

## Fichiers générés

```
models/
├── ppo_corridor_5.pth          # Checkpoint itération 5
├── ppo_corridor_10.pth         # Checkpoint itération 10
├── ...
├── training_metrics.csv        # Métriques complètes
├── training_curves_5.png       # Graphiques itération 5
├── training_curves_10.png      # Graphiques itération 10
├── training_curves.png         # Graphiques finaux
├── episodes_log.txt            # Log tous épisodes
├── iteration_summary.json      # Dernière itération sauvegardée
├── temp_training_metrics.csv   # Métriques temporaires (flush)
└── temp_episodes_log.txt       # Logs temporaires (flush)
```

---

## Avantages du contrôle steering

### Avantages

1. **Action space plus petit**: 2D au lieu de 4D
   - Moins de dimensions à explorer
   - Convergence potentiellement plus rapide
   - Moins de variance dans les gradients

2. **Contrôle plus naturel**: Comme une vraie voiture
   - Steering + vitesse = intuition humaine
   - Contraintes physiques respectées automatiquement
   - Comportement plus prévisible

3. **Moins de degrés de liberté**: 
   - Impossible de faire des mouvements "impossibles"
   - Comportement plus cohérent et stable
   - Moins de risque de divergence

4. **Apprentissage facilité**:
   - Espace d'actions plus structuré
   - Corrélations naturelles entre actions
   - Meilleure généralisation

### Inconvénients

1. **Moins de flexibilité**: 
   - Impossible de tourner sur place
   - Impossible de faire des mouvements latéraux
   - Rayon de braquage minimum

2. **Dépendance à la conversion**:
   - Qualité dépend de `steer_angle_to_wheel_speeds()`
   - Paramètres physiques doivent être corrects (wheelbase, track_width)
   - Erreurs de modélisation peuvent affecter performance

3. **Contraintes mécaniques**:
   - Limité par max_steering_angle (30°)
   - Comportement à basse vitesse peut être instable
   - Moins adapté aux manœuvres complexes

---

## Troubleshooting

### Problème: tkinter non trouvé

```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk
pip install pillow
```

### Problème: CUDA out of memory

Réduire `num_envs` dans config.yaml:
```yaml
training:
  num_envs: 16  # Au lieu de 30
```

### Problème: Entraînement ne progresse pas

1. Vérifier curriculum: phase trop difficile?
2. Réduire learning rate: `--lr 0.0002`
3. Augmenter entropy: `ent_coef: 0.02` dans config
4. Utiliser `--fresh-start` pour recommencer

### Problème: Robot tourne en rond

1. Vérifier paramètres physiques (wheelbase, track_width)
2. Vérifier max_steering_angle (30° par défaut)
3. Augmenter entropy pour plus d'exploration
4. Vérifier que la conversion steering→wheels est correcte

---

## Notes importantes

- Action space: 2 dimensions `[steering_angle, speed]` dans `[-1, 1]`
- Conversion automatique vers 4 vitesses de roues dans l'environnement
- Les checkpoints contiennent tout l'état (model, optimizer, iteration, metrics)
- Resume automatique depuis le dernier checkpoint
- Les fichiers temp sont flushés à chaque sauvegarde
- Le curriculum progresse automatiquement selon la distance moyenne
- Paramètres physiques critiques: wheelbase, track_width, wheel_radius

---

## Comparaison avec ppo_no_steer

### Différences principales

| Aspect | ppo_steer | ppo_no_steer |
|--------|-----------|--------------|
| Action space | 2D `[steering, speed]` | 4D `[wheel_FL, FR, RL, RR]` |
| Actor head | `Linear(64, 2)` | `Linear(64, 4)` |
| Contrôle | Haut niveau (voiture) | Bas niveau (roues) |
| Conversion | Automatique dans env | Pas de conversion |
| Flexibilité | Limitée (contraintes physiques) | Maximale |
| Apprentissage | Plus simple (2D) | Plus complexe (4D) |
| Checkpoints | Non compatibles | Non compatibles |

### Quand utiliser ppo_steer?

- Comportement type voiture souhaité
- Apprentissage plus rapide prioritaire
- Contraintes physiques réalistes importantes
- Action space réduit préféré

### Quand utiliser ppo_no_steer?

- Manœuvres complexes nécessaires
- Contrôle fin des roues requis
- Flexibilité maximale prioritaire
- Mouvements non conventionnels possibles

---

## Références

- Configuration complète: `config.yaml`
- Environnement: `corridor_env.py`
- Agent PPO: `train_ppo.py`
- Conversion steering: fonction `steer_angle_to_wheel_speeds()` dans `corridor_env.py`
- Générateur de corridors: `corridor_generator_similar.py`

---
