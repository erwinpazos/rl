# PPO Steer - Contrôle par volant et vitesse

Version avec contrôle haut niveau par volant et vitesse (action space: 2 dimensions).

**Emplacement**: `mujoco/workspace/ppo_steer/`

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Scripts disponibles](#scripts-disponibles)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Architecture du réseau](#architecture-du-réseau)
- [Configuration](#configuration)

---

## 🎯 Vue d'ensemble

Cette version contrôle le robot comme une voiture:
- **Action space**: `Box(-1.0, 1.0, (2,))` 
- **Actions**: `[steering_angle, speed]`
  - `steering_angle`: Angle de volant normalisé [-1, 1] → [-30°, +30°]
  - `speed`: Vitesse normalisée [-1, 1] → [-max_speed, +max_speed]
- Conversion automatique vers vitesses des 4 roues via `steer_angle_to_wheel_speeds()`
- Plus naturel et plus simple à apprendre

**Prérequis**: Environnement Docker lancé (voir [README principal](../../../README.md))

---

## 🔄 Conversion steering → wheel speeds

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

## 📜 Scripts disponibles

### 1. train_ppo.py - Entraînement

Identique à ppo_no_steer mais avec action space 2D.

**Usage:**
```bash
# Dans l'environnement Docker
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

# Nouveau démarrage
python train_ppo.py --fresh-start

# Override paramètres
python train_ppo.py --lr 0.0003 --num-envs 16
```

# Nouveau démarrage
python train_ppo.py --fresh-start

# Override paramètres
python train_ppo.py --lr 0.0003 --num-envs 16
```

**Différence clé avec ppo_no_steer:**
- Action space: 2D au lieu de 4D
- Actor head: `Linear(64, 2)` au lieu de `Linear(64, 4)`
- Conversion automatique dans `corridor_env.py`

---

### 2. test_ppo.py - Test d'un modèle

Identique à ppo_no_steer.

**Usage:**
```bash
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

# Test avec rendu et vision
python test_ppo.py --render --show-vision --num-episodes 5

# Test sur corridor difficile
python test_ppo.py --render --bump-ratio 1.0

# Test checkpoint spécifique
python test_ppo.py --model models/ppo_corridor_50.pth --render
```

---

### 3. visualize_corridor_map.py - Visualisation CNN

Identique à ppo_no_steer.

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
# Dans l'environnement Docker
cd ~/workspace/ppo_steer

# Visualisation aléatoire
python visualize_corridor_map.py

# Position spécifique avec rendu
python visualize_corridor_map.py --x 50 --y 0.5 --angle 15 --render
```

---

### 4. plot_metrics.py - Graphiques de métriques

Identique à ppo_no_steer.

**Usage:**
```bash
python plot_metrics.py [CSV_FILE]
```

---

### 5. corridor_env.py - Environnement Gymnasium

**Différence clé:** Action space et conversion

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
    
    # Appliquer aux actuateurs
    self.data.ctrl[:] = wheel_speeds
```

---

## 🔄 Pipeline d'entraînement complet

Le pipeline est identique à ppo_no_steer, seule la dimension de l'action change.

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

**Différence principale:** 
- Agent produit 2 actions au lieu de 4
- Environnement convertit automatiquement en 4 vitesses de roues

---

## 🧠 Architecture du réseau

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

## ⚙️ Configuration (config.yaml)

Configuration identique à ppo_no_steer. Voir [ppo_no_steer/README.md](../ppo_no_steer/README.md#configuration-configyaml) pour détails complets.

**Paramètres clés:**
```yaml
training:
  total_timesteps: 8000000
  num_envs: 30
  num_steps: 1024

ppo:
  lr: 0.0004
  gamma: 0.995
  gae_lambda: 0.98

robot:
  max_steering_angle: 30.0  # Angle max en degrés
  max_speed: 1.0           # Vitesse max en m/s

curriculum:
  enabled: true
  bump_ratio_schedule:
    - phase: 1
      bump_ratio: 0.5
      distance_threshold: 10
    # ... (voir config.yaml)
```

---

## 🎮 Interprétation des actions

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

## 📊 Métriques trackées

Identiques à ppo_no_steer. Voir [ppo_no_steer/README.md](../ppo_no_steer/README.md#métriques-trackées).

---

## 🎓 Curriculum Learning

Identique à ppo_no_steer. Voir [ppo_no_steer/README.md](../ppo_no_steer/README.md#curriculum-learning).

---

## 🚀 Exemples d'utilisation

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

---

## 🔍 Avantages vs ppo_no_steer

### Avantages du steering

1. **Action space plus petit**: 2D au lieu de 4D
   - Moins de dimensions à explorer
   - Convergence potentiellement plus rapide

2. **Contrôle plus naturel**: Comme une vraie voiture
   - Steering + vitesse = intuition humaine
   - Contraintes physiques respectées automatiquement

3. **Moins de degrés de liberté**: 
   - Impossible de faire des mouvements "impossibles"
   - Comportement plus cohérent

### Inconvénients

1. **Moins de flexibilité**: 
   - Impossible de tourner sur place
   - Impossible de faire des mouvements latéraux

2. **Dépendance à la conversion**:
   - Qualité dépend de `steer_angle_to_wheel_speeds()`
   - Paramètres physiques doivent être corrects

---

## 📝 Notes importantes

- L'architecture du réseau est identique à ppo_no_steer sauf l'actor head
- La conversion steering→wheels est faite dans l'environnement, pas dans l'agent
- Les checkpoints ne sont PAS compatibles entre no_steer et steer (dimensions différentes)
- Le curriculum et les métriques fonctionnent exactement pareil
- Les fichiers générés ont la même structure

---

## 🔄 Migration depuis ppo_no_steer

Pour migrer un entraînement de no_steer vers steer:

1. **Impossible de réutiliser les checkpoints** (dimensions différentes)
2. **Possible de réutiliser config.yaml** (compatible)
3. **Possible de réutiliser les corridors XML** (identiques)
4. **Recommandé**: Commencer un nouvel entraînement avec `--fresh-start`

---

## 📚 Voir aussi

- [README général](../README.md): Vue d'ensemble du projet
- [ppo_no_steer/README.md](../ppo_no_steer/README.md): Version 4 roues indépendantes
- Configuration détaillée dans `config.yaml`
- Architecture détaillée dans `corridor_env.py`

---
