# PPO Small Robot Steer - Navigation avec Contrôle par Volant

## Vue d'ensemble

Ce projet implémente un agent PPO (Proximal Policy Optimization) pour naviguer un robot 4 roues dans des corridors avec **contrôle par volant** (steering + speed) au lieu de 4 roues indépendantes.

**Innovation clé**: 
- **Contrôle simplifié**: 2 actions (angle volant + vitesse) au lieu de 4 roues
- **Vision 2 canaux**: Grilles binaires séparées pour obstacles et trous
- **Système de récompenses simplifié**: Pas de guidage artificiel, apprentissage naturel
- **Configuration JSON**: Tous les paramètres externalisés et modifiables

## Architecture du Système

### 1. Environnement (`corridor_env.py`)

**Concept**: Robot 4 roues naviguant dans un corridor avec **contrôle par volant** (steering + speed).

**Contrôle par volant**:
- **Action 0**: Angle de volant (±1.0 → ±30°)
- **Action 1**: Vitesse (±1.0 → ±1.0 m/s)
- **Conversion automatique**: Steering + speed → vitesses des 4 roues via cinématique

**Observation (3655 valeurs)** - **Vision 2 canaux + historique simplifié**:

**1. État robot (7 valeurs)**:
- Position: `(x, y, z)` du centre du robot
- Vitesse: `(vx, vy, vz)` du centre de masse  
- Angle: orientation du robot (radians)

**2. Historique positions (48 valeurs)** - **Anticipation sans bounding box**:
- **Structure**: 8 frames × 6 valeurs = 48 total
- **Contenu par frame**: 3 positions + 3 vitesses (plus de bounding box)
- **Fréquence**: Sauvegarde toutes les 10 steps
- **Format relatif**: Différences par rapport à l'état actuel
- **Objectif**: Apprendre la dynamique sans redondance

**3. Grille vision 2 canaux (3600 valeurs)** - **Perception binaire optimisée**:
- **Dimensions**: 60 lignes × 30 colonnes × 2 canaux = 3600 valeurs
- **Canal 0 - Obstacles**: 1.0 = bump OU murs latéraux, 0.0 = navigable
- **Canal 1 - Trous**: 1.0 = trou OU extérieur avant/arrière, 0.0 = navigable
- **Sol navigable**: Les deux canaux à 0.0
- **Logique physique**:
  - Côtés du couloir = murs infinis (obstacles)
  - Devant/derrière du couloir = vide (trous, robot tombe)

**Vision ego-centrique**:
- **Couverture**: 0.8m derrière + 5.2m devant × 3m largeur
- **Robot fixe**: Toujours à la ligne 8, colonne 15
- **Rotation**: Grille tourne avec le robot (comme caméra embarquée)

**Actions (2 valeurs)** - **Contrôle par volant**:
- `action[0]`: Angle de volant (±1.0 → ±30°)
- `action[1]`: Vitesse (±1.0 → ±1.0 m/s)

**Conversion en vitesses roues**:
```python
steering_angle = action[0] * 30.0  # degrés
speed = action[1] * 1.0           # m/s

# Conversion en yaw rate puis vitesses roues
yaw_rate = steering_angle * (max_yaw_rate / max_steering_angle)
w_left = (speed - yaw_rate * track_width/2) / wheel_radius
w_right = (speed + yaw_rate * track_width/2) / wheel_radius
```

**Récompenses simplifiées**:
- **Succès**: +100 (x ≥ 100m)
- **Échecs**: -10 (tombé, retourné, collision)
- **Progression**: +10 × delta_x (récompense continue pour avancer)
- **Pas de bonus complexes**: Évite le reward hacking

**Terminaisons** - **Détection précise et naturelle**:

**1. Fell (tombé dans trou)**:
```python
if z < 0.15:  # Seuil de hauteur critique
    return -10.0, True, {'reason': 'fell'}
```
- **Logique**: Robot tombe quand il n'y a pas de géométrie sous lui
- **Seuil**: 0.15m sous la hauteur normale (0.45m spawn)
- **Inclut**: Tomber dans un trou OU sortir du couloir (pas de sol à l'extérieur)

**2. Flipped (robot retourné)**:
```python
quat = self.data.qpos[3:7]
up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)  # Composante Z du vecteur "up"
if up_z < 0:  # Robot à l'envers
    return -10.0, True, {'reason': 'flipped'}
```
- **Logique**: Calcul du vecteur "up" depuis le quaternion d'orientation
- **Robuste**: Détecte retournement quelle que soit l'orientation XY

**3. Success (objectif atteint)**:
```python
if x >= 100.0:  # Fin du couloir
    return 100.0, True, {'reason': 'success'}
```

**4. Truncated (timeout)**:
- **Limite**: 3000 steps par épisode
- **Calcul**: ~10 minutes à 30 FPS, assez pour parcourir 100m à vitesse normale

**Note importante**: Pas de terminaison artificielle pour sortie du couloir ! Le robot apprend naturellement en tombant dans le vide (pas de géométrie hors du couloir).

### 2. Configuration JSON (`config.json`)

**Système de configuration externalisé** pour tous les paramètres d'entraînement:

```json
{
  "training": {
    "total_timesteps": 8000000,
    "num_envs": 32,
    "num_steps": 1024
  },
  "ppo": {
    "update_epochs": 10,
    "num_minibatches": 32,
    "clip_coef": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5
  },
  "optimizer": {
    "lr": 5e-4,
    "eps": 1e-5
  },
  "environment": {
    "max_steps": 3000,
    "corridor_xml": null
  },
  "network": {
    "robot_state_dim": 7,
    "history_dim": 48,
    "grid_dim": 3600,
    "cnn_channels": [32, 64, 128]
  }
}
```

**Outils de configuration**:
- `config.json` - Configuration principale
- `config_example.json` - Version documentée avec commentaires
- `config_debug.json` / `config_production.json` - Presets
- `validate_config.py` - Validation et comparaison
- `generate_config.py` - Génération automatique

### 3. Architecture Réseau (`train_ppo.py`)

**Réseau multi-branches optimisé pour vision 2 canaux**:

```
Observation (3655) → 3 branches parallèles → Fusion → Actor/Critic

Branch 1: Robot State (7) → MLP(64→64) → 64
Branch 2: History (48) → MLP(128→64→32) → 32  
Branch 3: Grid 2-channel (60×30×2) → CNN → 128

Fusion: Concat(64+32+128) → MLP(128→64) → 64
Output: Actor(64→2) + Critic(64→1)
```

**CNN pour grille 2 canaux**:
```
60×30×2 → Conv2d(32,3×3,s=2) → 30×15×32
        → Conv2d(64,3×3,s=2) → 15×8×64  
        → Conv2d(128,3×3,s=2) → 8×4×128
        → Flatten → Linear(4096→128) → 128
```

**Historique simplifié** - **Sans bounding box**:
- **Fréquence**: sauvegarde toutes les 10 steps
- **Longueur**: 8 frames (80 steps d'historique total)
- **Contenu**: 3 positions + 3 vitesses par frame (6 valeurs)
- **Format**: coordonnées relatives à la position actuelle
- **Objectif**: Apprendre la dynamique sans redondance

### 4. Entraînement PPO

**Configuration via JSON**:
- Tous les hyperparamètres externalisés
- Presets debug/production disponibles
- Override possible via ligne de commande

**Hyperparamètres par défaut**:
- **Environnements parallèles**: 32
- **Steps par rollout**: 1024
- **Batch size**: 32,768 (32×1024)
- **Minibatches**: 32 (1024 samples chacun)
- **Update epochs**: 10
- **Learning rate**: 5e-4
- **Gamma**: 0.995
- **GAE lambda**: 0.98
- **Clip coefficient**: 0.2

**Optimisations**:
- AsyncVectorEnv pour parallélisation
- Buffers GPU pour vitesse
- Sauvegarde automatique tous les 10 itérations
- Reprise d'entraînement automatique
- Métriques par batch de 20 épisodes

## Utilisation

### Entraînement

```bash
# Entraînement standard avec config.json
python train_ppo.py

# Avec configuration spécifique
python train_ppo.py --config config_debug.json

# Override de paramètres
python train_ppo.py --timesteps 20000000 --num-envs 64

# Nouveau démarrage (ignorer modèles existants)
python train_ppo.py --fresh-start
```

### Test

```bash
# Test avec rendu
python test_ppo.py --render

# Test avec corridor spécifique
python test_ppo.py --render --corridor corridor_100.xml

# Plus d'épisodes
python test_ppo.py --episodes 10
```

### Visualisation

**Outil de débogage** pour comprendre la vision 2 canaux:

```bash
# Vision 2 canaux du robot
python visualize_corridor_map.py

# Position spécifique
python visualize_corridor_map.py --x 50 --y 0.5 --angle 10

# Test positions multiples
python visualize_corridor_map.py --test-multiple
```

**Exemple de sortie** (vision 2 canaux):

```
GRILLE 2 CANAUX - 60×30×2:

CANAL 0 - OBSTACLES (bumps + murs latéraux):
    012345678901234567890123456789
  0|XXXXXXXXXX████████████XXXXXXXX  
  1|XXXXXXXXXX████████████XXXXXXXX  
  8|XXXXXXXXXX██████🤖█████XXXXXXXX  ← Robot
  
CANAL 1 - TROUS (holes + extérieur avant/arrière):
    012345678901234567890123456789
  0|░░░░░░░░░░████████████░░░░░░░░░░
  1|░░░░░░░░░░████████████░░░░░░░░░░
  8|░░░░░░░░░░██████🤖█████░░░░░░░░░░

GRILLE COMBINÉE (perception finale):
  X = obstacle (canal 0)
  ░ = trou (canal 1)  
  █ = sol navigable (les deux canaux à 0)
  🤖 = robot (toujours centré)
```

## Fichiers

### Core
- **`corridor_env.py`**: Environnement avec contrôle par volant et vision 2 canaux
- **`train_ppo.py`**: Entraînement PPO avec configuration JSON
- **`test_ppo.py`**: Test et évaluation des modèles
- **`visualize_corridor_map.py`**: Visualisation de la vision 2 canaux

### Configuration
- **`config.json`**: Configuration principale
- **`config_example.json`**: Version documentée
- **`config_debug.json`** / **`config_production.json`**: Presets
- **`validate_config.py`**: Validation des configurations
- **`generate_config.py`**: Génération automatique de configs

### Modèles
- **`four_wheels_robot.xml`**: Modèle MuJoCo du robot
- **`corridor_100.xml`**: Corridor de test fixe

### Documentation
- **`REWARD_SYSTEM.md`**: Documentation du système de récompenses simplifié
- **`CONFIG_README.md`**: Guide d'utilisation des configurations

## Innovations Clés

### 1. Contrôle par Volant
**Problème**: Contrôle 4 roues indépendantes complexe et peu naturel
**Solution**: 2 actions (steering + speed) converties automatiquement en vitesses roues
**Avantage**: Contrôle plus intuitif, espace d'actions réduit

### 2. Vision 2 Canaux Binaires
**Problème**: Grille unique avec valeurs continues difficile à apprendre
**Solution**: 2 grilles binaires séparées (obstacles/trous) avec logique physique claire
**Avantage**: CNN plus efficace, représentation plus claire

### 3. Système de Récompenses Simplifié
**Problème**: Récompenses complexes causent du reward hacking
**Solution**: Seulement succès/échecs/progression, pas de guidage artificiel
**Avantage**: Apprentissage plus naturel et robuste

### 4. Configuration JSON Externalisée
**Problème**: Paramètres hardcodés difficiles à ajuster
**Solution**: Tous les paramètres dans des fichiers JSON avec outils de validation
**Avantage**: Expérimentation facile, reproductibilité, presets

### 5. Observation Sans Bounding Box
**Problème**: Bounding box redondante en vision ego-centrique
**Solution**: Historique simplifié (positions + vitesses uniquement)
**Avantage**: Observation plus compacte (3655 vs 1902 valeurs), moins de redondance

## Défis Techniques Résolus

### 1. Contrôle Simplifié mais Efficace
**Défi**: Réduire l'espace d'actions sans perdre en capacité de manœuvre
**Solution**: Conversion steering/speed → vitesses roues via cinématique arcade
**Résultat**: Actions intuitives avec toutes les capacités de mouvement

### 2. Vision Binaire Optimisée
**Défi**: Représentation efficace de l'environnement pour CNN
**Solution**: 2 canaux binaires avec logique physique (obstacles vs trous)
**Résultat**: Apprentissage plus rapide, représentation plus claire

### 3. Récompenses Sans Biais
**Défi**: Éviter le reward hacking tout en guidant l'apprentissage
**Solution**: Récompenses minimales (succès/échecs/progression uniquement)
**Résultat**: Apprentissage naturel sans comportements artificiels

### 4. Configuration Flexible
**Défi**: Permettre l'expérimentation facile sans recompilation
**Solution**: Système JSON complet avec validation et presets
**Résultat**: Itération rapide, reproductibilité, partage de configs

## Résultats Attendus

**Métriques de succès**:
- Distance moyenne > 50m
- Taux de succès > 10% (atteindre 100m)
- Contrôle fluide avec steering/speed

**Comportements recherchés**:
- Navigation fluide avec contrôle par volant
- Évitement efficace obstacles/trous
- Utilisation optimale de la vision 2 canaux
- Adaptation rapide aux configurations

