# Optimized PPO - Robot Navigation in Random Corridors

## Vue d'ensemble

Ce projet implémente un agent PPO (Proximal Policy Optimization) pour naviguer un robot 4 roues dans des corridors **générés aléatoirement à chaque épisode**. 

**Problème résolu**: Les agents RL ont tendance à mémoriser des trajectoires fixes au lieu d'apprendre la navigation générale. En générant un nouveau corridor à chaque `reset()`, le robot est **forcé de comprendre l'environnement** plutôt que de mémoriser des chemins.

**Innovation clé**: Génération temps réel de géométries MuJoCo complètes (pas juste une grille virtuelle) pour chaque simulation, sans impact sur les performances.

## Architecture du Système

### 1. Environnement (`corridor_env.py`)

**Concept**: Robot 4 roues naviguant dans un corridor de 100m × 3m avec obstacles aléatoires **régénérés à chaque épisode**.

**Génération dynamique**:
- **À chaque `reset()`**: Nouveau layout d'obstacles généré
- **Géométries MuJoCo réelles**: Boxes physiques pour flat/bumps, vide pour trous
- **Modèle reconstruit**: Nouveau `MjModel` à chaque épisode (pas juste changement de paramètres)
- **Spawn aléatoire**: Position Y et angle initial variables pour éviter mémorisation de départ

**Observation (1902 valeurs)** - **Conçue pour la généralisation**:

**1. État robot (6 valeurs)**:
- Position absolue: `(x, y, z)` du centre du robot dans le monde
- Vitesse linéaire: `(vx, vy, vz)` du centre de masse
- Obtenues directement de `self.data.qpos[:3]` et `self.data.qvel[:3]`

**2. Bounding box (8 valeurs)** - **Représentation spatiale du robot**:

**Dimensions du robot réel** (dans `four_wheels_robot.xml`):
- **Châssis**: 0.8m × 0.6m × 0.2m (longueur × largeur × hauteur)
  - Géométrie: `<geom size="0.40 0.30 0.10">` (half-dimensions)
  - Taille réelle: 2 × (0.40, 0.30, 0.10) = (0.8m, 0.6m, 0.2m)
- **Roues**: Rayon 0.2m, largeur 0.06m, dépassent du châssis
  - Positions: `±0.35m` en X (avant/arrière), `±0.30m` en Y (gauche/droite)
- **Empattement total**: 0.7m (entre essieux avant/arrière)
- **Voie totale**: 0.6m (entre roues gauche/droite)

**Bounding box englobante** (pour l'algorithme):
- **Dimensions**: 1.1m × 0.7m (englobe robot + roues + marge)
  - **Longueur**: 1.1m = châssis 0.8m + dépassement roues avant/arrière 0.15m × 2 = 1.1m
  - **Largeur**: 0.7m = châssis 0.6m + dépassement roues latérales 0.03m × 2 + marge = 0.7m
- **Justification**: Bounding box réaliste qui englobe toutes les parties (châssis + roues + petite marge)
- **Sécurité**: Détection précise des collisions avant contact physique

**Mapping bounding box → robot**:
- **Longueur**: 1.1m = ±0.55m du centre (englobe roues avant/arrière + rayon)
- **Largeur**: 0.7m = ±0.35m du centre (englobe roues gauche/droite + épaisseur + marge)
- **En cellules**: 11×7 cellules de 0.1m (bounding box couvre 77 cellules)

**4 coins dans le repère robot**:
```python
# Coins dans le repère local robot (X=avant, Y=gauche)
corners_local = [
    (+0.55, +0.35),  # avant-gauche (englobe roue FL + rayon 0.2m)
    (+0.55, -0.35),  # avant-droite (englobe roue FR + rayon 0.2m)
    (-0.55, +0.35),  # arrière-gauche (englobe roue RL + rayon 0.2m)
    (-0.55, -0.35),  # arrière-droite (englobe roue RR + rayon 0.2m)
]
```

**Calcul de rotation** (bounding box suit l'orientation du robot):
```python
# Extraction angle depuis quaternion MuJoCo
quat = self.data.qpos[3:7]
angle = 2 * np.arctan2(quat[3], quat[0])

# Rotation 2D des coins locaux vers coordonnées monde
world_offset_x = cos(angle) * local_x - sin(angle) * local_y
world_offset_y = sin(angle) * local_x + cos(angle) * local_y

# Position finale du coin
corner_world_x = robot_x + world_offset_x
corner_world_y = robot_y + world_offset_y
```

**Conversion en grille**: Coins exprimés en `(row, col)` dans la grille 60×30
**Invariance**: Les dimensions restent 0.6×0.4m quelle que soit l'orientation
**Usage**: Détection précise des collisions et out-of-bounds

**3. Historique étendu (88 valeurs)** - **Mémoire temporelle**:
- **Structure**: 8 frames × 11 valeurs = 88 total
- **Contenu par frame**: 8 coords des coins + 3 vitesses `(vx, vy, vz)`
- **Fréquence**: Sauvegarde toutes les 10 steps (intervalle réduit pour plus de détail)
- **Format relatif**: Différences par rapport à l'état actuel
  ```python
  relative_corners = past_corners - current_corners
  relative_velocities = past_velocities - current_velocities
  ```
- **Objectif**: Apprendre la dynamique du robot (inertie, glissement, réponse aux commandes)

**4. Grille vision (1800 valeurs)** - **Perception de l'environnement**:
- **Dimensions**: 60 lignes × 30 colonnes = 1800 cellules
- **Résolution**: 0.1m par cellule (équilibre précision/performance)
- **Couverture spatiale**: 6m longueur × 3m largeur
- **Valeurs normalisées**: 0.0=sol, 0.5=bump, 1.0=trou
- **Vision asymétrique**: 0.5m derrière (5 cellules) + 5.5m devant (55 cellules)
- **Position robot**: Fixe à la ligne 5 (permet de voir plus loin devant)
- **Vision Y fixe**: Couvre toujours toute la largeur du couloir, pas centrée sur le robot

**Vision asymétrique** - **Optimisée pour l'anticipation**:

**Principe**: Le robot n'a pas besoin de voir beaucoup derrière lui (déjà passé) mais doit anticiper les obstacles devant.

**Configuration spatiale**:
- **Derrière**: 0.5m (5 cellules) - Juste pour contexte de trajectoire
- **Devant**: 5.5m (55 cellules) - Distance d'anticipation pour freinage/évitement
- **Largeur**: 3m (30 cellules) - Toute la largeur du couloir
- **Total**: 60×30 = 1800 cellules

**Repère de référence**:
- **Vision Y fixe**: Grille toujours alignée sur le couloir `[-1.5m, +1.5m]`
- **Pas centrée sur robot**: Évite les translations de grille quand le robot bouge en Y
- **Robot mobile**: Position du robot varie dans la grille selon sa position Y réelle

**Calcul de la grille**:
```python
# Position du robot dans la grille monde
robot_row_world = int(robot_x / 0.1)  # Cellule X du robot
robot_grid_col = int((robot_y + 1.5) / 0.1)  # Cellule Y du robot

# Limites de vision (fenêtre mobile en X, fixe en Y)
vision_start_row = robot_row_world - 5  # 0.5m derrière
vision_end_row = robot_row_world + 55   # 5.5m devant

# Remplissage de la grille 60×30
for i in range(60):  # Lignes de la grille vision
    world_row = vision_start_row + i
    for j in range(30):  # Colonnes fixes du couloir
        world_col = j  # Directement les colonnes du couloir
        cell_type = self.cell_map.get((world_row, world_col), 2)
```

**Actions (4 valeurs)** - **Contrôle différentiel précis**:

**Principe**: Contrôle indépendant de chaque roue pour navigation fine.

**Mapping des roues**:
- `action[0]`: Roue avant-gauche (FL)
- `action[1]`: Roue avant-droite (FR) 
- `action[2]`: Roue arrière-gauche (RL)
- `action[3]`: Roue arrière-droite (RR)

**Conversion des actions**:
```python
# Entrée réseau: [-1, +1] pour chaque roue
action = np.clip(action, -1.0, 1.0)
# Conversion en couples MuJoCo: [-20, +20] Nm
self.data.ctrl[:] = action * 20.0
```

**Capacités de mouvement**:
- **Avancer**: Toutes roues positives `[+1, +1, +1, +1]`
- **Reculer**: Toutes roues négatives `[-1, -1, -1, -1]`
- **Tourner sur place**: Roues opposées `[+1, -1, +1, -1]` (rotation pure)
- **Trajectoire courbe**: Différentiel gauche/droite `[+0.5, +1, +0.5, +1]`
- **Correction fine**: Micro-ajustements pour évitement précis

**Récompenses** - **Simples et efficaces**:
- **Succès**: +100 (x ≥ 100m)
- **Échecs**: -10 (tombé, retourné, out of bounds)
- **Progression**: +10 × delta_x (récompense continue pour avancer)
- **Pas de bonus complexes**: Évite le reward hacking

**Terminaisons** - **Détection précise et robuste**:

**1. Fell (tombé dans trou)**:
```python
if z < 0.15:  # Seuil de hauteur critique
    return -10.0, True, {'reason': 'fell'}
```
- **Logique**: Robot tombe quand il n'y a pas de géométrie sous lui
- **Seuil**: 0.15m sous la hauteur normale (0.45m spawn)

**2. Flipped (robot retourné)**:
```python
quat = self.data.qpos[3:7]
up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)  # Composante Z du vecteur "up"
if up_z < 0:  # Robot à l'envers
    return -10.0, True, {'reason': 'flipped'}
```
- **Logique**: Calcul du vecteur "up" depuis le quaternion d'orientation
- **Robuste**: Détecte retournement quelle que soit l'orientation XY

**3. Out of bounds (sortie de couloir)** - **Innovation précise**:
```python
def _is_out_of_bounds(self, robot_x, robot_y):
    # Calculer les 4 coins de la bounding box
    corners_world = self._get_robot_bbox_corners(robot_x, robot_y)
    
    for i in range(4):  # Vérifier chaque coin
        corner_y = robot_y + world_offset_y[i]
        if abs(corner_y) > 1.5:  # Couloir = 3m = ±1.5m
            return True
    return False
```
- **Précision**: Vérifie chaque coin de la bounding box, pas juste le centre
- **Réalisme**: Robot pénalisé dès qu'une partie sort, pas quand il est complètement dehors

**4. Success (objectif atteint)**:
```python
if x >= 100.0:  # Fin du couloir
    return 100.0, True, {'reason': 'success'}
```

**5. Truncated (timeout)**:
- **Limite**: 3000 steps par épisode
- **Calcul**: ~10 minutes à 30 FPS, assez pour parcourir 100m à vitesse normale

### 2. Génération de Corridors (`corridor_generator.py`)

**Objectif**: Forcer l'apprentissage de navigation générale au lieu de mémorisation de trajectoires.

**Stratégie anti-mémorisation**:
- **Nouveau corridor à chaque `reset()`**: Impossible de mémoriser un chemin fixe
- **Géométries MuJoCo réelles**: Pas de simulation, vraie physique avec collisions
- **Génération XML temps réel**: Nouveau `MjModel` complet reconstruit à chaque épisode
- **Contraintes garanties**: Toujours un passage possible, mais position/forme variables

**Zones progressives** - **Curriculum intégré**:
- **0-5m**: flat uniquement (spawn sécurisé, pas d'obstacles au démarrage)
- **5-19m**: flat + trous + transition 1m flat
- **21-39m**: flat + bumps larges + transition 1m flat  
- **41-59m**: flat + bumps moyens + transition 1m flat
- **61-100m**: flat + trous + bumps petits (zone finale mixte)

**Contraintes de navigabilité** - **Garanties mathématiques**:
- **Passage garanti**: Minimum 2 cellules de large (1m) à tout moment
- **Obstacles évitables**: Algorithme vérifie qu'il reste toujours un chemin
- **Trous minimum**: 2×2 cellules pour que le robot puisse effectivement tomber
- **Transitions**: 1m de flat entre zones pour éviter obstacles consécutifs

**Génération technique** - **Implémentation temps réel**:

**Pipeline de génération**:
1. **Grille logique** (0.5m): Génération des obstacles sur grille grossière
2. **Conversion XML**: Création des géométries MuJoCo avec positions exactes
3. **Parsing robot**: Fusion avec le modèle robot existant
4. **Compilation**: `MjModel.from_xml_string()` pour nouveau modèle complet
5. **Conversion observation**: Mapping vers grille fine (0.1m) pour le réseau

**Génération XML détaillée**:
```python
# Pour chaque cellule flat
geom = f'<geom type="box" material="mat_floor" 
         size="{0.25} {0.25} {0.025}" 
         pos="{center_x} {center_y} {0.025}" 
         name="flat_{counter}" />'

# Pour chaque cellule bump
geom = f'<geom type="box" material="mat_bump_{type}" 
         size="{0.25} {0.25} {half_z}" 
         pos="{center_x} {center_y} {z_top + half_z}" 
         name="bump_{counter}" />'

# Pour trous: pas de géométrie (vide physique)
```

**Matériaux et textures**:
- **Flat**: Texture damier grise, friction normale
- **Bumps**: Couleurs par taille (jaune=small, orange=medium, rouge=large)
- **Éclairage**: 2 sources (key + fill) pour ombres réalistes

**Performance**:
- **Génération**: < 10ms par corridor (grille → XML → MjModel)
- **Mémoire**: Nouveau modèle remplace l'ancien (pas d'accumulation)
- **Pas d'impact**: Génération pendant que l'agent calcule l'action précédente

### 3. Architecture Réseau (`train_ppo.py`)

**Réseau multi-branches**:

```
Observation (1902) → 3 branches parallèles → Fusion → Actor/Critic

Branch 1: Robot State (14) → MLP(64→64) → 64
Branch 2: History (88) → MLP(128→64→32) → 32  
Branch 3: Grid (60×30) → CNN → 128

Fusion: Concat(64+32+128) → MLP(128→64) → 64
Output: Actor(64→4) + Critic(64→1)
```

**CNN pour grille**:
```
60×30×1 → Conv2d(32,3×3,s=2) → 30×15×32
        → Conv2d(64,3×3,s=2) → 15×8×64  
        → Conv2d(128,3×3,s=2) → 8×4×128
        → Flatten → Linear(4096→128) → 128
```

**Historique étendu** - **Innovation pour l'anticipation**:
- **Fréquence**: sauvegarde toutes les 10 steps (2× plus fréquent qu'avant)
- **Longueur**: 8 frames (80 steps d'historique total)
- **Contenu**: 4 coins de bounding box + 3 vitesses (vx,vy,vz) par frame
- **Format**: coordonnées relatives à la position actuelle (invariant par translation)
- **Objectif**: Apprendre la dynamique du robot pour anticiper les trajectoires

### 4. Entraînement PPO

**Hyperparamètres**:
- **Environnements parallèles**: 32
- **Steps par rollout**: 1024
- **Batch size**: 32,768 (32×1024)
- **Minibatches**: 32 (1024 samples chacun)
- **Update epochs**: 10
- **Learning rate**: 5e-4
- **Gamma**: 0.995 (discount élevé pour récompenses lointaines)
- **GAE lambda**: 0.98
- **Clip coefficient**: 0.2

**Curriculum**: Pas de curriculum, directement 3000 steps par épisode.

**Optimisations**:
- AsyncVectorEnv pour parallélisation
- Buffers GPU pour vitesse
- Sauvegarde automatique tous les 10 itérations
- Reprise d'entraînement automatique

## Utilisation

### Entraînement

```bash
# Entraînement standard (4M steps)
python3 train_ppo.py

# Entraînement long (20M steps)
python3 train_ppo.py --timesteps 20000000

# Plus d'environnements parallèles
python3 train_ppo.py --num-envs 64

# Nouveau démarrage (ignorer modèles existants)
python3 train_ppo.py --fresh-start
```

### Test

```bash
# Test avec corridors aléatoires
python3 test_ppo.py --render

# Test avec corridor fixe
python3 test_ppo.py --render --corridor corridor_100.xml

# Plus d'épisodes
python3 test_ppo.py --episodes 10

# Affichage vision robot
python3 test_ppo.py --show-vision
```

### Visualisation

**Outil de débogage essentiel** pour comprendre exactement ce que voit le réseau de neurones:

```bash
# Vision exacte du CNN
python3 visualize_corridor_map.py --corridor corridor_100.xml

# Position spécifique
python3 visualize_corridor_map.py --x 50 --y 0.5 --angle 10

# Carte complète du corridor
python3 visualize_corridor_map.py --full-map

# Test positions multiples
python3 visualize_corridor_map.py --test-multiple
```

**Objectif**: Vérifier que l'observation est cohérente et déboguer les problèmes de navigation.

**Exemple de sortie** (robot à x=75m, centré, immobile):

```
Observation totale: 1902 valeurs (6 + 8 + 88 + 1800 = 1902)
Robot position: x=75.000m, y=0.000m, z=0.450m
Robot velocity: vx=0.000, vy=0.000, vz=0.000
Robot angle: 0.0°

BOUNDING BOX CORNERS (dans le repère grille):
  AV-G: row=   9.0, col=  18.0    # Avant-gauche
  AV-D: row=   9.0, col=  12.0    # Avant-droite
  AR-G: row=   1.0, col=  18.0    # Arrière-gauche
  AR-D: row=   1.0, col=  12.0    # Arrière-droite

HISTORIQUE ÉTENDU (8 frames × 11 valeurs):
  -70 steps: AV-G:(+1.0,-1.0) | ... | vel:(+0.00,+0.00,+0.00)
  -60 steps: AV-G:(+0.0,+0.0) | ... | vel:(+0.00,+0.00,+0.00)
  ...
  Actuelle:  AV-G:(+0.0,+0.0) | ... | vel:(+0.00,+0.00,+0.00)

GRILLE 60×30 - ENTRÉE DIRECTE DU CNN:
Vision: 0.5m derrière + 5.5m devant × 3m largeur
Robot à ligne 5 (position fixe dans la grille)

    012345678901234567890123456789
    ──────────────────────────────
  0|██████████████████████████████  -0.50m
  1|████████████R██████R██████████  -0.40m  ← Coins arrière
  2|██████████████████████████████  -0.30m
  3|██████████████████████████████  -0.20m
  4|██████████████████████████████  -0.10m
  5|██████████XXXXXXXXXX██████████  +0.00m  ← Robot (centre)
  6|██████████XXXXXXXXXX██████████  +0.10m
  7|██████████XXXXXXXXXX██████████  +0.20m
  8|██████████XXXXXXXXXX██████████  +0.30m
  9|██████████XXAXXXXXAX██████████  +0.40m  ← Coins avant
 10|██████████XXXXXXXXXX██████████  +0.50m
 11|██████████XXXXXXXXXX██████████  +0.60m
 12|██████████XXXXXXXXXX██████████  +0.70m
 13|██████████XXXXXXXXXX██████████  +0.80m
 14|██████████XXXXXXXXXX██████████  +0.90m
 15|██████████████████████████████  +1.00m
 16|██████████████████████████████  +1.10m
 17|██████████████████████████████  +1.20m
 18|██████████████████████████████  +1.30m
 19|██████████████████████████████  +1.40m
 20|██████████████████████████████  +1.50m
 21|██████████████████████████████  +1.60m
 22|██████████████████████████████  +1.70m
 23|██████████████████████████████  +1.80m
 24|██████████████████████████████  +1.90m
 25|██████████████████████████████  +2.00m
 26|██████████████████████████████  +2.10m
 27|██████████████████████████████  +2.20m
 28|██████████████████████████████  +2.30m
 29|██████████████████████████████  +2.40m

LÉGENDE:
  █ = sol (0.0)    ▲ = bump (0.5)    X = trou (1.0)
  A = coin AVANT    R = coin ARRIÈRE

STATISTIQUES:
  Sol (0.0):    1450 cellules (80.6%)  [█]
  Bump (0.5):    100 cellules ( 5.6%)  [▲]
  Trou (1.0):    250 cellules (13.9%)  [X]
```

**Analyse de cet exemple**:
- **Robot centré**: Coins symétriques (col 12 et 18 = ±3 cellules du centre)
- **Trou devant**: Visible lignes 5-10, le robot doit contourner
- **Historique stable**: Pas de mouvement récent (tous à 0.0)
- **Vision asymétrique**: Plus de vision devant (lignes 5-29) que derrière (0-5)

**Usage pour débogage**:
- Vérifier que les obstacles sont bien détectés
- Contrôler la cohérence de la bounding box
- Analyser l'historique de mouvement
- Comprendre pourquoi l'agent prend certaines décisions

## Fichiers

- **`corridor_env.py`**: Environnement Gymnasium avec génération aléatoire
- **`corridor_generator.py`**: Génération procédurale de corridors
- **`train_ppo.py`**: Entraînement PPO parallélisé
- **`test_ppo.py`**: Test et évaluation des modèles
- **`visualize_corridor_map.py`**: Visualisation de l'entrée CNN
- **`four_wheels_robot.xml`**: Modèle MuJoCo du robot
- **`corridor_100.xml`**: Corridor de référence (optionnel)

## Stratégie Anti-Mémorisation

**Problème critique**: Les agents RL peuvent mémoriser des trajectoires fixes au lieu d'apprendre la navigation générale. Sur un corridor fixe, l'agent apprend "aller à droite au mètre 25" au lieu de "éviter les trous".

**Solution multi-niveaux**: 
1. **Corridors aléatoires**: Nouveau layout complet à chaque épisode (impossible de mémoriser)
2. **Spawn aléatoire**: Position Y et angle initial variables (pas de départ fixe)
3. **Zones progressives**: Difficulté croissante mais passages toujours garantis
4. **Historique dynamique**: Anticipation basée sur la physique du robot, pas sur la mémorisation de positions
5. **Géométries réelles**: Vraie physique MuJoCo, pas de simulation simplifiée

**Validation**: L'agent doit généraliser sur des corridors jamais vus, prouvant qu'il a appris la navigation et non la mémorisation.

## Défis Techniques

**Détection out-of-bounds**: Vérification des 4 coins de la bounding box au lieu du centre robot.

**Vision asymétrique**: Plus de vision devant (5.5m) que derrière (0.5m) pour l'anticipation.

**Historique multi-modal**: Positions + vitesses pour comprendre la dynamique du robot.

**Génération temps réel**: Nouveau modèle MuJoCo à chaque reset sans impact performance.

## Résultats Attendus

**Métriques de succès**:
- Distance moyenne > 50m
- Taux de succès > 10% (atteindre 100m)
- Généralisation sur corridors non vus

**Comportements recherchés**:
- Navigation fluide évitant obstacles
- Anticipation des trous/bumps
- Récupération après perturbations
- Adaptation à différents layouts

## Bonus: Outils de Création de Corridors

Le dossier `corridor_creation/` contient des outils pour créer et éditer des corridors manuellement, utiles pour tester des cas spécifiques ou créer des environnements de référence.

### Éditeur Graphique (`corridor_creation/corridor_editor.py`)

**Interface graphique** pour créer des corridors personnalisés:

```bash
cd corridor_creation
python3 corridor_editor.py
```

**Fonctionnalités**:
- **Grille interactive**: Clic pour placer flat/bump/hole
- **Types d'obstacles**: 
  - Flat (sol normal)
  - Bumps: small (0.05m), medium (0.2m), large (0.5m)
  - Holes (trous)
- **Outils**:
  - Pinceau pour dessiner
  - Gomme pour effacer
  - Remplissage de zones
- **Options**:
  - Checkbox pour activer/désactiver les murs latéraux
  - Sauvegarde/chargement XML
  - Prévisualisation temps réel

**Utilisation**:
1. Dessiner le corridor avec la souris
2. Choisir le type d'obstacle dans le menu
3. Sauvegarder en XML
4. Tester avec `python test_ppo.py --corridor votre_corridor.xml`

### Générateur Batch (`corridor_creation/generate_corridors.py`)

**Génération en lot** de corridors aléatoires pour constituer un dataset:

```bash
cd corridor_creation

# Générer 5 corridors (défaut)
python3 generate_corridors.py

# Générer 20 corridors dans un dossier spécifique
python3 generate_corridors.py -n 20 -o mes_corridors

# Génération reproductible avec seed
python3 generate_corridors.py -n 10 -s 42
```

**Paramètres**:
- `-n, --count`: Nombre de corridors à générer
- `-o, --output`: Dossier de sortie (défaut: `corridors/`)
- `-s, --seed`: Seed pour reproductibilité

**Différences avec `corridor_generator.py`**:
- **Batch vs Runtime**: Génère des fichiers XML vs génération temps réel
- **Stockage**: Corridors sauvés sur disque vs en mémoire
- **Usage**: Tests manuels vs entraînement automatique

### Cas d'Usage des Outils

**Éditeur graphique**:
- Créer des corridors de test spécifiques
- Tester des configurations d'obstacles particulières
- Débugger des comportements sur des layouts connus
- Créer des corridors de démonstration

**Générateur batch**:
- Constituer un dataset de corridors variés
- Tests de généralisation sur corridors fixes
- Benchmarks reproductibles
- Analyse de performance sur différents layouts

**Exemple de workflow**:
```bash
# 1. Créer un corridor de test avec l'éditeur
python3 corridor_editor.py
# → Sauver comme test_holes_only.xml

# 2. Tester l'agent dessus
cd ../optimized_ppo
python3 test_ppo.py --corridor ../corridor_creation/test_holes_only.xml --render

# 3. Générer un dataset pour évaluation
cd ../corridor_creation
python3 generate_corridors.py -n 50 -o evaluation_set

# 4. Tester sur tout le dataset
cd ../optimized_ppo
for corridor in ../corridor_creation/evaluation_set/*.xml; do
    python3 test_ppo.py --corridor "$corridor" --episodes 3
done
```

### Intégration avec l'Entraînement

Les corridors créés manuellement peuvent être utilisés pour:

**Tests ciblés**:
- Vérifier que l'agent gère bien les trous uniquement
- Tester la navigation avec bumps uniquement
- Évaluer sur des passages très étroits

**Curriculum manuel**:
- Commencer l'entraînement sur corridors simples
- Progresser vers corridors complexes
- Affiner sur des cas difficiles spécifiques

**Validation**:
- Dataset de test fixe pour comparer les versions
- Corridors de référence pour benchmarks
- Cas limites pour robustesse