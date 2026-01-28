# Système de Récompenses Simplifié

## Philosophie

Le système de récompenses a été **drastiquement simplifié** pour éviter le reward hacking et encourager un apprentissage plus naturel. L'agent doit apprendre par lui-même les stratégies d'évitement sans être guidé par des récompenses artificielles.

## Récompenses et Terminaisons

### ✅ **Succès** (+100, terminé)
```python
if x >= 100.0:  # Atteindre la fin du corridor
    return 100.0, True, {'reason': 'success'}
```
- **Condition** : Robot atteint x ≥ 100m
- **Récompense** : +100 (grosse récompense finale)
- **Terminaison** : Oui

### ❌ **Échecs** (-10, terminé)

#### 1. Tombé dans un trou
```python
if z < 0.15:  # Hauteur critique
    return -10.0, True, {'reason': 'fell'}
```
- **Condition** : Robot tombe sous 0.15m de hauteur
- **Cause** : Trou dans le corridor OU sortie du corridor (avant/arrière)
- **Récompense** : -10
- **Terminaison** : Oui

#### 2. Robot retourné
```python
quat = self.data.qpos[3:7]
up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
if up_z < 0:  # Robot à l'envers
    return -10.0, True, {'reason': 'flipped'}
```
- **Condition** : Robot complètement retourné (up_z < 0)
- **Cause** : Collision violente, mauvaise manœuvre
- **Récompense** : -10
- **Terminaison** : Oui

#### 3. Collision avec bump
```python
if self._is_colliding_with_bump():
    return -10.0, True, {'reason': 'collision'}
```
- **Condition** : Contact physique avec un pilier/bump
- **Détection** : Via les contacts MuJoCo entre géométries robot et bump
- **Récompense** : -10
- **Terminaison** : Oui

### 🏃 **Progression continue** (variable, non terminé)
```python
delta_x = x - self.prev_x
reward = delta_x * 10.0  # Récompense pour avancer
```
- **Condition** : À chaque step
- **Calcul** : 10 × distance parcourue en X
- **Exemples** :
  - Avancer de 0.1m → +1.0
  - Reculer de 0.05m → -0.5
  - Rester immobile → 0.0
- **Terminaison** : Non

### ⏱️ **Timeout** (0, tronqué)
```python
if step_count >= max_steps:  # Défaut: 1000 steps
    # Pas de récompense spéciale, juste truncated=True
```
- **Condition** : Limite de temps atteinte
- **Récompense** : 0 (pas de pénalité)
- **Terminaison** : Tronqué (pas terminé)

## Avantages de cette Approche

### 🎯 **Simplicité**
- **4 cas seulement** : succès, échecs (3 types), progression
- **Pas de reward hacking** : Impossible de tricher le système
- **Facile à déboguer** : Comportements prévisibles

### 🧠 **Apprentissage naturel**
- **Pas de guidage artificiel** : L'agent découvre les stratégies par lui-même
- **Exploration encouragée** : Pas de pénalités préventives qui limitent l'exploration
- **Généralisation** : Stratégies apprises fonctionnent sur tous les corridors

### ⚡ **Performance**
- **Calculs rapides** : Pas de rayon laser, pas de calculs complexes
- **Moins de bugs** : Système simple = moins d'erreurs
- **Stable** : Récompenses cohérentes et prévisibles

## Comportements Attendus

### 🎯 **Objectifs clairs**
1. **Avancer** : Récompense continue pour progression en X
2. **Éviter les dangers** : Apprendre à ne pas tomber/se retourner/percuter
3. **Atteindre la fin** : Grosse récompense finale pour motivation

### 🚫 **Ce qui est découragé**
- **Reculer** : Récompense négative
- **Rester immobile** : Pas de récompense
- **Prendre des risques** : Pénalités pour échecs

### 🤖 **Stratégies émergentes attendues**
- **Navigation prudente** : Évitement naturel des obstacles
- **Vitesse optimale** : Équilibre entre vitesse et sécurité
- **Récupération** : Apprendre à se remettre d'erreurs mineures

## Comparaison Ancien vs Nouveau

| Aspect | Ancien Système | Nouveau Système |
|--------|----------------|-----------------|
| **Complexité** | 7+ composants | 4 composants |
| **Récompenses** | Continues complexes | Simples et claires |
| **Guidage** | Fort (laser, rotation) | Minimal (progression) |
| **Risque de hack** | Élevé | Très faible |
| **Performance** | Calculs lourds | Calculs légers |
| **Débug** | Difficile | Facile |
| **Généralisation** | Risque de sur-apprentissage | Apprentissage robuste |

## Métriques de Succès

### 📊 **Métriques principales**
- **Distance moyenne** : Progression typique avant échec
- **Taux de succès** : % d'épisodes atteignant 100m
- **Raisons d'échec** : Distribution des types d'échecs
- **Temps de survie** : Durée moyenne des épisodes

### 🎯 **Objectifs d'entraînement**
- **Distance > 50m** : Agent apprend la navigation de base
- **Succès > 10%** : Agent maîtrise les corridors complexes
- **Échecs équilibrés** : Pas de domination d'un type d'échec
- **Progression stable** : Amélioration continue des métriques

Ce système simplifié devrait permettre un apprentissage plus robuste et naturel, sans les biais introduits par des récompenses artificielles complexes.