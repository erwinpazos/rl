# Changements pour bump_ratio

## Résumé
Au lieu de `obstacle_type` ("holes", "bumps", "both"), on utilise maintenant `bump_ratio` (0.0 à 1.0) :
- Phase 1 : bump_ratio=0.0 (100% holes, 0% bumps)
- Phase 2 : bump_ratio=0.3 (100% holes + 30% bumps)
- Phase 3 : bump_ratio=0.6 (100% holes + 60% bumps)

## Fichiers modifiés

### ✅ corridor_generator_similar.py
- `generate_corridor_xml()` : Ajout paramètre `bump_ratio`
- `save_corridor()` : Ajout paramètre `bump_ratio`

### ✅ corridor_env.py  
- `_build_model_from_new_generator()` : Utilise `bump_ratio` au lieu de `obstacle_type`
- `update_curriculum_params()` : Accepte `bump_ratio`

### ✅ config.yaml
- Remplacé `obstacle_type_schedule` par `bump_ratio_schedule`

### ✅ train_ppo.py (partiellement)
- `get_curriculum_state()` : Retourne `bump_ratio` au lieu de `obstacle_type`
- Lecture de `bump_ratio_schedule` au lieu de `obstacle_type_schedule`

### ❌ train_ppo.py (à finir)
- `make_env()` : Remplacer `obstacle_type` par `bump_ratio`
- `update_curriculum()` : Passer `bump_ratio` aux envs
- `debug_render_episode()` : Utiliser `bump_ratio`
- Tous les logs/prints : Afficher bump_ratio au lieu de obstacle_type

## Statut
Modifications partielles - le code ne compile pas encore.
Il faut finir de remplacer toutes les références à `obstacle_type` par `bump_ratio` dans train_ppo.py.
