#!/usr/bin/env python3
"""Debug de la génération des trous."""

from corridor_generator_similar import CorridorGenerator

gen = CorridorGenerator()

# Test avec seed 4318
seed = 4318
print(f"=== DEBUG SEED {seed} ===")

holes = gen.generate_hole_pattern(100.0, seed)
print(f"Trous générés: {len(holes)}")
for i, (x, y) in enumerate(holes):
    print(f"  Trou {i+1}: ({x:.2f}, {y:.2f})")

floor_tiles = gen.generate_floor_tiles(100.0, 3.0, holes)
print(f"\nTuiles de sol générées: {len(floor_tiles)}")

# Vérifier si les tuiles sont bien supprimées aux positions des trous
print(f"\n=== VÉRIFICATION SUPPRESSION TUILES ===")
for i, (hole_x, hole_y) in enumerate(holes[:5]):  # Vérifier les 5 premiers trous
    print(f"\nTrou {i+1} à ({hole_x:.2f}, {hole_y:.2f}):")
    
    # Positions des tuiles qui devraient être supprimées
    tile_y1 = hole_y - 0.25
    tile_y2 = hole_y + 0.25
    
    print(f"  Devrait supprimer tuiles à ({hole_x:.2f}, {tile_y1:.2f}) et ({hole_x:.2f}, {tile_y2:.2f})")
    
    # Vérifier si ces tuiles existent dans floor_tiles
    tile1_exists = (round(hole_x, 2), round(tile_y1, 2)) in [(round(x, 2), round(y, 2)) for x, y in floor_tiles]
    tile2_exists = (round(hole_x, 2), round(tile_y2, 2)) in [(round(x, 2), round(y, 2)) for x, y in floor_tiles]
    
    print(f"  Tuile 1 ({hole_x:.2f}, {tile_y1:.2f}) existe: {tile1_exists} {'❌ ERREUR!' if tile1_exists else '✅ OK'}")
    print(f"  Tuile 2 ({hole_x:.2f}, {tile_y2:.2f}) existe: {tile2_exists} {'❌ ERREUR!' if tile2_exists else '✅ OK'}")