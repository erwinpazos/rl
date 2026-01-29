#!/usr/bin/env python3
"""Test du générateur de trous."""

from corridor_generator_similar import CorridorGenerator

gen = CorridorGenerator()

# Test avec différents seeds
for seed in [42, 123, 456]:
    print(f"\n=== SEED {seed} ===")
    holes = gen.generate_hole_pattern(100.0, seed)
    print(f"Nombre de trous générés: {len(holes)}")
    
    for i, (x, y) in enumerate(holes):
        print(f"Trou {i+1}: ({x:.2f}, {y:.2f})")
    
    # Test aussi les bumps
    bumps = gen.generate_bump_pattern(100.0, seed)
    print(f"Nombre de bumps générés: {len(bumps)}")

print("\n=== TEST GÉNÉRATION COMPLÈTE ===")
gen.save_corridor("test_holes.xml", length=100.0, width=3.0, seed=42)