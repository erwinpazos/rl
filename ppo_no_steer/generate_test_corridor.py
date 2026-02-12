"""Générer un corridor de test avec le corridor_generator_similar.py"""
from corridor_generator_similar import CorridorGenerator

# Créer générateur
generator = CorridorGenerator()

# Générer un corridor avec seed fixe pour reproductibilité
seed = 12345
length = 30.0  # Court pour analyse
width = 3.0
bump_ratio = 0.3  # 30% de bumps

print(f"Génération corridor:")
print(f"  Length: {length}m")
print(f"  Width: {width}m")
print(f"  Seed: {seed}")
print(f"  Bump ratio: {bump_ratio*100:.0f}%")

# Générer le XML
xml_str = generator.generate_corridor_xml(
    length=length,
    width=width,
    seed=seed,
    name="test_corridor",
    obstacle_type="holes",
    bump_ratio=bump_ratio
)

# Sauvegarder
output_file = "corridor_test_generated.xml"
with open(output_file, 'w') as f:
    f.write(xml_str)

print(f"\n✅ Corridor sauvegardé: {output_file}")

# Afficher les positions des trous et bumps
holes = generator.generate_hole_pattern(length, seed)
print(f"\n🕳️  TROUS ({len(holes)}):")
for i, (x, y) in enumerate(holes[:10]):  # Premiers 10
    print(f"  Hole {i+1}: x={x:.2f}m, y={y:.2f}m")
if len(holes) > 10:
    print(f"  ... et {len(holes)-10} autres")

# Générer les bumps
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)

bumps = []
if bump_ratio > 0.0 and len(holes) > 1:
    holes_sorted = sorted(holes, key=lambda h: h[0])
    num_spaces = len(holes_sorted) - 1
    num_bumps = int(num_spaces * bump_ratio)
    
    bump_indices = random.sample(range(num_spaces), min(num_bumps, num_spaces))
    
    for bump_idx in bump_indices:
        if bump_idx < num_spaces:
            hole1_x, hole1_y = holes_sorted[bump_idx]
            hole2_x, hole2_y = holes_sorted[bump_idx + 1]
            middle_x = (hole1_x + hole2_x) / 2.0
            
            bump_y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
            bump_y = random.choice(bump_y_positions)
            
            bumps.append((middle_x, bump_y))

print(f"\n💥 BUMPS ({len(bumps)}):")
for i, (x, y) in enumerate(bumps):
    print(f"  Bump {i+1}: x={x:.2f}m, y={y:.2f}m")

print(f"\n📊 Résumé:")
print(f"  Trous: {len(holes)}")
print(f"  Bumps: {len(bumps)}")
print(f"  Ratio bumps/espaces: {len(bumps)}/{num_spaces if len(holes) > 1 else 0} = {bump_ratio*100:.0f}%")
