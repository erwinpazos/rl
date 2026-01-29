#!/usr/bin/env python3
"""
Analyseur de structure du corridor pour comprendre le pattern et générer des corridors similaires.
"""
import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_corridor_structure():
    """Analyse la structure du corridor existant."""
    
    # Lire le fichier XML
    with open('ppo_small_robot_steer/corridor_3x100_no_full_obstacles.xml', 'r') as f:
        content = f.read()
    
    print("=" * 80)
    print("ANALYSE DE LA STRUCTURE DU CORRIDOR")
    print("=" * 80)
    
    # 1. ANALYSE DES CELLULES DE SOL (floor_flat)
    floor_pattern = r'floor_flat_\d+.*?size="([0-9.]+) ([0-9.]+) ([0-9.]+)".*?pos="([0-9.]+) ([0-9.-]+) ([0-9.]+)"'
    floors = re.findall(floor_pattern, content)
    
    print(f"\n📦 CELLULES DE SOL (floor_flat):")
    print(f"   Nombre total: {len(floors)}")
    if floors:
        size_x, size_y, size_z = floors[0][:3]
        print(f"   Taille standard: {size_x} × {size_y} × {size_z} m")
        print(f"   Hauteur Z: {floors[0][5]} m")
    
    # 2. ANALYSE DES BUMPS
    bump_pattern = r'floor_bump_\d+.*?pos="([0-9.]+) ([0-9.-]+) ([0-9.]+)".*?size="([0-9.]+) ([0-9.]+) ([0-9.]+)"'
    bumps = re.findall(bump_pattern, content)
    
    print(f"\n🔺 BUMPS (obstacles):")
    print(f"   Nombre total: {len(bumps)}")
    if bumps:
        size_x, size_y, size_z = bumps[0][3:6]
        print(f"   Taille standard: {size_x} × {size_y} × {size_z} m")
        print(f"   Hauteur Z: {bumps[0][2]} m")
    
    # Analyser positions des bumps
    bump_positions = [(float(x), float(y)) for x, y, z, sx, sy, sz in bumps]
    bump_x = [pos[0] for pos in bump_positions]
    bump_y = [pos[1] for pos in bump_positions]
    
    print(f"   Position X: {min(bump_x):.2f}m à {max(bump_x):.2f}m")
    print(f"   Position Y: {min(bump_y):.2f}m à {max(bump_y):.2f}m")
    
    # Analyser espacement des bumps
    bump_x_sorted = sorted(set(bump_x))
    if len(bump_x_sorted) > 1:
        x_spacings = [bump_x_sorted[i+1] - bump_x_sorted[i] for i in range(len(bump_x_sorted)-1)]
        print(f"   Espacement X: {min(x_spacings):.2f}m à {max(x_spacings):.2f}m (moy: {np.mean(x_spacings):.2f}m)")
    
    # 3. ANALYSE DES TROUS
    hole_pattern = r'floor_hole_tile_\d+.*?size="([0-9.]+) ([0-9.]+) ([0-9.]+)".*?pos="([0-9.]+) ([0-9.-]+) ([0-9.]+)"'
    holes = re.findall(hole_pattern, content)
    
    print(f"\n🕳️  TROUS (holes):")
    print(f"   Nombre total: {len(holes)}")
    if holes:
        size_x, size_y, size_z = holes[0][:3]
        print(f"   Taille standard: {size_x} × {size_y} × {size_z} m")
        print(f"   Hauteur Z: {holes[0][5]} m")
    
    # Analyser positions des trous
    hole_positions = [(float(x), float(y)) for sx, sy, sz, x, y, z in holes]
    hole_x = [pos[0] for pos in hole_positions]
    hole_y = [pos[1] for pos in hole_positions]
    
    print(f"   Position X: {min(hole_x):.2f}m à {max(hole_x):.2f}m")
    print(f"   Position Y: {min(hole_y):.2f}m à {max(hole_y):.2f}m")
    
    # 4. ANALYSE DES MURS
    wall_pattern = r'wall_(left|right).*?size="([0-9.]+) ([0-9.]+) ([0-9.]+)".*?pos="([0-9.]+) ([0-9.-]+) ([0-9.]+)"'
    walls = re.findall(wall_pattern, content)
    
    print(f"\n🧱 MURS:")
    print(f"   Nombre: {len(walls)}")
    for side, sx, sy, sz, x, y, z in walls:
        print(f"   Mur {side}: taille {sx}×{sy}×{sz}m, pos ({x}, {y}, {z})")
    
    # 5. DIMENSIONS GLOBALES
    print(f"\n📏 DIMENSIONS GLOBALES:")
    
    # Largeur du corridor (basée sur les murs)
    if len(walls) >= 2:
        wall_left_y = float([w for w in walls if w[0] == 'left'][0][5])
        wall_right_y = float([w for w in walls if w[0] == 'right'][0][5])
        corridor_width = wall_right_y - wall_left_y
        print(f"   Largeur corridor: {corridor_width:.3f}m")
        print(f"   Mur gauche Y: {wall_left_y:.3f}m")
        print(f"   Mur droit Y: {wall_right_y:.3f}m")
    
    # Longueur du corridor (basée sur les positions max)
    all_x = bump_x + hole_x
    if all_x:
        corridor_length = max(all_x)
        print(f"   Longueur corridor: {corridor_length:.1f}m")
    
    # 6. PATTERN DÉTAILLÉ
    print(f"\n🎯 PATTERN DÉTAILLÉ:")
    
    # Grille de cellules (basée sur les floor_flat)
    if floors:
        cell_size = float(floors[0][0]) * 2  # size="0.250" → cellule de 0.5m
        print(f"   Taille cellule: {cell_size}m × {cell_size}m")
        
        # Calculer nombre de cellules
        if walls:
            n_cells_x = int(corridor_length / cell_size)
            n_cells_y = int(corridor_width / cell_size)
            print(f"   Grille: {n_cells_x} × {n_cells_y} cellules")
    
    # 7. PATTERN DES OBSTACLES
    print(f"\n🔄 PATTERN DES OBSTACLES:")
    
    # Analyser le pattern Y des bumps
    unique_y_bumps = sorted(set(bump_y))
    print(f"   Positions Y des bumps: {unique_y_bumps}")
    
    # Analyser le pattern Y des trous
    unique_y_holes = sorted(set(hole_y))
    print(f"   Positions Y des trous: {unique_y_holes}")
    
    # Séquence des bumps
    bumps_sorted = sorted(bump_positions, key=lambda p: p[0])
    print(f"\n   Séquence des bumps (X, Y):")
    for i, (x, y) in enumerate(bumps_sorted[:10]):  # Premiers 10
        print(f"     Bump {i+1}: ({x:.2f}, {y:.2f})")
    if len(bumps_sorted) > 10:
        print(f"     ... et {len(bumps_sorted)-10} autres")
    
    # Séquence des trous
    holes_sorted = sorted(hole_positions, key=lambda p: p[0])
    print(f"\n   Séquence des trous (X, Y):")
    for i, (x, y) in enumerate(holes_sorted):
        print(f"     Trou {i+1}: ({x:.2f}, {y:.2f})")
    
    return {
        'floors': floors,
        'bumps': bumps,
        'holes': holes,
        'walls': walls,
        'bump_positions': bump_positions,
        'hole_positions': hole_positions,
        'corridor_width': corridor_width if 'corridor_width' in locals() else 3.0,
        'corridor_length': corridor_length if 'corridor_length' in locals() else 100.0,
        'cell_size': cell_size if 'cell_size' in locals() else 0.5
    }


def generate_similar_corridor(length=100.0, width=3.0, cell_size=0.5):
    """Génère un corridor avec le même style que l'original."""
    
    print(f"\n" + "=" * 80)
    print(f"GÉNÉRATION D'UN CORRIDOR SIMILAIRE")
    print(f"=" * 80)
    print(f"Longueur: {length}m, Largeur: {width}m, Cellule: {cell_size}m")
    
    # Paramètres basés sur l'analyse
    bump_size = (0.25, 0.25, 0.25)  # Taille des bumps
    bump_height = 0.275  # Hauteur des bumps
    hole_size = (0.25, 0.5, 0.025)  # Taille des trous
    floor_size = (0.25, 0.25, 0.025)  # Taille des cellules de sol
    
    # Positions Y possibles (basées sur l'analyse)
    y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]  # 6 positions Y
    hole_y_positions = [-0.5, 0.5, 1.0]  # 3 positions Y pour les trous
    
    # Générer pattern des bumps (similaire à l'original)
    bumps = []
    bump_id = 0
    
    # Pattern observé: bumps tous les ~2m avec variation Y
    for x in np.arange(2.25, length, 2.0):  # Commence à 2.25m, tous les 2m
        # Choisir position Y selon un pattern cyclique
        y_idx = (bump_id // 2) % len(y_positions)  # Change tous les 2 bumps
        if bump_id % 11 == 10:  # Parfois sauter une position (gap observé)
            y_idx = (y_idx + 2) % len(y_positions)
        
        y = y_positions[y_idx]
        bumps.append((x, y, bump_height, bump_size))
        bump_id += 1
    
    # Générer pattern des trous (plus espacés)
    holes = []
    hole_id = 0
    
    # Pattern observé: trous tous les ~4-8m
    for x in np.arange(6.25, length, 4.0):  # Commence à 6.25m, tous les 4m
        if hole_id % 4 == 3:  # Parfois gap plus grand
            x += 8.0
        
        # Choisir position Y cyclique
        y_idx = hole_id % len(hole_y_positions)
        y = hole_y_positions[y_idx]
        
        if x < length:
            holes.append((x, y, 0.025, hole_size))
            hole_id += 1
    
    print(f"Généré: {len(bumps)} bumps, {len(holes)} trous")
    
    return {
        'bumps': bumps,
        'holes': holes,
        'length': length,
        'width': width,
        'cell_size': cell_size
    }


def visualize_corridor(data):
    """Visualise le corridor généré."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Vue d'ensemble
    ax1.set_title("Vue d'ensemble du corridor", fontsize=14)
    ax1.set_xlim(0, data['length'])
    ax1.set_ylim(-1.6, 1.6)
    ax1.set_xlabel("X (longueur) [m]")
    ax1.set_ylabel("Y (largeur) [m]")
    ax1.grid(True, alpha=0.3)
    
    # Murs
    ax1.axhline(y=-1.5, color='gray', linewidth=3, label='Murs')
    ax1.axhline(y=1.5, color='gray', linewidth=3)
    
    # Bumps
    for x, y, z, size in data['bumps']:
        rect = plt.Rectangle((x-size[0], y-size[1]), size[0]*2, size[1]*2, 
                           color='blue', alpha=0.7)
        ax1.add_patch(rect)
    
    # Trous
    for x, y, z, size in data['holes']:
        rect = plt.Rectangle((x-size[0], y-size[1]), size[0]*2, size[1]*2, 
                           color='red', alpha=0.7)
        ax1.add_patch(rect)
    
    ax1.legend(['Murs', 'Bumps (obstacles)', 'Trous'])
    
    # Vue détaillée (premiers 20m)
    ax2.set_title("Vue détaillée (0-20m)", fontsize=14)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-1.6, 1.6)
    ax2.set_xlabel("X (longueur) [m]")
    ax2.set_ylabel("Y (largeur) [m]")
    ax2.grid(True, alpha=0.3)
    
    # Murs
    ax2.axhline(y=-1.5, color='gray', linewidth=3)
    ax2.axhline(y=1.5, color='gray', linewidth=3)
    
    # Bumps (premiers 20m)
    for x, y, z, size in data['bumps']:
        if x <= 20:
            rect = plt.Rectangle((x-size[0], y-size[1]), size[0]*2, size[1]*2, 
                               color='blue', alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(x, y, f'{x:.1f}', ha='center', va='center', fontsize=8)
    
    # Trous (premiers 20m)
    for x, y, z, size in data['holes']:
        if x <= 20:
            rect = plt.Rectangle((x-size[0], y-size[1]), size[0]*2, size[1]*2, 
                               color='red', alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(x, y, f'{x:.1f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('corridor_structure_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique sauvegardé: corridor_structure_analysis.png")


if __name__ == "__main__":
    # Analyser le corridor existant
    analysis = analyze_corridor_structure()
    
    # Générer un corridor similaire
    generated = generate_similar_corridor(
        length=100.0,
        width=analysis['corridor_width'],
        cell_size=analysis['cell_size']
    )
    
    # Visualiser
    visualize_corridor(generated)
    
    print(f"\n" + "=" * 80)
    print("RÉSUMÉ DE LA STRUCTURE:")
    print("=" * 80)
    print(f"✅ Cellules de sol: {analysis['cell_size']}m × {analysis['cell_size']}m")
    print(f"✅ Bumps: 0.25m × 0.25m × 0.25m, hauteur 0.275m")
    print(f"✅ Trous: 0.25m × 0.5m × 0.025m")
    print(f"✅ Espacement bumps: ~2m")
    print(f"✅ Espacement trous: ~4-8m")
    print(f"✅ Positions Y bumps: 6 niveaux (-1.25 à +1.25)")
    print(f"✅ Positions Y trous: 3 niveaux (-0.5, +0.5, +1.0)")
    print(f"✅ Pattern cyclique avec gaps occasionnels")