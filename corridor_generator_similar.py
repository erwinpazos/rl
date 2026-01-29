#!/usr/bin/env python3
"""
Générateur de corridors basé sur l'analyse de la structure existante.
Reproduit le même style avec des variations contrôlées.
"""
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple
import random

class CorridorGenerator:
    """Générateur de corridors basé sur le pattern analysé."""
    
    def __init__(self):
        # Paramètres basés sur l'analyse
        self.cell_size = 0.5  # Taille d'une cellule de grille
        self.floor_tile_size = (0.25, 0.25, 0.025)  # Taille des tuiles de sol
        self.bump_size = (0.25, 0.25, 0.25)  # Taille des bumps
        self.hole_size = (0.25, 0.5, 0.025)  # Taille des trous
        
        # Positions Y possibles (6 niveaux pour bumps, 3 pour trous)
        self.bump_y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
        self.hole_y_positions = [-0.5, 0.5, 1.0]
        
        # Pattern observé
        self.bump_spacing_mean = 2.0  # Espacement moyen des bumps
        self.bump_spacing_std = 0.5   # Variation de l'espacement
        self.hole_spacing_mean = 4.0  # Espacement moyen des trous
        
    def generate_bump_pattern(self, length: float, seed: int = None) -> List[Tuple[float, float]]:
        """Génère le pattern des bumps basé sur l'analyse."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        bumps = []
        x = 2.25  # Position de départ (observée)
        y_cycle_pos = 0  # Position dans le cycle Y
        
        # Pattern cyclique observé: -1.25 → -0.75 → -0.25 → 0.25 → 0.75 → 1.25 → 0.75 → 0.25 → ...
        y_pattern = [
            -1.25, -0.75, -0.25, 0.25, 0.75, 1.25,  # Montée
            0.75, 0.25, -0.25, -0.75, -1.25,        # Descente
        ]
        
        while x < length - 1.0:
            # Position Y selon le pattern cyclique
            y = y_pattern[y_cycle_pos % len(y_pattern)]
            bumps.append((x, y))
            
            # Espacement avec variation (observé: 1-4m, moyenne 2m)
            if random.random() < 0.1:  # 10% de chance de gap plus grand
                spacing = random.uniform(3.0, 4.0)
            else:
                spacing = random.uniform(1.0, 2.5)
            
            x += spacing
            y_cycle_pos += 1
        
        return bumps
    
    def generate_hole_pattern(self, length: float, seed: int = None) -> List[Tuple[float, float]]:
        """Génère le pattern des trous basé sur l'analyse."""
        if seed is not None:
            random.seed(seed + 1000)  # Seed différent pour les trous
        
        holes = []
        x = 6.25  # Position de départ (observée)
        y_cycle_pos = 0
        
        # Pattern cyclique observé: -0.5 → 0.5 → 1.0 → -0.5 → ...
        y_pattern = [-0.5, 0.5, 1.0]
        
        while x < length - 2.0:
            # Position Y selon le pattern cyclique
            y = y_pattern[y_cycle_pos % len(y_pattern)]
            holes.append((x, y))
            
            # Espacement observé: 4m avec gaps occasionnels
            if random.random() < 0.15:  # 15% de chance de gap plus grand
                spacing = random.uniform(8.0, 12.0)
            else:
                spacing = 4.0
            
            x += spacing
            y_cycle_pos += 1
        
        return holes
    
    def generate_floor_tiles(self, length: float, width: float) -> List[Tuple[float, float]]:
        """Génère toutes les tuiles de sol."""
        tiles = []
        
        # Grille basée sur cell_size = 0.5m
        # Positions centrées sur 0.25, 0.75, 1.25, etc.
        x_positions = np.arange(0.25, length, 0.5)
        y_positions = np.arange(-width/2 + 0.25, width/2, 0.5)
        
        for x in x_positions:
            for y in y_positions:
                tiles.append((x, y))
        
        return tiles
    
    def generate_corridor_xml(self, length: float = 100.0, width: float = 3.0, 
                            seed: int = None, name: str = "generated_corridor") -> str:
        """Génère le XML complet du corridor."""
        
        # Générer les patterns
        bumps = self.generate_bump_pattern(length, seed)
        holes = self.generate_hole_pattern(length, seed)
        floor_tiles = self.generate_floor_tiles(length, width)
        
        # Créer l'élément racine
        root = ET.Element('mujoco')
        root.set('model', name)
        
        # Compiler
        compiler = ET.SubElement(root, 'compiler')
        compiler.set('angle', 'degree')
        compiler.set('autolimits', 'true')
        
        # Options
        option = ET.SubElement(root, 'option')
        option.set('timestep', '0.005')
        option.set('gravity', '0 0 -9.81')
        
        # Size
        size = ET.SubElement(root, 'size')
        size.set('njmax', '4000')
        size.set('nconmax', '1000')
        
        # Assets (matériaux)
        asset = ET.SubElement(root, 'asset')
        
        materials = [
            ('mat_floor', '0.85 0.85 0.85 1'),
            ('mat_bump', '0.25 0.6 0.9 1'),
            ('mat_wall', '0.4 0.4 0.4 1'),
            ('mat_hole', '0.8 0.4 0.2 1')
        ]
        
        for name, rgba in materials:
            mat = ET.SubElement(asset, 'material')
            mat.set('name', name)
            mat.set('rgba', rgba)
        
        # Worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Murs (basés sur l'analyse)
        wall_thickness = 0.025
        wall_height = 2.5
        wall_length = length / 2
        wall_center_x = length / 2
        
        # Mur gauche
        wall_left = ET.SubElement(worldbody, 'geom')
        wall_left.set('name', 'wall_left')
        wall_left.set('type', 'box')
        wall_left.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_left.set('pos', f'{wall_center_x:.3f} {-width/2 - wall_thickness:.3f} {wall_height:.3f}')
        wall_left.set('material', 'mat_wall')
        
        # Mur droit
        wall_right = ET.SubElement(worldbody, 'geom')
        wall_right.set('name', 'wall_right')
        wall_right.set('type', 'box')
        wall_right.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_right.set('pos', f'{wall_center_x:.3f} {width/2 + wall_thickness:.3f} {wall_height:.3f}')
        wall_right.set('material', 'mat_wall')
        
        # Tuiles de sol
        for i, (x, y) in enumerate(floor_tiles):
            tile = ET.SubElement(worldbody, 'geom')
            tile.set('type', 'box')
            tile.set('material', 'mat_floor')
            tile.set('name', f'floor_flat_{i}')
            tile.set('size', f'{self.floor_tile_size[0]:.3f} {self.floor_tile_size[1]:.3f} {self.floor_tile_size[2]:.3f}')
            tile.set('pos', f'{x:.3f} {y:.3f} {self.floor_tile_size[2]:.3f}')
        
        # Trous (avant les bumps pour l'ordre)
        for i, (x, y) in enumerate(holes):
            hole = ET.SubElement(worldbody, 'geom')
            hole.set('name', f'floor_hole_tile_{i}')
            hole.set('type', 'box')
            hole.set('size', f'{self.hole_size[0]:.3f} {self.hole_size[1]:.3f} {self.hole_size[2]:.3f}')
            hole.set('pos', f'{x:.3f} {y:.3f} {self.hole_size[2]:.3f}')
            hole.set('group', '5')
            hole.set('contype', '0')
            hole.set('conaffinity', '0')
            hole.set('rgba', '0.8 0.4 0.2 0.5')
        
        # Bumps
        for i, (x, y) in enumerate(bumps):
            bump = ET.SubElement(worldbody, 'geom')
            bump.set('name', f'floor_bump_{i}')
            bump.set('type', 'box')
            bump.set('material', 'mat_bump')
            bump.set('pos', f'{x:.3f} {y:.3f} 0.275')
            bump.set('size', f'{self.bump_size[0]:.3f} {self.bump_size[1]:.3f} {self.bump_size[2]:.3f}')
        
        # Convertir en string XML
        xml_str = ET.tostring(root, encoding='unicode')
        
        # Formatter pour lisibilité
        import xml.dom.minidom
        dom = xml.dom.minidom.parseString(xml_str)
        formatted_xml = dom.toprettyxml(indent='  ')
        
        # Nettoyer les lignes vides
        lines = [line for line in formatted_xml.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def save_corridor(self, filename: str, length: float = 100.0, width: float = 3.0, seed: int = None):
        """Sauvegarde un corridor généré."""
        xml_content = self.generate_corridor_xml(length, width, seed, filename.replace('.xml', ''))
        
        with open(filename, 'w') as f:
            f.write(xml_content)
        
        print(f"Corridor sauvegardé: {filename}")
        return filename


def generate_multiple_corridors():
    """Génère plusieurs corridors avec différents seeds."""
    generator = CorridorGenerator()
    
    corridors = [
        ("corridor_generated_1.xml", 100.0, 3.0, 42),
        ("corridor_generated_2.xml", 100.0, 3.0, 123),
        ("corridor_generated_3.xml", 80.0, 3.0, 456),
        ("corridor_generated_4.xml", 120.0, 3.0, 789),
        ("corridor_generated_wide.xml", 100.0, 4.0, 999),
    ]
    
    print("=" * 60)
    print("GÉNÉRATION DE CORRIDORS MULTIPLES")
    print("=" * 60)
    
    for filename, length, width, seed in corridors:
        generator.save_corridor(filename, length, width, seed)
        
        # Statistiques
        bumps = generator.generate_bump_pattern(length, seed)
        holes = generator.generate_hole_pattern(length, seed)
        
        print(f"  {filename}: {length}m × {width}m, {len(bumps)} bumps, {len(holes)} trous")
    
    print(f"\n✅ {len(corridors)} corridors générés avec succès!")


if __name__ == "__main__":
    # Générer un corridor de test
    generator = CorridorGenerator()
    
    print("=" * 60)
    print("GÉNÉRATEUR DE CORRIDORS SIMILAIRES")
    print("=" * 60)
    
    # Test avec le même style que l'original
    test_file = "corridor_test_similar.xml"
    generator.save_corridor(test_file, length=100.0, width=3.05, seed=42)
    
    # Statistiques
    bumps = generator.generate_bump_pattern(100.0, 42)
    holes = generator.generate_hole_pattern(100.0, 42)
    
    print(f"\nStatistiques du corridor généré:")
    print(f"  Longueur: 100.0m")
    print(f"  Largeur: 3.05m")
    print(f"  Bumps: {len(bumps)}")
    print(f"  Trous: {len(holes)}")
    print(f"  Tuiles de sol: ~1200")
    
    print(f"\nPremiers bumps générés:")
    for i, (x, y) in enumerate(bumps[:10]):
        print(f"  Bump {i+1}: ({x:.2f}, {y:.2f})")
    
    print(f"\nPremiers trous générés:")
    for i, (x, y) in enumerate(holes[:5]):
        print(f"  Trou {i+1}: ({x:.2f}, {y:.2f})")
    
    # Générer plusieurs variantes
    print(f"\n" + "=" * 60)
    generate_multiple_corridors()