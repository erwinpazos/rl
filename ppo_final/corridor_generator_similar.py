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
        """Génère le pattern des bumps - serpentin avec point de départ et direction aléatoires."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        bumps = []
        x = 2.25  # Position de départ
        
        # Pattern serpentin avec point de départ ET direction ALÉATOIRES
        y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
        
        # Commencer à une position Y aléatoire
        start_idx = random.randint(0, len(y_positions) - 1)
        
        # Direction aléatoire dès le début
        if start_idx == 0:
            direction = 1  # Forcé vers la droite si on est tout à gauche
        elif start_idx == len(y_positions) - 1:
            direction = -1  # Forcé vers la gauche si on est tout à droite
        else:
            direction = random.choice([1, -1])  # Direction vraiment aléatoire
        
        current_idx = start_idx
        
        while x < length - 1.0:
            y = y_positions[current_idx]
            bumps.append((x, y))
            
            # Espacement avec variation
            if random.random() < 0.1:  # 10% de chance de gap plus grand
                spacing = random.uniform(3.0, 4.0)
            else:
                spacing = random.uniform(1.5, 2.5)
            
            x += spacing
            
            # Avancer dans le serpentin
            current_idx += direction
            
            # Inverser direction aux extrêmes
            if current_idx >= len(y_positions):
                current_idx = len(y_positions) - 2
                direction = -1
            elif current_idx < 0:
                current_idx = 1
                direction = 1
        
        return bumps
    
    def generate_hole_pattern(self, length: float, seed: int = None) -> List[Tuple[float, float]]:
        """Génère le pattern des trous avec positions aléatoires."""
        if seed is not None:
            random.seed(seed + 1000)  # Seed différent pour les trous
        
        holes = []
        
        # Positions Y possibles pour les trous
        y_positions = [-0.5, 0.5, 1.0]
        
        # Approche plus robuste: générer des trous à intervalles plus réguliers
        # Basé sur l'analyse: 15 trous sur 100m = 1 trou tous les ~6.7m en moyenne
        target_holes = max(8, int(length / 7.0))  # Au moins 8 trous, ou 1 tous les 7m
        
        # GARANTIR des trous dans les premiers 20m (zone de test du visualizer)
        early_holes = max(2, target_holes // 4)  # Au moins 2 trous dans les premiers 20m
        
        # Diviser le corridor en 2 zones: début (0-20m) et reste (20m-fin)
        early_zone_end = min(20.0, length * 0.3)  # 20m ou 30% du corridor
        
        # Zone 1: Premiers 20m - garantir des trous visibles
        start_x = 6.25
        for i in range(early_holes):
            x = start_x + i * (early_zone_end - start_x) / early_holes
            # Variation aléatoire ±1m
            x += random.uniform(-1.0, 1.0)
            x = max(start_x, min(early_zone_end - 1.0, x))  # Garder dans la zone
            
            y = random.choice(y_positions)
            holes.append((x, y))
        
        # Zone 2: Reste du corridor
        remaining_holes = target_holes - early_holes
        if remaining_holes > 0:
            segment_length = (length - 5.0 - early_zone_end) / remaining_holes
            
            for i in range(remaining_holes):
                segment_start = early_zone_end + i * segment_length
                segment_end = early_zone_end + (i + 1) * segment_length
                
                # Variation aléatoire dans le segment
                variation = segment_length * 0.3
                x = random.uniform(
                    max(segment_start, segment_start + variation),
                    min(segment_end, segment_end - variation)
                )
                
                y = random.choice(y_positions)
                holes.append((x, y))
        
        return holes
    
    def generate_floor_tiles(self, length: float, width: float, holes: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Génère toutes les tuiles de sol selon le pattern original, SAUF aux positions des trous.
        
        IMPORTANT: Les trous sont des ABSENCES de sol, pas des overlays visuels.
        Un trou de 0.25×0.5 centré sur (x, y) supprime les tuiles de sol à:
        - (x, y-0.25) et (x, y+0.25)
        """
        tiles = []
        
        # Pattern original : positions Y de -1.25 à +1.25 par pas de 0.5
        # X de 0.25 à length par pas de 0.5
        x_positions = np.arange(0.25, length, 0.5)
        y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]  # 6 positions Y exactes
        
        # Créer set des positions de trous pour vérification rapide
        hole_positions = set()
        for hole_x, hole_y in holes:
            # IMPORTANT: Arrondir les positions des trous à la grille des tuiles
            # Les tuiles sont à X = 0.25, 0.75, 1.25, ... (pas de 0.5)
            # Les trous peuvent être à des positions flottantes, il faut les mapper à la grille
            grid_x = round(hole_x / 0.5) * 0.5 + 0.25  # Arrondir à la grille des tuiles
            
            # Un trou de taille 0.25×0.5 centré sur hole_y supprime 2 tuiles de sol
            # Exemple: trou à (6.25, -0.5) supprime tuiles à (6.25, -0.75) et (6.25, -0.25)
            tile_y1 = hole_y - 0.25  # Tuile du bas
            tile_y2 = hole_y + 0.25  # Tuile du haut
            hole_positions.add((round(grid_x, 2), round(tile_y1, 2)))
            hole_positions.add((round(grid_x, 2), round(tile_y2, 2)))
        
        # Générer toutes les tuiles SAUF celles supprimées par les trous
        for x in x_positions:
            for y in y_positions:
                # Vérifier si cette position est supprimée par un trou
                if (round(x, 2), round(y, 2)) not in hole_positions:
                    tiles.append((x, y))
        
        return tiles
    
    def generate_corridor_xml(self, length: float = 100.0, width: float = 3.0, 
                            seed: int = None, name: str = "generated_corridor") -> str:
        """Génère le XML complet du corridor."""
        
        # Générer les patterns
        bumps = self.generate_bump_pattern(length, seed)
        holes = self.generate_hole_pattern(length, seed)
        floor_tiles = self.generate_floor_tiles(length, width, holes)  # Passer les trous pour les exclure
        
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
        
        # Murs (positions exactes de l'original)
        wall_thickness = 0.025
        wall_height = 2.5
        wall_length = length / 2
        wall_center_x = length / 2
        
        # Mur gauche (Y = -1.525)
        wall_left = ET.SubElement(worldbody, 'geom')
        wall_left.set('name', 'wall_left')
        wall_left.set('type', 'box')
        wall_left.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_left.set('pos', f'{wall_center_x:.3f} -1.525 {wall_height:.3f}')
        wall_left.set('material', 'mat_wall')
        
        # Mur droit (Y = +1.525)
        wall_right = ET.SubElement(worldbody, 'geom')
        wall_right.set('name', 'wall_right')
        wall_right.set('type', 'box')
        wall_right.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_right.set('pos', f'{wall_center_x:.3f} 1.525 {wall_height:.3f}')
        wall_right.set('material', 'mat_wall')
        
        # Tuiles de sol (AVEC ABSENCES pour les trous - c'est ça le secret!)
        for i, (x, y) in enumerate(floor_tiles):
            tile = ET.SubElement(worldbody, 'geom')
            tile.set('type', 'box')
            tile.set('material', 'mat_floor')
            tile.set('name', f'floor_flat_{i}')
            tile.set('size', f'{self.floor_tile_size[0]:.3f} {self.floor_tile_size[1]:.3f} {self.floor_tile_size[2]:.3f}')
            tile.set('pos', f'{x:.3f} {y:.3f} {self.floor_tile_size[2]:.3f}')
        
        # Trous (géométries VISUELLES invisibles pour marquer les positions, OPTIONNEL)
        # Les vrais trous sont les ABSENCES de tuiles de sol ci-dessus
        for i, (x, y) in enumerate(holes):
            hole = ET.SubElement(worldbody, 'geom')
            hole.set('name', f'floor_hole_tile_{i}')
            hole.set('type', 'box')
            hole.set('size', f'{self.hole_size[0]:.3f} {self.hole_size[1]:.3f} {self.hole_size[2]:.3f}')  # 0.25 × 0.5 × 0.025
            hole.set('pos', f'{x:.3f} {y:.3f} {self.hole_size[2]:.3f}')
            hole.set('group', '5')  # Invisible (comme dans l'original)
            hole.set('contype', '0')  # Pas de collision
            hole.set('conaffinity', '0')  # Pas d'affinité
            hole.set('rgba', '0.8 0.4 0.2 0.5')  # Couleur orange transparente
        
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