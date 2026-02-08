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
        self.hole_size = (0.25, 0.5, 0.025)  # Taille des trous (2 tuiles de large Y, 1 tuile de long X)
        
        # Positions Y possibles (6 niveaux pour bumps, 4 pour trous) - SYMÉTRIQUES
        self.bump_y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
        self.hole_y_positions = [-1.0, -0.5, 0.5, 1.0]
        
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
        x = 5.0  # Position de départ (zone de sécurité avant 5m)
        y_cycle_pos = 0  # Position dans le cycle Y
        
        # Pattern cyclique observé: -1.25 → -0.75 → -0.25 → 0.25 → 0.75 → 1.25 → 0.75 → 0.25 → ...
        y_pattern = [
            -1.25, -0.75, -0.25, 0.25, 0.75, 1.25,  # Montée
            0.75, 0.25, -0.25, -0.75, -1.25,        # Descente
        ]
        
        # NOUVEAU: Arrêter les bumps à 100m pour laisser une zone de succès
        while x < min(length, 100.0) - 1.0:
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
            np.random.seed(seed + 1000)
        
        holes = []
        x = 5.0  # Position de départ (zone de sécurité avant 5m)
        previous_y = None  # Tracker la position Y précédente
        
        # NOUVEAU: Positions Y possibles pour distribution aléatoire
        y_positions = [-1.0, -0.5, 0.5, 1.0]
        
        # NOUVEAU: Arrêter les trous à 100m pour laisser une zone de succès
        while x < min(length, 100.0) - 2.0:
            # NOUVEAU: Éviter que le trou suivant soit au même endroit Y que le précédent
            if previous_y is not None:
                # Créer une liste pondérée : position précédente = 10% de chance, autres = 30% chacune
                available_positions = []
                for y_pos in y_positions:
                    if y_pos == previous_y:
                        # Réduire la probabilité de répéter la même position Y
                        available_positions.extend([y_pos] * 1)  # 1 occurrence = ~10% de chance
                    else:
                        # Positions différentes ont plus de chance
                        available_positions.extend([y_pos] * 3)  # 3 occurrences = ~30% de chance chacune
                
                y = random.choice(available_positions)
            else:
                # Premier trou : position complètement aléatoire
                y = random.choice(y_positions)
            
            holes.append((x, y))
            previous_y = y  # Mémoriser pour le prochain trou
            
            # Espacement réduit pour plus de densité
            if random.random() < 0.08:  # 8% de chance de gap plus grand (réduit de 15%)
                spacing = random.uniform(5.0, 7.0)  # Gaps plus petits (au lieu de 8-12m)
            else:
                spacing = random.uniform(2.0, 3.0)  # Espacement de base réduit (au lieu de 4m fixe)
            
            x += spacing
        
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
    
    def generate_corridor_xml(self, length: float = 110.0, width: float = 3.0, 
                            seed: int = None, name: str = "generated_corridor",
                            obstacle_type: str = "holes", bump_ratio: float = 0.0) -> str:
        """Génère le XML complet du corridor.
        
        Args:
            length: Longueur du corridor en mètres
            width: Largeur du corridor en mètres  
            seed: Seed pour la génération aléatoire
            name: Nom du corridor
            obstacle_type: Type d'obstacles - DEPRECATED, utiliser bump_ratio à la place
            bump_ratio: Ratio de bumps à ajouter (0.0 = aucun, 0.5 = 1 sur 2, 1.0 = 100%)
                       Les bumps sont répartis uniformément (ex: 0.5 = bump, vide, bump, vide...)
        """
        
        # Toujours générer les holes en premier
        holes = self.generate_hole_pattern(length, seed)
        
        # Générer les bumps entre les holes
        bumps = []
        if bump_ratio > 0.0 and len(holes) > 1:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            # Trier les holes par position X
            holes_sorted = sorted(holes, key=lambda h: h[0])
            
            # Positions Y possibles pour les bumps
            y_positions = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
            
            # Calculer le nombre exact de bumps à placer
            num_spaces = len(holes_sorted) - 1  # Nombre d'espaces entre holes
            num_bumps = int(num_spaces * bump_ratio)  # Nombre exact de bumps
            
            # Répartir uniformément les bumps sur les espaces
            # Ex: 0.5 = prendre 1 espace sur 2, 0.33 = prendre 1 espace sur 3
            if num_bumps > 0:
                step = num_spaces / num_bumps  # Espacement entre bumps
                bump_indices = [int(i * step) for i in range(num_bumps)]
            else:
                bump_indices = []
            
            previous_bump_y = None  # Tracker la position Y du bump précédent
            
            # Pour chaque espace qui doit avoir un bump
            for bump_idx in bump_indices:
                if bump_idx < num_spaces:
                    hole1_x, hole1_y = holes_sorted[bump_idx]
                    hole2_x, hole2_y = holes_sorted[bump_idx + 1]
                    
                    # Calculer la position X au milieu entre les deux holes
                    middle_x = (hole1_x + hole2_x) / 2.0
                    
                    # Choisir une position Y avec pondération pour éviter répétition
                    if previous_bump_y is not None:
                        # Créer une liste pondérée : position précédente = 10% de chance, autres = 18% chacune
                        available_positions = []
                        for y_pos in y_positions:
                            if y_pos == previous_bump_y:
                                # Réduire la probabilité de répéter la même position Y
                                available_positions.extend([y_pos] * 1)  # 1 occurrence = ~10% de chance
                            else:
                                # Positions différentes ont plus de chance
                                available_positions.extend([y_pos] * 2)  # 2 occurrences = ~18% de chance chacune
                        
                        bump_y = random.choice(available_positions)
                    else:
                        # Premier bump : position complètement aléatoire
                        bump_y = random.choice(y_positions)
                    
                    bumps.append((middle_x, bump_y))
                    previous_bump_y = bump_y  # Mémoriser pour le prochain bump
        
        floor_tiles = self.generate_floor_tiles(length, width)
        
        # NOUVEAU: Supprimer les tuiles de sol aux positions des trous pour créer de vrais trous
        if holes:
            filtered_floor_tiles = []
            for tile_x, tile_y in floor_tiles:
                # Vérifier si cette tuile est dans une zone de trou
                is_in_hole = False
                for hole_x, hole_y in holes:
                    # Arrondir la position du trou à la grille des tuiles (0.25, 0.75, 1.25...)
                    # pour s'assurer qu'on supprime exactement les bonnes tuiles
                    hole_x_grid = round((hole_x - 0.25) / 0.5) * 0.5 + 0.25
                    
                    # Distance entre la tuile et le centre du trou aligné sur la grille
                    dx = abs(tile_x - hole_x_grid)
                    dy = abs(tile_y - hole_y)
                    
                    # Un trou fait 1 tuile en X (0.5m) et 2 tuiles en Y (0.5m)
                    # Donc on supprime les tuiles à dx < 0.25m (1 seule tuile) et dy < 0.5m (2 tuiles)
                    if dx < 0.25 and dy < 0.5:
                        is_in_hole = True
                        break
                
                # Garder seulement les tuiles qui ne sont pas dans un trou
                if not is_in_hole:
                    filtered_floor_tiles.append((tile_x, tile_y))
            
            floor_tiles = filtered_floor_tiles
        
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
        
        # Murs (position au centre de la box, donc décaler de wall_thickness/2)
        wall_thickness = 0.025
        wall_height = 2.5
        wall_length = length / 2
        wall_center_x = length / 2
        
        # Mur gauche (centre à -width/2, donc le mur va de -width/2-thickness à -width/2+thickness)
        # On veut que le bord intérieur soit à -width/2, donc centre à -width/2 + wall_thickness/2
        wall_left = ET.SubElement(worldbody, 'geom')
        wall_left.set('name', 'wall_left')
        wall_left.set('type', 'box')
        wall_left.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_left.set('pos', f'{wall_center_x:.3f} {-width/2 + wall_thickness/2:.3f} {wall_height:.3f}')
        wall_left.set('material', 'mat_wall')
        
        # Mur droit (centre à +width/2, donc le mur va de +width/2-thickness à +width/2+thickness)
        # On veut que le bord intérieur soit à +width/2, donc centre à +width/2 - wall_thickness/2
        wall_right = ET.SubElement(worldbody, 'geom')
        wall_right.set('name', 'wall_right')
        wall_right.set('type', 'box')
        wall_right.set('size', f'{wall_length:.3f} {wall_thickness:.3f} {wall_height:.3f}')
        wall_right.set('pos', f'{wall_center_x:.3f} {width/2 - wall_thickness/2:.3f} {wall_height:.3f}')
        wall_right.set('material', 'mat_wall')
        
        # Tuiles de sol (avec trous créés par suppression de tuiles)
        for i, (x, y) in enumerate(floor_tiles):
            tile = ET.SubElement(worldbody, 'geom')
            tile.set('type', 'box')
            tile.set('material', 'mat_floor')
            tile.set('name', f'floor_flat_{i}')
            tile.set('size', f'{self.floor_tile_size[0]:.3f} {self.floor_tile_size[1]:.3f} {self.floor_tile_size[2]:.3f}')
            tile.set('pos', f'{x:.3f} {y:.3f} {self.floor_tile_size[2]:.3f}')
        
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
    
    def save_corridor(self, filename: str, length: float = 110.0, width: float = 3.0, 
                     seed: int = None, obstacle_type: str = "holes", bump_ratio: float = 0.0):
        """Sauvegarde un corridor généré.
        
        Args:
            filename: Nom du fichier XML
            length: Longueur du corridor en mètres
            width: Largeur du corridor en mètres
            seed: Seed pour la génération aléatoire
            obstacle_type: DEPRECATED - utiliser bump_ratio
            bump_ratio: Ratio de bumps (0.0 à 1.0)
        """
        xml_content = self.generate_corridor_xml(length, width, seed, filename.replace('.xml', ''), obstacle_type, bump_ratio)
        
        with open(filename, 'w') as f:
            f.write(xml_content)
        
        obstacle_desc = f"holes + {int(bump_ratio*100)}% bumps"
        
        print(f"Corridor sauvegardé: {filename} ({obstacle_desc})")
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