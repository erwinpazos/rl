"""
Visualisation EXACTE de ce que le CNN simplifié reçoit en entrée.
Grille 120×80 avec cellules 0.05m, robot intégré dans la grille.
"""
from corridor_env import CorridorEnv
import numpy as np
import argparse

def visualize_full_corridor(corridor_xml="corridor_100.xml"):
    """Print the entire corridor cell map - INSPIRÉ DU CODE QUI MARCHE."""
    # Create environment
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Get the cell map
    cell_map = env.cell_map
    
    # Find dimensions COMME DANS LE CODE QUI MARCHE
    if not cell_map:
        print("ERROR: cell_map is empty!")
        return
        
    max_row = max(r for r, c in cell_map.keys()) if cell_map else 0
    max_col = max(c for r, c in cell_map.keys()) if cell_map else 0
    
    print("="*80)
    print("CORRIDOR CELL MAP COMPLET")
    print("="*80)
    print(f"Corridor XML: {corridor_xml}")
    print(f"Dimensions: {max_row+1} rows × {max_col+1} cols")
    print(f"Cell size: {env.cell_size}m × {env.cell_size}m (grille cell_map)")
    print(f"Total length: {(max_row+1) * env.cell_size}m")
    print(f"Total width: {(max_col+1) * env.cell_size}m")
    print()
    print("Legend: ▓ = flat (0)  △ = ramp (1)  ░ = hole (2)  ■ = bump (3)")
    print("="*80)
    print()
    
    # Print the map - CONVERTIR de 0.0625m vers affichage 0.25m
    max_display_row = min(max_row + 1, int(15.0 / 0.25))  # 15m ÷ 0.25m = 60 lignes d'affichage
    for display_row in range(max_display_row):
        line = f"Row {display_row:3d} ({display_row*0.25:5.1f}m): "
        
        # Chaque ligne d'affichage = 4 lignes de cell_map (0.25m ÷ 0.0625m = 4)
        for display_col in range(12):  # 12 colonnes d'affichage (3m ÷ 0.25m)
            # Échantillonner 4×4 cellules de la cell_map pour chaque cellule d'affichage
            votes = []
            for sub_r in range(4):
                for sub_c in range(4):
                    cell_map_row = display_row * 4 + sub_r
                    cell_map_col = display_col * 4 + sub_c
                    cell_type = cell_map.get((cell_map_row, cell_map_col), 2)
                    votes.append(cell_type)
            
            # Prendre la majorité (ou trou si égalité)
            if votes.count(0) > 8:  # Majorité sol
                display_type = 0
            elif votes.count(1) > 8:  # Majorité rampe
                display_type = 1
            elif votes.count(3) > 8:  # Majorité bump
                display_type = 3
            else:  # Défaut trou
                display_type = 2
            if display_type == 0:
                line += "▓"  # flat
            elif display_type == 1:
                line += "△"  # ramp
            elif display_type == 2:
                line += "░"  # hole
            elif display_type == 3:
                line += "■"  # bump
            else:
                line += "?"
        print(line)
    
    print()
    print("="*80)
    
    # Statistics COMME DANS LE CODE QUI MARCHE
    total_cells = (max_row + 1) * (max_col + 1)
    flat_count = sum(1 for v in cell_map.values() if v == 0)
    ramp_count = sum(1 for v in cell_map.values() if v == 1)
    hole_count = sum(1 for v in cell_map.values() if v == 2)
    bump_count = sum(1 for v in cell_map.values() if v == 3)
    
    print("STATISTICS:")
    print(f"  Total cells: {total_cells}")
    print(f"  Flat cells:  {flat_count} ({100*flat_count/total_cells:.1f}%)")
    print(f"  Ramp cells:  {ramp_count} ({100*ramp_count/total_cells:.1f}%)")
    print(f"  Hole cells:  {hole_count} ({100*hole_count/total_cells:.1f}%)")
    print(f"  Bump cells:  {bump_count} ({100*bump_count/total_cells:.1f}%)")
    print("="*80)
    
    env.close()

def visualize_cnn_input(corridor_xml="corridor_100.xml", robot_x=None, robot_y=None, robot_angle=None):
    """Visualise EXACTEMENT ce que le CNN simplifié reçoit en entrée."""
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Position du robot
    if robot_x is None:
        robot_x = np.random.uniform(5.0, 95.0)  # Position aléatoire
    if robot_y is None:
        robot_y = np.random.uniform(-1.0, 1.0)
    if robot_angle is None:
        robot_angle = np.random.uniform(-np.pi/4, np.pi/4)
    
    # Reset pour initialiser l'historique
    env.reset()
    
    # Positionner le robot à la position voulue
    env.data.qpos[0] = robot_x
    env.data.qpos[1] = robot_y
    env.data.qpos[2] = 0.45
    env.data.qpos[3] = np.cos(robot_angle / 2)
    env.data.qpos[4] = 0
    env.data.qpos[5] = 0
    env.data.qpos[6] = np.sin(robot_angle / 2)
    
    # Mettre à jour l'historique avec la nouvelle position
    env._update_position_history()
    
    # Obtenir observation EXACTE
    obs = env._get_obs()
    
    # Décoder l'observation AVEC HISTORIQUE DES 4 COINS
    robot_state = obs[:6]  # pos(3) + vel(3)
    bbox_corners = obs[6:14]  # 4 coins actuels × (row, col)
    corners_history = obs[14:54].reshape(5, 8)  # 5 positions × 8 coords relatives
    grid = obs[54:].reshape(120, 60)  # Grille 120×60
    
    print("="*100)
    print("ENTRÉE EXACTE DU CNN AVEC HISTORIQUE")
    print("="*100)
    print(f"Observation totale: {obs.shape[0]} valeurs (6 + 8 + 40 + {120*60} = {6 + 8 + 40 + 120*60})")
    print(f"Robot position: x={robot_state[0]:.3f}m, y={robot_state[1]:.3f}m, z={robot_state[2]:.3f}m")
    print(f"Robot velocity: vx={robot_state[3]:.3f}, vy={robot_state[4]:.3f}, vz={robot_state[5]:.3f}")
    print(f"Robot angle: {np.degrees(robot_angle):.1f}°")
    print()
    
    # Bounding box corners
    corner_names = ['AV-G', 'AV-D', 'AR-G', 'AR-D']
    print("BOUNDING BOX CORNERS (dans le repère grille):")
    for i, name in enumerate(corner_names):
        row = bbox_corners[i*2]
        col = bbox_corners[i*2+1]
        print(f"  {name}: row={row:6.1f}, col={col:6.1f}")
    print()
    
    # Historique des 4 coins (coordonnées relatives)
    print("HISTORIQUE DES 4 COINS (coordonnées relatives à la position actuelle):")
    corner_names = ['AV-G', 'AV-D', 'AR-G', 'AR-D']
    for i in range(5):
        corners = corners_history[i]  # 8 valeurs
        steps_ago = (4-i) * 20  # 20 steps par position
        if steps_ago == 0:
            print(f"  Actuelle (diff=0):  ", end="")
        else:
            print(f"  -{steps_ago:2d} steps:        ", end="")
        
        # Afficher les 4 coins
        corner_str = []
        for j, name in enumerate(corner_names):
            row_diff = corners[j*2]
            col_diff = corners[j*2+1]
            corner_str.append(f"{name}:({row_diff:+.1f},{col_diff:+.1f})")
        print(" | ".join(corner_str))
    print()
    
    # Grille 120×60 - afficher TOUTE LA LARGEUR
    print("GRILLE 120×60 (centre 40×60) - ENTRÉE DIRECTE DU CNN:")
    print(f"Cellules {env.cell_size}m, vision {env.vision_length}m×{env.vision_width}m")
    print("Vision Y FIXE: couvre exactement toute la largeur du couloir (3m)")
    print(f"Robot à ligne 40 (fixe), colonne variable selon sa position Y")
    print()
    
    # En-tête colonnes (toute la largeur)
    print("    ", end="")
    for j in range(0, 60):  # TOUTES les colonnes 0-59
        print(f"{j%10}", end="")
    print()
    print("    " + "─" * 60)
    
    # Afficher grille avec symboles
    for i in range(20, 60):  # Lignes 20-59 (centre 40 lignes)
        line = f"{i:3d}|"
        
        for j in range(0, 60):  # TOUTES les colonnes 0-59
            val = grid[i, j]
            
            # Vérifier si c'est un coin de la bounding box
            is_corner = False
            corner_type = ''
            for k, name in enumerate(corner_names):
                corner_row = round(bbox_corners[k*2])
                corner_col = round(bbox_corners[k*2+1])
                if corner_row == i and corner_col == j:
                    is_corner = True
                    if 'AV' in name:
                        corner_type = 'A'  # Avant
                    else:
                        corner_type = 'R'  # aRrière
                    break
            
            if is_corner:
                symbol = corner_type  # A pour avant, A pour arrière
            else:
                # Symboles selon valeur normalisée
                if val == 0.0:
                    symbol = '▓'  # Sol
                elif val == 0.5:
                    symbol = '△'  # Rampe
                elif val == 1.0:
                    symbol = '░'  # Trou
                else:
                    symbol = '?'  # Valeur inattendue
            
            line += symbol
        
        # Distance relative au robot (robot théoriquement à ligne 40)
        relative_dist = (i - 40) * env.cell_size
        print(f"{line} {relative_dist:+.2f}m")
    
    print()
    print("LÉGENDE:")
    print("  ▓ = sol (0.0)    △ = rampe (0.5)    ░ = trou (1.0)")
    print("  A = coin AVANT de la bounding box    R = coin ARRIÈRE de la bounding box")
    print(f"  Vision: {env.grid_rows}×{env.grid_cols} cellules de {env.cell_size}m")
    print(f"  Couvre: {env.vision_length}m devant/derrière × {env.vision_width}m largeur FIXE")
    print(f"  Vision Y FIXE: toujours centrée sur le couloir, pas sur le robot")
    print(f"  Robot représenté par ses 4 coins de bounding box (pas rempli dans la grille)")
    print()
    
    # Statistiques de la grille
    unique_vals, counts = np.unique(grid, return_counts=True)
    print("STATISTIQUES GRILLE:")
    total_cells = grid.size
    for val, count in zip(unique_vals, counts):
        pct = 100 * count / total_cells
        if val == 0.0:
            print(f"  Sol (0.0):    {count:5d} cellules ({pct:5.1f}%)")
        elif val == 0.5:
            print(f"  Rampe (0.5):  {count:5d} cellules ({pct:5.1f}%)")
        elif val == 0.75:
            print(f"  Robot (0.75): {count:5d} cellules ({pct:5.1f}%)")
        elif val == 1.0:
            print(f"  Trou (1.0):   {count:5d} cellules ({pct:5.1f}%)")
        else:
            print(f"  Autre ({val:.2f}): {count:5d} cellules ({pct:5.1f}%)")
    
    print()
    print("STRUCTURE OBSERVATION POUR CNN:")
    print(f"  1. Robot state:      6 valeurs {robot_state}")
    print(f"  2. BBox corners:     8 valeurs {bbox_corners}")
    print(f"  3. Corners history:  40 valeurs (5 positions × 8 coords relatives)")
    print(f"  4. Grille unifiée:   {grid.size} valeurs (120×60)")
    print(f"  TOTAL:              {obs.shape[0]} valeurs → CNN")
    print("="*100)
    
    env.close()

def test_multiple_positions(corridor_xml="corridor_100.xml"):
    """Teste plusieurs positions pour voir la cohérence."""
    positions = [
        (10.0, 0.0, 0.0),      # Début, centré
        (25.0, -0.5, 0.1),     # Milieu, légèrement à gauche
        (50.0, 1.0, -0.2),     # Milieu, à droite
        (75.0, 0.0, 0.0),      # Fin, centré
    ]
    
    for i, (x, y, angle) in enumerate(positions):
        print(f"\n{'='*20} POSITION {i+1} {'='*20}")
        visualize_cnn_input(corridor_xml, x, y, angle)
        if i < len(positions) - 1:
            input("Appuyez sur Entrée pour la position suivante...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise l'entrée exacte du CNN simplifié")
    parser.add_argument("--corridor", type=str, default="corridor_100.xml", 
                       help="Fichier XML du corridor (défaut: corridor_100.xml)")
    parser.add_argument("--x", type=float, help="Position X du robot (défaut: aléatoire)")
    parser.add_argument("--y", type=float, help="Position Y du robot (défaut: aléatoire)")
    parser.add_argument("--angle", type=float, help="Angle du robot en DEGRÉS (défaut: aléatoire)")
    parser.add_argument("--test-multiple", action="store_true",
                       help="Teste plusieurs positions prédéfinies")
    parser.add_argument("--full-map", action="store_true",
                       help="Affiche la carte complète du corridor")
    args = parser.parse_args()
    
    if args.test_multiple:
        test_multiple_positions(args.corridor)
    elif args.full_map:
        visualize_full_corridor(args.corridor)
    else:
        # Convertir angle de degrés en radians si fourni
        angle_rad = np.radians(args.angle) if args.angle is not None else None
        visualize_cnn_input(args.corridor, args.x, args.y, angle_rad)