"""
Visualisation EXACTE de ce que le CNN reçoit en entrée.
Grille 60×30×2 avec 2 canaux binaires, robot intégré dans la grille.
"""
from corridor_env import CorridorEnv
import numpy as np
import argparse

def visualize_full_corridor(corridor_xml="corridor_100.xml"):
    """Print the entire corridor cell map."""
    # Create environment
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Get the cell map
    cell_map = env.cell_map
    
    # Find dimensions
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
    print("Legend: ▓ = flat (0)  △ = bump (1)  ░ = hole (2)")
    print("="*80)
    print()
    
    # Print the map
    max_display_row = min(max_row + 1, int(15.0 / 0.25))
    for display_row in range(max_display_row):
        line = f"Row {display_row:3d} ({display_row*0.25:5.1f}m): "
        
        for display_col in range(12):
            votes = []
            for sub_r in range(4):
                for sub_c in range(4):
                    cell_map_row = display_row * 4 + sub_r
                    cell_map_col = display_col * 4 + sub_c
                    cell_type = cell_map.get((cell_map_row, cell_map_col), 2)
                    votes.append(cell_type)
            
            if votes.count(0) > 8:
                display_type = 0
            elif votes.count(1) > 8:
                display_type = 1
            else:
                display_type = 2
                
            if display_type == 0:
                line += "▓"
            elif display_type == 1:
                line += "△"
            elif display_type == 2:
                line += "░"
            else:
                line += "?"
        print(line)
    
    print()
    print("="*80)
    
    # Statistics
    total_cells = (max_row + 1) * (max_col + 1)
    flat_count = sum(1 for v in cell_map.values() if v == 0)
    bump_count = sum(1 for v in cell_map.values() if v == 1)
    hole_count = sum(1 for v in cell_map.values() if v == 2)
    
    print("STATISTICS:")
    print(f"  Total cells: {total_cells}")
    print(f"  Flat cells:  {flat_count} ({100*flat_count/total_cells:.1f}%)")
    print(f"  Bump cells:  {bump_count} ({100*bump_count/total_cells:.1f}%)")
    print(f"  Hole cells:  {hole_count} ({100*hole_count/total_cells:.1f}%)")
    print("="*80)
    
    env.close()

def visualize_cnn_input(corridor_xml="corridor_100.xml", robot_x=None, robot_y=None, robot_angle=None):
    """Visualise EXACTEMENT ce que le CNN reçoit en entrée avec 2 canaux."""
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Position du robot
    if robot_x is None:
        robot_x = np.random.uniform(5.0, 95.0)
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
    
    # Décoder l'observation AVEC HISTORIQUE SIMPLIFIÉ
    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
    history_simplified = obs[7:55].reshape(8, 6)  # 8 frames × 6 valeurs (3 positions + 3 vitesses)
    grid = obs[55:].reshape(60, 30, 2)  # Grille 60×30×2
    
    print("="*100)
    print("ENTRÉE EXACTE DU CNN AVEC HISTORIQUE SIMPLIFIÉ - 2 CANAUX")
    print("="*100)
    print(f"Observation totale: {obs.shape[0]} valeurs (7 + 48 + {60*30*2} = {7 + 48 + 60*30*2})")
    print(f"Robot position: x={robot_state[0]:.3f}m, y={robot_state[1]:.3f}m, z={robot_state[2]:.3f}m")
    print(f"Robot velocity: vx={robot_state[3]:.3f}, vy={robot_state[4]:.3f}, vz={robot_state[5]:.3f}")
    print(f"Robot angle: {np.degrees(robot_angle):.1f}°")
    print()
    
    # Historique simplifié (positions + vitesses, pas de bounding box)
    print("HISTORIQUE SIMPLIFIÉ (8 frames × 6 valeurs: 3 positions + 3 vitesses relatives):")
    for i in range(8):
        frame_data = history_simplified[i]  # 6 valeurs
        positions = frame_data[:3]  # 3 premiers = positions
        velocities = frame_data[3:]  # 3 derniers = vitesses
        
        steps_ago = (7-i) * 10  # 10 steps par position
        if steps_ago == 0:
            print(f"  Actuelle (diff=0):  ", end="")
        else:
            print(f"  -{steps_ago:2d} steps:        ", end="")
        
        # Afficher position et vitesses relatives
        pos_str = f"pos:({positions[0]:+.2f},{positions[1]:+.2f},{positions[2]:+.2f})"
        vel_str = f"vel:({velocities[0]:+.2f},{velocities[1]:+.2f},{velocities[2]:+.2f})"
        print(f"{pos_str} | {vel_str}")
    print()
    
    # Grille 60×30×2 - CENTRÉE ET ORIENTÉE selon le robot
    print("GRILLE 60×30×2 - ENTRÉE DIRECTE DU CNN:")
    print(f"Cellules {env.cell_size}m, vision {env.vision_length}m×{env.vision_width}m")
    print("Vision EGO-CENTRIQUE: grille centrée ET orientée selon le robot")
    print(f"Robot à ligne {env.robot_row_in_grid} (0.8m derrière), colonne {env.grid_cols//2} (centre)")
    print(f"La grille TOURNE avec le robot (angle={np.degrees(robot_angle):.1f}°)")
    print("Le robot voit toujours 'devant' vers le haut de la grille")
    print("2 CANAUX BINAIRES: [0]=Obstacles, [1]=Trous")
    print()
    
    # Afficher les 2 canaux séparément d'abord
    print("CANAL 0 - OBSTACLES (bumps + murs latéraux):")
    print("    ", end="")
    for j in range(0, 30, 3):  # Tous les 3 pour lisibilité
        print(f"{j:2d}", end=" ")
    print()
    print("    " + "─" * 30)
    
    for i in range(0, 20):  # 20 premières lignes
        line = f"{i:2d}|"
        for j in range(30):
            val = grid[i, j, 0]  # Canal 0
            if val > 0.5:
                line += '█'  # Obstacle
            else:
                line += '·'  # Libre
        relative_dist = (i - env.robot_row_in_grid) * env.cell_size
        print(f"{line} {relative_dist:+.1f}m")
    
    print()
    print("CANAL 1 - TROUS (trous + extérieur avant/arrière):")
    print("    ", end="")
    for j in range(0, 30, 3):
        print(f"{j:2d}", end=" ")
    print()
    print("    " + "─" * 30)
    
    for i in range(0, 20):  # 20 premières lignes
        line = f"{i:2d}|"
        for j in range(30):
            val = grid[i, j, 1]  # Canal 1
            if val > 0.5:
                line += '░'  # Trou
            else:
                line += '·'  # Libre
        relative_dist = (i - env.robot_row_in_grid) * env.cell_size
        print(f"{line} {relative_dist:+.1f}m")
    
    print()
    print("GRILLE COMBINÉE (ce que voit le robot):")
    print("    ", end="")
    for j in range(0, 30):  # TOUTES les colonnes 0-29
        print(f"{j%10}", end="")
    print()
    print("    " + "─" * 30)
    
    # Afficher grille combinée (combinaison des 2 canaux)
    for i in range(0, 30):  # Lignes 0-29 (robot à ligne 8)
        line = f"{i:3d}|"
        
        for j in range(0, 30):  # TOUTES les colonnes 0-29
            # Extraire les 2 canaux
            obstacle = grid[i, j, 0]
            trou = grid[i, j, 1]
            
            # Marquer la position du robot
            if i == env.robot_row_in_grid and j == env.grid_cols // 2:
                symbol = '🤖'  # Robot au centre
            else:
                # Symboles selon les 2 canaux binaires
                if obstacle > 0.5 and trou > 0.5:
                    symbol = '?'  # Erreur (ne devrait pas arriver)
                elif obstacle > 0.5:
                    symbol = '█'  # Obstacle (bump ou mur latéral)
                elif trou > 0.5:
                    symbol = '░'  # Trou (trou ou extérieur avant/arrière)
                else:
                    symbol = '▓'  # Sol navigable
            
            line += symbol
        
        # Distance relative au robot
        relative_dist = (i - env.robot_row_in_grid) * env.cell_size
        print(f"{line} {relative_dist:+.2f}m")
    
    print()
    print("LÉGENDE:")
    print("  █ = obstacle (bump OU mur latéral)    ▓ = sol navigable    ░ = trou (trou OU extérieur)")
    print("  🤖 = robot (toujours au centre de la grille ego-centrique)")
    print(f"  Vision: {env.grid_rows}×{env.grid_cols}×2 cellules de {env.cell_size}m")
    print(f"  Couvre: {env.vision_length}m devant/derrière × {env.vision_width}m largeur")
    print(f"  Vision EGO-CENTRIQUE: grille tourne avec le robot (angle={np.degrees(robot_angle):.1f}°)")
    print(f"  Robot toujours au centre (ligne {env.robot_row_in_grid}, col {env.grid_cols//2})")
    print(f"  'Devant' = toujours vers le haut de la grille, quelle que soit l'orientation réelle")
    print()
    
    # Statistiques des 2 canaux
    print("STATISTIQUES DES 2 CANAUX:")
    total_cells = grid.shape[0] * grid.shape[1]
    
    # Canal 0: Obstacles
    obstacle_count = np.sum(grid[:, :, 0] > 0.5)
    print(f"  Canal 0 (Obstacles): {obstacle_count:5d} cellules ({100*obstacle_count/total_cells:5.1f}%)  [█]")
    
    # Canal 1: Trous
    trou_count = np.sum(grid[:, :, 1] > 0.5)
    print(f"  Canal 1 (Trous):     {trou_count:5d} cellules ({100*trou_count/total_cells:5.1f}%)  [░]")
    
    # Sol navigable (les deux à 0)
    sol_count = np.sum((grid[:, :, 0] <= 0.5) & (grid[:, :, 1] <= 0.5))
    print(f"  Sol navigable:       {sol_count:5d} cellules ({100*sol_count/total_cells:5.1f}%)  [▓]")
    
    print()
    print("STRUCTURE OBSERVATION POUR CNN:")
    print(f"  1. Robot state:      7 valeurs {robot_state}")
    print(f"  2. History simplifié: 48 valeurs (8 frames × 6: 3 positions + 3 vitesses)")
    print(f"  3. Grille 2 canaux:  {grid.size} valeurs (60×30×2)")
    print(f"  TOTAL:              {obs.shape[0]} valeurs → CNN avec 2 canaux d'entrée")
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
    parser = argparse.ArgumentParser(description="Visualise l'entrée exacte du CNN avec 2 canaux")
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
        test_multiple_positions(args.corridor)
    else:
        # Convertir angle de degrés en radians si fourni
        angle_rad = np.radians(args.angle) if args.angle is not None else None
        visualize_cnn_input(args.corridor, args.x, args.y, angle_rad)