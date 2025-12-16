"""
Visualize the corridor cell map as the robot would see it.
INSPIRÉ DU CODE QUI MARCHE dans ppo/visualize_corridor_map_old.py
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

def visualize_model_exact_vision(corridor_xml="corridor_100.xml", random_spawn=True, test_position=None):
    """Visualize EXACTLY what the model sees: robot state + wheel positions + grid with rows."""
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    if test_position:
        # Position de test spécifique
        import numpy as np
        robot_x, robot_y, robot_angle = test_position
        env.data.qpos[0] = robot_x
        env.data.qpos[1] = robot_y
        env.data.qpos[2] = 0.45
        env.data.qpos[3] = np.cos(robot_angle / 2)
        env.data.qpos[4] = 0
        env.data.qpos[5] = 0
        env.data.qpos[6] = np.sin(robot_angle / 2)
        obs = env._get_obs()
    elif random_spawn:
        # Spawn COMPLÈTEMENT aléatoire dans tout le couloir
        import numpy as np
        robot_x = np.random.uniform(5.0, 95.0)  # X aléatoire dans tout le couloir
        robot_y = np.random.uniform(-1.0, 1.0)  # Y aléatoire
        robot_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Angle aléatoire
        
        # Positionner le robot
        env.data.qpos[0] = robot_x
        env.data.qpos[1] = robot_y
        env.data.qpos[2] = 0.45
        env.data.qpos[3] = np.cos(robot_angle / 2)  # Quaternion
        env.data.qpos[4] = 0
        env.data.qpos[5] = 0
        env.data.qpos[6] = np.sin(robot_angle / 2)
        
        # Obtenir observation
        obs = env._get_obs()
    else:
        # Position manuelle
        env.data.qpos[0] = 10.0
        env.data.qpos[1] = 0.0
        obs = env._get_obs()
        robot_x = 10.0
        robot_y = 0.0
        robot_angle = 0.0
    
    # Décoder l'observation EXACTE du modèle
    robot_state = obs[:6]  # pos(3) + vel(3)
    wheel_positions = obs[6:14]  # 4 roues × (row, col)
    grid = obs[14:].reshape(32, 64)  # Grille 32×64 CORRIGÉE
    
    print("="*80)
    print("VISION EXACTE DU MODÈLE (2062 valeurs d'observation)")
    print("="*80)
    print(f"Robot position: x={robot_state[0]:.3f}m, y={robot_state[1]:.3f}m, z={robot_state[2]:.3f}m")
    print(f"Robot velocity: vx={robot_state[3]:.3f}, vy={robot_state[4]:.3f}, vz={robot_state[5]:.3f}")
    print(f"Robot angle: {np.degrees(robot_angle):.1f}°")
    print()
    
    # Positions des roues (comme le modèle les voit)
    wheel_names = ['FL', 'FR', 'RL', 'RR']
    print("POSITIONS DES ROUES (dans la grille, comme vues par le modèle):")
    for i, name in enumerate(wheel_names):
        row = int(wheel_positions[i*2])
        col = int(wheel_positions[i*2+1])
        print(f"  {name}: row={row:3d}, col={col:2d}")
    print()
    
    # Grille EXACTE 32×64 - afficher centre 20×40 (toute la largeur)
    robot_row_grid = 16  # Robot au centre de la grille 32×64
    robot_col_grid = 32  # Robot au centre des colonnes
    print("GRILLE 32×64 (centre 20×40) - EXACTEMENT comme vue par le modèle:")
    print("Cellules 6.25cm, vision 2×4m FIXE (toute largeur couloir)")
    print("Colonnes: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    print("          " + "─" * 40)
    
    symbols = {0: '▓', 1: '△', 2: '░', 3: 'R', 4: 'r', 5: 'X'}  # Robot avec info sol dessous
    
    for i in range(6, 26):  # Lignes 6-25 (centre 20 lignes)
        line = f"L{i:2d}: "
        
        # Créer la ligne avec les symboles (toute la largeur)
        for j in range(12, 52):  # Colonnes 12-51 (centre 40 colonnes)
            val = int(grid[i, j])
            line += symbols.get(val, '?')
        
        # Distance relative au robot (robot au centre ligne 16)
        relative_dist = (i - 16) * env.cell_size  # Robot à ligne 16
        print(f"{line} ({relative_dist:+.3f}m)")
    
    print()
    print("LÉGENDE:")
    print("  ▓ = sol (0)    △ = rampe (1)    ░ = trou (2)")
    print("  R = robot sur sol (3)    r = robot sur rampe (4)    X = robot sur trou (5)")
    print("  Vision: 32×64 cellules de 6.25cm = 2×4m FIXE (toute largeur couloir)")
    print("  Robot ligne 16 (X centré), colonne variable selon Y. Voit TOUJOURS toute largeur couloir")
    print()
    print("OBSERVATION TOTALE:")
    print(f"  Robot state: 6 valeurs {robot_state}")
    print(f"  Wheel positions: 8 valeurs {wheel_positions}")
    print(f"  Grid: 2048 valeurs (32×64)")
    print(f"  TOTAL: 2062 valeurs pour le réseau de neurones")
    print("="*80)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize corridor cell map")
    parser.add_argument("--corridor", type=str, default="corridor_100.xml", 
                       help="Corridor XML file to visualize (default: corridor_100.xml)")
    parser.add_argument("--model-vision", action="store_true",
                       help="Show EXACT model vision with random spawn and wheel positions")
    args = parser.parse_args()
    
    if args.model_vision:
        visualize_model_exact_vision(corridor_xml=args.corridor, random_spawn=True)
    else:
        visualize_full_corridor(corridor_xml=args.corridor)