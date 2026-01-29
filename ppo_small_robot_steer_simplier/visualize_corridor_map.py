"""
Visualisation EXACTE de ce que le CNN reçoit en entrée.
Grille 60×30×2 avec 2 canaux binaires, robot intégré dans la grille.
"""
from corridor_env import CorridorEnv
import numpy as np
import argparse
import mujoco
from mujoco import viewer
import time
import os
import sys

# Importer le générateur local
try:
    from corridor_generator_similar import CorridorGenerator
except ImportError:
    print("⚠️  Générateur de corridor non trouvé. Option --random désactivée.")
    CorridorGenerator = None

def generate_random_corridor(seed=None):
    """Génère un corridor aléatoire temporaire."""
    if CorridorGenerator is None:
        print("❌ Générateur de corridor non disponible!")
        return None
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    print(f"🎲 Génération d'un corridor aléatoire (seed={seed})...")
    
    generator = CorridorGenerator()
    temp_filename = f"temp_random_corridor_{seed}.xml"
    
    try:
        # Générer avec des paramètres variés
        length = np.random.uniform(80.0, 120.0)
        width = np.random.uniform(2.5, 3.5)
        
        generator.save_corridor(temp_filename, length, width, seed)
        
        # Statistiques
        bumps = generator.generate_bump_pattern(length, seed)
        holes = generator.generate_hole_pattern(length, seed)
        
        print(f"✅ Corridor généré: {length:.1f}m × {width:.1f}m")
        print(f"   {len(bumps)} bumps, {len(holes)} trous")
        
        return temp_filename
        
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return None


def cleanup_temp_files():
    """Nettoie les fichiers temporaires."""
    for file in os.listdir('.'):
        if file.startswith('temp_random_corridor_') and file.endswith('.xml'):
            try:
                os.remove(file)
                print(f"🗑️  Supprimé: {file}")
            except:
                pass


def render_corridor_3d(corridor_xml="corridor_3x100_no_full_obstacles.xml", robot_x=None, robot_y=None, robot_angle=None):
    """Ouvre le rendu 3D MuJoCo du corridor avec le robot."""
    print(f"🎬 RENDU 3D DU CORRIDOR: {corridor_xml}")
    
    # Créer environnement
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Reset pour initialiser
    obs, info = env.reset()
    
    # Positionner le robot si spécifié
    if robot_x is not None:
        env.data.qpos[0] = robot_x
    if robot_y is not None:
        env.data.qpos[1] = robot_y
    if robot_angle is not None:
        # Convertir angle en quaternion
        env.data.qpos[3] = np.cos(robot_angle / 2)
        env.data.qpos[4] = 0
        env.data.qpos[5] = 0
        env.data.qpos[6] = np.sin(robot_angle / 2)
    
    # Mettre à jour la simulation
    mujoco.mj_forward(env.model, env.data)
    
    print(f"Position robot: x={env.data.qpos[0]:.2f}m, y={env.data.qpos[1]:.2f}m")
    print(f"Angle robot: {np.degrees(2 * np.arctan2(env.data.qpos[6], env.data.qpos[3])):.1f}°")
    print("\n🎮 CONTRÔLES:")
    print("  Souris : Rotation caméra")
    print("  Molette : Zoom")
    print("  ESC : Quitter")
    print("  C : Info caméra")
    
    # Variables globales pour le callback
    quit_requested = False
    
    def key_callback(keycode):
        nonlocal quit_requested
        if keycode == 256:  # ESC
            quit_requested = True
        elif keycode == ord('c') or keycode == ord('C'):
            print(f"Caméra - Azimuth: {v.cam.azimuth:.1f}°, Elevation: {v.cam.elevation:.1f}°, Distance: {v.cam.distance:.1f}m")
    
    try:
        with viewer.launch_passive(env.model, env.data, key_callback=key_callback) as v:
            # Configuration caméra
            robot_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = robot_id
            v.cam.azimuth = 180
            v.cam.elevation = -20
            v.cam.distance = 8
            
            print("✅ Rendu 3D ouvert ! Explorez le corridor...")
            
            while v.is_running() and not quit_requested:
                v.sync()
                time.sleep(0.02)  # 50 FPS
                
    except Exception as e:
        print(f"❌ Erreur rendu 3D: {e}")
    finally:
        env.close()
        print("🔚 Rendu 3D fermé")


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
    history_simplified = obs[7:31].reshape(4, 6)  # 4 frames × 6 valeurs (3 positions + 3 vitesses)
    grid = obs[31:].reshape(57, 40, 2)  # Grille 57×40×2 (2 canaux, plus large)
    
    print("="*100)
    print("ENTRÉE EXACTE DU CNN AVEC HISTORIQUE SIMPLIFIÉ - 2 CANAUX")
    print("="*100)
    print(f"Observation totale: {obs.shape[0]} valeurs (7 + 24 + {57*40*2} = {7 + 24 + 57*40*2})")
    print(f"Robot position: x={robot_state[0]:.3f}m, y={robot_state[1]:.3f}m, z={robot_state[2]:.3f}m")
    print(f"Robot velocity: vx={robot_state[3]:.3f}, vy={robot_state[4]:.3f}, vz={robot_state[5]:.3f}")
    print(f"Robot angle: {np.degrees(robot_angle):.1f}°")
    print()
    
    # Historique simplifié (positions + vitesses, pas de bounding box)
    print("HISTORIQUE SIMPLIFIÉ (4 frames × 6 valeurs: 3 positions + 3 vitesses relatives):")
    for i in range(4):
        frame_data = history_simplified[i]  # 6 valeurs
        positions = frame_data[:3]  # 3 premiers = positions
        velocities = frame_data[3:]  # 3 derniers = vitesses
        
        steps_ago = (3-i) * 15  # 15 steps par position (interval réduit)
        if steps_ago == 0:
            print(f"  Actuelle (diff=0):  ", end="")
        else:
            print(f"  -{steps_ago:2d} steps:        ", end="")
        
        # Afficher position et vitesses relatives
        pos_str = f"pos:({positions[0]:+.2f},{positions[1]:+.2f},{positions[2]:+.2f})"
        vel_str = f"vel:({velocities[0]:+.2f},{velocities[1]:+.2f},{velocities[2]:+.2f})"
        print(f"{pos_str} | {vel_str}")
    print()
    
    # Grille 57×40×2 - CENTRÉE ET ORIENTÉE selon le robot
    print("GRILLE 57×40×2 - ENTRÉE DIRECTE DU CNN:")
    print(f"Cellules {env.cell_size}m, vision {env.vision_length}m×{env.vision_width}m")
    print("Vision EGO-CENTRIQUE: grille centrée ET orientée selon le robot")
    print(f"Robot à ligne {env.robot_row_in_grid} (0.5m derrière), colonne {env.grid_cols//2} (centre)")
    print(f"La grille TOURNE avec le robot (angle={np.degrees(robot_angle):.1f}°)")
    print("Le robot voit toujours 'devant' vers le haut de la grille")
    print("2 CANAUX BINAIRES: [0]=Obstacles, [1]=Trous")
    print()
    
    # Afficher les 2 canaux séparément d'abord
    print("CANAL 0 - OBSTACLES (bumps + murs latéraux):")
    print("    ", end="")
    for j in range(0, 40, 4):  # Tous les 4 pour lisibilité (40 colonnes)
        print(f"{j:2d}", end=" ")
    print()
    print("    " + "─" * 40)
    
    for i in range(0, 20):  # 20 premières lignes
        line = f"{i:2d}|"
        for j in range(40):  # 40 colonnes
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
    for j in range(0, 40, 4):  # Tous les 4 pour lisibilité
        print(f"{j:2d}", end=" ")
    print()
    print("    " + "─" * 40)
    
    for i in range(0, 20):  # 20 premières lignes
        line = f"{i:2d}|"
        for j in range(40):  # 40 colonnes
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
    for j in range(0, 40):  # TOUTES les colonnes 0-39
        print(f"{j%10}", end="")
    print()
    print("    " + "─" * 40)
    
    # Afficher grille combinée (combinaison des 2 canaux)
    for i in range(0, 40):  # Lignes 0-39 (robot à ligne 5)
        line = f"{i:3d}|"
        
        for j in range(0, 40):  # TOUTES les colonnes 0-39
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
    print(f"  2. History simplifié: 24 valeurs (4 frames × 6: 3 positions + 3 vitesses)")
    print(f"  3. Grille 2 canaux:  {grid.size} valeurs (57×40×2)")
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
    parser.add_argument("--corridor", type=str, default="corridor_3x100_no_full_obstacles.xml", 
                       help="Fichier XML du corridor (défaut: corridor_3x100_no_full_obstacles.xml)")
    parser.add_argument("--random", action="store_true",
                       help="Génère et utilise un corridor aléatoire au lieu du corridor fixe")
    parser.add_argument("--seed", type=int, help="Seed pour la génération aléatoire (défaut: aléatoire)")
    parser.add_argument("--x", type=float, help="Position X du robot (défaut: aléatoire)")
    parser.add_argument("--y", type=float, help="Position Y du robot (défaut: aléatoire)")
    parser.add_argument("--angle", type=float, help="Angle du robot en DEGRÉS (défaut: aléatoire)")
    parser.add_argument("--test-multiple", action="store_true",
                       help="Teste plusieurs positions prédéfinies")
    parser.add_argument("--full-map", action="store_true",
                       help="Affiche la carte complète du corridor")
    parser.add_argument("--render", action="store_true",
                       help="Ouvre le rendu 3D MuJoCo du corridor")
    args = parser.parse_args()
    
    # Déterminer le corridor à utiliser
    corridor_xml = args.corridor
    temp_file = None
    
    # Si --random OU --seed est fourni, générer un corridor
    if args.random or args.seed is not None:
        temp_file = generate_random_corridor(args.seed)
        if temp_file:
            corridor_xml = temp_file
        else:
            print("⚠️  Utilisation du corridor par défaut à la place")
    
    try:
        if args.render:
            # Convertir angle de degrés en radians si fourni
            angle_rad = np.radians(args.angle) if args.angle is not None else None
            
            # D'ABORD afficher les grilles 2D
            print("📊 VISUALISATION 2D DES GRILLES CNN:")
            print("=" * 50)
            visualize_cnn_input(corridor_xml, args.x, args.y, angle_rad)
            
            # PUIS ouvrir le rendu 3D
            print("\n" + "=" * 50)
            print("🎬 OUVERTURE DU RENDU 3D...")
            input("Appuyez sur Entrée pour ouvrir le rendu 3D...")
            render_corridor_3d(corridor_xml, args.x, args.y, angle_rad)
        elif args.test_multiple:
            test_multiple_positions(corridor_xml)
        elif args.full_map:
            visualize_full_corridor(corridor_xml)
        else:
            # Convertir angle de degrés en radians si fourni
            angle_rad = np.radians(args.angle) if args.angle is not None else None
            visualize_cnn_input(corridor_xml, args.x, args.y, angle_rad)
    
    finally:
        # Nettoyer les fichiers temporaires
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"🗑️  Fichier temporaire supprimé: {temp_file}")
            except:
                pass