"""
Visualiser EXACTEMENT les DEUX COUCHES que voit le modèle.
"""
from corridor_env import CorridorEnv
import numpy as np
import argparse

def visualize_model_two_layers(corridor_xml="corridor_100.xml"):
    """Visualiser les DEUX COUCHES exactes du modèle avec robot aléatoire."""
    env = CorridorEnv(corridor_xml=corridor_xml)
    
    # Position ALÉATOIRE du robot dans le couloir
    robot_x = np.random.uniform(5.0, 95.0)  # X aléatoire
    robot_y = np.random.uniform(-1.0, 1.0)  # Y aléatoire dans couloir
    robot_angle = np.random.uniform(-np.pi/4, np.pi/4)  # Angle aléatoire
    
    # Positionner le robot
    env.data.qpos[0] = robot_x
    env.data.qpos[1] = robot_y
    env.data.qpos[2] = 0.45
    env.data.qpos[3] = np.cos(robot_angle / 2)  # Quaternion
    env.data.qpos[4] = 0
    env.data.qpos[5] = 0
    env.data.qpos[6] = np.sin(robot_angle / 2)
    
    # Obtenir observation EXACTE du modèle
    obs = env._get_obs()
    
    # Décoder l'observation
    robot_state = obs[:6]  # pos(3) + vel(3)
    wheel_positions = obs[6:14]  # 4 roues × (row, col)
    env_grid = obs[14:14+2048].reshape(32, 64)  # COUCHE 1: Environnement
    robot_grid = obs[14+2048:].reshape(32, 64)  # COUCHE 2: Robot
    
    print("="*80)
    print("VISION EXACTE DU MODÈLE - DEUX COUCHES SÉPARÉES")
    print("="*80)
    print(f"Robot position: x={robot_state[0]:.3f}m, y={robot_state[1]:.3f}m, z={robot_state[2]:.3f}m")
    print(f"Robot velocity: vx={robot_state[3]:.3f}, vy={robot_state[4]:.3f}, vz={robot_state[5]:.3f}")
    print(f"Robot angle: {np.degrees(robot_angle):.1f}°")
    print()
    
    # Positions des roues
    wheel_names = ['FL', 'FR', 'RL', 'RR']
    print("POSITIONS DES ROUES (dans la grille):")
    for i, name in enumerate(wheel_names):
        row = int(wheel_positions[i*2])
        col = int(wheel_positions[i*2+1])
        print(f"  {name}: row={row:3d}, col={col:2d}")
    print()
    
    print("OBSERVATION TOTALE:")
    print(f"  Robot state: 6 valeurs")
    print(f"  Wheel positions: 8 valeurs")
    print(f"  COUCHE 1 - Environnement: 2048 valeurs (32×64)")
    print(f"  COUCHE 2 - Robot: 2048 valeurs (32×64)")
    print(f"  TOTAL: 4110 valeurs pour le réseau de neurones")
    print()
    
    # Afficher les DEUX COUCHES côte à côte
    print("="*80)
    print("COUCHE 1: ENVIRONNEMENT (32×64)        |  COUCHE 2: ROBOT (32×64)")
    print("▓=sol, △=rampe, ░=trou                  |  ░=vide, R=robot")
    print("="*80)
    
    env_symbols = {0: '▓', 1: '△', 2: '░'}  # Environnement
    robot_symbols = {0: '░', 1: 'R'}  # Robot (0=vide, 1=corps)
    
    # Afficher centre 20×32 de chaque couche
    for i in range(6, 26):  # Lignes 6-25 (centre 20 lignes)
        # COUCHE 1: Environnement
        env_line = ""
        for j in range(16, 48):  # Colonnes 16-47 (centre 32 colonnes)
            val = int(env_grid[i, j])
            env_line += env_symbols.get(val, '?')
        
        # COUCHE 2: Robot
        robot_line = ""
        for j in range(16, 48):  # Colonnes 16-47 (centre 32 colonnes)
            val = int(robot_grid[i, j])
            robot_line += robot_symbols.get(val, '?')
        
        # Distance relative au robot
        relative_dist = (i - 16) * env.cell_size
        print(f"L{i:2d} ({relative_dist:+.3f}m): {env_line} | {robot_line}")
    
    print()
    print("="*80)
    print("EXPLICATION:")
    print("- COUCHE 1 montre l'environnement: sol, rampes, trous")
    print("- COUCHE 2 montre le robot: son corps avec orientation")
    print("- Le modèle voit les DEUX en même temps comme des calques")
    print("- Vision FIXE: -2m à +2m en Y (toute largeur couloir)")
    print("- Vision centrée: -1m à +1m en X (autour du robot)")
    print("="*80)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model's two-layer vision")
    parser.add_argument("--corridor", type=str, default="corridor_100.xml", 
                       help="Corridor XML file (default: corridor_100.xml)")
    args = parser.parse_args()
    
    visualize_model_two_layers(corridor_xml=args.corridor)