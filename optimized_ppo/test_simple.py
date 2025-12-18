"""
Test simple de l'environnement simplifié.
"""
import numpy as np
from corridor_env import CorridorEnv

def test_env():
    """Test basique de l'environnement avec historique."""
    print("🧪 TEST ENVIRONNEMENT AVEC HISTORIQUE")
    print("=" * 50)
    
    # Créer environnement
    env = CorridorEnv(corridor_xml="corridor_100.xml", max_steps=100)
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Taille grille: {env.grid_rows}×{env.grid_cols} = {env.grid_rows * env.grid_cols}")
    print(f"Historique: {env.history_length} positions × 3 coords = {env.history_length * 3}")
    print(f"Taille cellule: {env.cell_size}m")
    print(f"Vision: {env.vision_length}m × {env.vision_width}m")
    print(f"Stabilisation: {env.stabilization_steps} steps")
    
    # Reset
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Position initiale: x={info['x']:.2f}, y={info['y']:.2f}")
    
    # Décoder observation AVEC HISTORIQUE
    robot_state = obs[:6]
    bbox_corners = obs[6:14] 
    position_history = obs[14:29].reshape(5, 3)
    grid = obs[29:].reshape(env.grid_rows, env.grid_cols)
    
    print(f"Robot state: pos=({robot_state[0]:.2f}, {robot_state[1]:.2f}, {robot_state[2]:.2f})")
    print(f"Robot vel: vel=({robot_state[3]:.2f}, {robot_state[4]:.2f}, {robot_state[5]:.2f})")
    
    # Afficher bounding box
    corner_names = ['AV-G', 'AV-D', 'AR-G', 'AR-D']
    print("Bounding box corners (row, col):")
    for i, name in enumerate(corner_names):
        row, col = bbox_corners[i*2], bbox_corners[i*2+1]
        print(f"  {name}: ({row:.0f}, {col:.0f})")
    
    # Afficher échantillon de grille autour du robot
    print(f"\nGrille autour du robot (centre 20×20):")
    robot_row, robot_col = 40, 40  # Position théorique du robot
    
    for i in range(robot_row-10, robot_row+10):
        line = f"  {(i-robot_row)*0.05:+.2f}m: "
        for j in range(robot_col-10, robot_col+10):
            if 0 <= i < env.grid_rows and 0 <= j < env.grid_cols:
                val = grid[i, j]
                if val == 0.0:
                    line += '▓'  # Sol
                elif val == 0.5:
                    line += '△'  # Rampe  
                elif val == 0.75:
                    line += 'R'  # Robot
                else:
                    line += '░'  # Trou
            else:
                line += '?'  # Hors limites
        print(line)
    
    print("(▓=sol, △=rampe, R=robot, ░=trou)")
    
    # Test quelques steps
    print(f"\n🎮 Test de quelques actions...")
    for step in range(5):
        action = np.array([0.5, 0.5, 0.5, 0.5])  # Avancer
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"Step {step+1}: x={info['x']:.2f}, reward={reward:.3f}, done={term or trunc}")
        
        if term or trunc:
            print(f"  Terminé: {info.get('reason', 'unknown')}")
            break
    
    print("\n✅ Test terminé avec succès!")

if __name__ == "__main__":
    test_env()