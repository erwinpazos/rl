#!/usr/bin/env python3
"""
Test du contrôle par volant dans l'environnement.
"""
import numpy as np
from corridor_env import CorridorEnv

def test_steering_control():
    """Test basique du contrôle par volant."""
    print("=== Test du contrôle par volant ===")
    
    # Créer environnement
    env = CorridorEnv(max_steps=100)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Robot position: x={info['x']:.2f}, y={info['y']:.2f}, z={info['z']:.2f}")
    
    # Test différentes actions
    test_actions = [
        [0.0, 0.5],   # Tout droit, vitesse moyenne
        [1.0, 0.3],   # Virage à gauche max, vitesse faible
        [-1.0, 0.3],  # Virage à droite max, vitesse faible
        [0.0, -0.5],  # Tout droit, marche arrière
        [0.5, 0.8],   # Virage à gauche modéré, vitesse élevée
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest {i+1}: steering={action[0]:.1f} ({action[0]*30:.1f}°), speed={action[1]:.1f} ({action[1]*2:.1f} m/s)")
        
        # Faire quelques steps
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
            
            if step == 0:  # Afficher seulement le premier step de chaque action
                print(f"  → Position: x={info['x']:.2f}, y={info['y']:.2f}, z={info['z']:.2f}")
                print(f"  → Reward: {reward:.3f}")
            
            if terminated or truncated:
                print(f"  → Episode terminé: terminated={terminated}, truncated={truncated}")
                break
    
    env.close()
    print("\n✅ Test terminé avec succès !")

if __name__ == "__main__":
    test_steering_control()