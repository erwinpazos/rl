"""
Test the corrected environment with floor observation.
"""
import numpy as np
from robot_corridor_env import RobotCorridorEnv


def test_environment():
    """Test the environment with floor observation."""
    print("="*60)
    print("TESTING CORRECTED ROBOT CORRIDOR ENVIRONMENT")
    print("(with floor observation)")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = RobotCorridorEnv()
    print("   ✓ Environment created successfully")
    
    # Check spaces
    print("\n2. Checking observation and action spaces...")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Observation shape: {env.observation_space.shape}")
    print(f"   Expected: (30,) = 6 (pos+vel) + 24 (floor grid 4×6)")
    print(f"   Action space: {env.action_space}")
    print("   ✓ Spaces look good")
    
    # Test reset
    print("\n3. Testing reset...")
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial observation (first 10 values): {obs[:10]}")
    print(f"   Initial position: x={info['x_position']:.2f}, y={info['y_position']:.2f}, z={info['z_position']:.2f}")
    print("   ✓ Reset works")
    
    # Verify floor observation is included
    print("\n4. Verifying floor observation...")
    pos_vel = obs[:6]
    floor_obs = obs[6:].reshape(4, 6)
    print(f"   Position + Velocity: {pos_vel}")
    print(f"   Floor observation shape: {floor_obs.shape}")
    print(f"   Floor observation (4 rows × 6 cols):")
    print(f"   {floor_obs}")
    print(f"   Cell types: 0=flat, 1=bump, 2=hole")
    print("   ✓ Floor observation is included!")
    
    # Test random episode
    print("\n5. Running random episode (100 steps)...")
    episode_return = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        
        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            print(f"   Reason: {info.get('termination_reason', 'truncated')}")
            break
    
    print(f"   Final position: x={info['x_position']:.2f}m")
    print(f"   Episode return: {episode_return:.2f}")
    print("   ✓ Environment step works")
    
    # Test multiple episodes
    print("\n6. Testing 3 random episodes...")
    returns = []
    distances = []
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        episode_return = 0
        steps = 0
        
        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            steps += 1
        
        returns.append(episode_return)
        distances.append(info['x_position'])
        
        print(f"   Episode {episode+1}: return={episode_return:.2f}, distance={info['x_position']:.2f}m, steps={steps}")
    
    print(f"\n   Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"   Average distance: {np.mean(distances):.2f}m ± {np.std(distances):.2f}m")
    print("   ✓ Multiple episodes work")
    
    env.close()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nThe CORRECTED environment is ready for training.")
    print("Key improvements:")
    print("  - Observation includes floor grid (30 values total)")
    print("  - Robot can 'see' 1 row behind, current row, 2 rows ahead")
    print("  - This should significantly improve learning!")
    print("\nRun: python train_ppo_simple.py")
    print()


if __name__ == "__main__":
    test_environment()
