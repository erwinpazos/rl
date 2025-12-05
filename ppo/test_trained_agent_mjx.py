"""
Test the trained MJX agent with visualization.
"""
import torch
import torch.nn as nn
import numpy as np
import mujoco
from mujoco import viewer
import time


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean  # Deterministic for testing


def test_agent(model_path, num_episodes=5):
    """Test the trained agent with visualization."""
    
    print("="*60)
    print("TESTING TRAINED MJX AGENT")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    obs_size = 6
    action_size = 4
    
    agent = Agent(obs_size, action_size).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print("✓ Model loaded successfully")
    
    # Load MuJoCo model (CPU for visualization)
    print("\nLoading MuJoCo model for visualization...")
    model = mujoco.MjModel.from_xml_path("robot_simple_mjx.xml")
    data = mujoco.MjData(model)
    
    print("✓ MuJoCo model loaded")
    print(f"\nRunning {num_episodes} test episodes with visualization...")
    print("Press ESC to quit\n")
    
    episode_returns = []
    episode_lengths = []
    
    with viewer.launch_passive(model, data) as v:
        for episode in range(num_episodes):
            # Reset
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            
            episode_return = 0
            episode_length = 0
            done = False
            previous_x = data.qpos[0]
            
            print(f"Episode {episode + 1}/{num_episodes}")
            
            while not done and v.is_running():
                # Get observation
                pos = data.qpos[:3]
                vel = data.qvel[:3]
                obs = np.concatenate([pos, vel]).astype(np.float32)
                
                # Get action from agent
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = agent.get_action(obs_tensor)
                    action = action.cpu().numpy()[0]
                
                # Apply action
                data.ctrl[:] = action
                
                # Step simulation (5 substeps like training)
                for _ in range(5):
                    mujoco.mj_step(model, data)
                
                # Compute reward
                robot_x = data.qpos[0]
                robot_z = data.qpos[2]
                
                if robot_x >= 100.0:
                    reward = 100.0
                    done = True
                    print(f"  ✓ SUCCESS! Reached goal at x={robot_x:.2f}m")
                elif robot_z < 0.1 and robot_x > 0:
                    reward = -100.0
                    done = True
                    print(f"  ✗ FAILED: Fell at x={robot_x:.2f}m")
                elif robot_x < -1.0:
                    reward = -50.0
                    done = True
                    print(f"  ✗ FAILED: Went backward")
                else:
                    delta_x = robot_x - previous_x
                    previous_x = robot_x
                    reward = delta_x * 10.0 - 0.01
                
                episode_return += reward
                episode_length += 1
                
                # Check timeout
                if episode_length >= 1000:
                    done = True
                    print(f"  ⏱ TIMEOUT at x={robot_x:.2f}m")
                
                # Render
                v.sync()
                time.sleep(0.01)  # Slow down for visualization
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            print(f"  Return: {episode_return:.2f}, Length: {episode_length} steps")
            print(f"  Final position: x={data.qpos[0]:.2f}m\n")
            
            if not v.is_running():
                break
            
            # Pause between episodes
            time.sleep(1.0)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Episodes completed: {len(episode_returns)}")
    print(f"Average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} steps")
    print(f"Best return: {max(episode_returns):.2f}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Find latest model
        import glob
        models = glob.glob("models/ppo_robot_mjx_*/ppo_robot_mjx.pth")
        if not models:
            print("No trained models found!")
            print("Train a model first with: python train_ppo_simple_mjx.py")
            sys.exit(1)
        model_path = sorted(models)[-1]
    
    test_agent(model_path)
