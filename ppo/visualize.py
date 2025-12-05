"""
Visualize a trained agent in real-time with matplotlib.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from robot_corridor_env import RobotCorridorEnv
from train_ppo import Agent
import gymnasium as gym


def visualize_agent(model_path):
    """
    Visualize agent with real-time plotting.
    """
    # Create environment
    env = RobotCorridorEnv()
    
    # Create a dummy vectorized env for agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: env])
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"Visualizing agent from {model_path}")
    print("Close the plot window to stop.\n")
    
    # Setup plot
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Robot Corridor Agent Visualization', fontsize=14)
    
    # Initialize data storage
    positions_x = []
    positions_z = []
    rewards_history = []
    actions_history = []
    
    # Reset environment
    obs, info = env.reset()
    done = False
    episode_return = 0
    step = 0
    
    while not done and plt.fignum_exists(fig.number):
        # Get action
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_return += reward
        step += 1
        
        # Store data
        positions_x.append(info['x_position'])
        positions_z.append(info['z_position'])
        rewards_history.append(reward)
        actions_history.append(action.copy())
        
        # Update plots every 10 steps
        if step % 10 == 0:
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Plot 1: X position over time
            axes[0, 0].plot(positions_x, 'b-', linewidth=2)
            axes[0, 0].axhline(y=100, color='g', linestyle='--', label='Goal')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('X Position (m)')
            axes[0, 0].set_title(f'Progress (Current: {positions_x[-1]:.1f}m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Z position (height)
            axes[0, 1].plot(positions_z, 'r-', linewidth=2)
            axes[0, 1].axhline(y=0.1, color='r', linestyle='--', label='Fall threshold')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Z Position (m)')
            axes[0, 1].set_title(f'Height (Current: {positions_z[-1]:.2f}m)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Rewards
            axes[1, 0].plot(rewards_history, 'g-', alpha=0.6)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].set_title(f'Rewards (Total: {episode_return:.1f})')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Actions (wheel torques)
            if len(actions_history) > 0:
                actions_array = np.array(actions_history)
                for i in range(4):
                    axes[1, 1].plot(actions_array[:, i], label=f'Wheel {i+1}', alpha=0.7)
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Torque')
                axes[1, 1].set_title('Wheel Actions')
                axes[1, 1].legend()
                axes[1, 1].set_ylim([-1.1, 1.1])
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
    
    # Final statistics
    print(f"\nEpisode finished!")
    print(f"  Steps: {step}")
    print(f"  Total Return: {episode_return:.2f}")
    print(f"  Final X Position: {positions_x[-1]:.2f}m")
    print(f"  Termination: {info.get('termination_reason', 'truncated')}")
    
    # Keep plot open
    plt.ioff()
    plt.show()
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    
    args = parser.parse_args()
    
    visualize_agent(args.model_path)
