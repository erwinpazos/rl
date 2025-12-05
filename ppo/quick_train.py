"""
Quick training script with sensible defaults for the corridor task.
"""
import subprocess
import sys

def main():
    """Run training with optimized parameters for the corridor."""
    
    print("="*60)
    print("ROBOT CORRIDOR - PPO TRAINING")
    print("="*60)
    print("\nThis will train a PPO agent to navigate the corridor.")
    print("Training parameters:")
    print("  - Total timesteps: 2,000,000")
    print("  - Parallel environments: 4")
    print("  - Learning rate: 3e-4")
    print("  - Estimated time: 20-40 minutes (CPU)")
    print("\nProgress will be saved in the 'runs/' directory.")
    print("Monitor with: tensorboard --logdir runs")
    print("="*60)
    print()
    
    # Training command
    cmd = [
        sys.executable, "train_ppo.py",
        "--total-timesteps", "2000000",
        "--num-envs", "4",
        "--learning-rate", "3e-4",
        "--save-model",
        "--ent-coef", "0.01",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nTo test your agent:")
        print("  python test_trained_agent.py --model-path runs/LATEST_RUN/ppo_robot_corridor.pth")
        print("\nTo visualize:")
        print("  python visualize.py --model-path runs/LATEST_RUN/ppo_robot_corridor.pth")
        print()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial progress has been saved.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
