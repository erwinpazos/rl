# Training PPO Agent for Robot Corridor

This directory contains the implementation for training a PPO (Proximal Policy Optimization) agent to navigate the 4-wheel robot through the corridor.

## Files

- **`robot_corridor_env.py`**: Gymnasium environment wrapper for the corridor
- **`train_ppo.py`**: PPO training script
- **`test_trained_agent.py`**: Script to test trained agents
- **`ppo_continuous_action.py`**: Original CleanRL PPO implementation (reference)

## Installation

```bash
pip install gymnasium torch numpy mujoco tensorboard tyro
```

## Quick Start

### 1. Train the Agent

Basic training (2M timesteps, ~30 minutes on CPU):
```bash
cd ppo
python train_ppo.py
```

With custom parameters:
```bash
python train_ppo.py --total-timesteps 5000000 --num-envs 8 --learning-rate 3e-4
```

### 2. Monitor Training

Open TensorBoard to visualize training progress:
```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

### 3. Test the Trained Agent

```bash
python test_trained_agent.py --model-path runs/RobotCorridor-v0__ppo_robot_corridor__1__TIMESTAMP/ppo_robot_corridor.pth --episodes 10
```

With rendering:
```bash
python test_trained_agent.py --model-path MODEL_PATH --episodes 5 --render
```

## Environment Details

### Observation Space (6 dimensions)
- `x, y, z`: Robot position in 3D space
- `vx, vy, vz`: Robot velocity

### Action Space (4 dimensions)
- 4 continuous values in [-1, 1] representing torque for each wheel

### Rewards
- **Progress**: +10 × forward_distance (encourages moving forward)
- **Time penalty**: -0.01 per step (encourages efficiency)
- **Success**: +100 for reaching x ≥ 100m
- **Failure**: -100 for falling (z < 0.1m)
- **Failure**: -50 for going backward (x < -1m)

### Episode Termination
- **Success**: Robot reaches x ≥ 100m
- **Failure**: Robot falls (z < 0.1m) or goes too far backward
- **Truncation**: Maximum 1000 steps reached

## Training Parameters

Default hyperparameters (can be modified via command line):

```python
total_timesteps = 2000000      # Total training steps
learning_rate = 3e-4           # Learning rate
num_envs = 4                   # Parallel environments
num_steps = 2048               # Steps per rollout
gamma = 0.99                   # Discount factor
gae_lambda = 0.95              # GAE lambda
clip_coef = 0.2                # PPO clip coefficient
ent_coef = 0.01                # Entropy coefficient
```

## Expected Results

After training for 2M timesteps:
- **Average Return**: 50-200 (depends on how far the robot gets)
- **Success Rate**: 10-50% (reaching the goal)
- **Average Distance**: 20-60m

With more training (5M+ timesteps):
- **Success Rate**: 50-80%
- **Average Distance**: 60-90m

## Tips for Better Performance

1. **Increase training time**: Use `--total-timesteps 5000000` or more
2. **More parallel environments**: Use `--num-envs 8` or `--num-envs 16`
3. **Adjust entropy**: Try `--ent-coef 0.02` for more exploration
4. **Learning rate**: Try `--learning-rate 1e-4` for more stable learning

## Troubleshooting

### Agent doesn't learn
- Check that the environment is working: run a few random episodes
- Increase `--ent-coef` for more exploration
- Reduce `--learning-rate` for more stable updates

### Training is slow
- Increase `--num-envs` to parallelize
- Use GPU if available (automatic with `--cuda`)
- Reduce `--num-steps` (but may hurt performance)

### Agent falls immediately
- This is normal at the beginning of training
- The agent needs to learn balance first (~100k steps)
- After that, it should start making progress

## Advanced Usage

### Save videos during training
```bash
python train_ppo.py --capture-video
```

### Track with Weights & Biases
```bash
python train_ppo.py --track --wandb-project-name my-project
```

### Custom environment parameters
Modify `robot_corridor_env.py` to change:
- Corridor length
- Reward scaling
- Observation space
- Maximum episode length

## Architecture

The PPO agent uses:
- **Actor (Policy)**: 2-layer MLP (64 units each) with Tanh activation
- **Critic (Value)**: 2-layer MLP (64 units each) with Tanh activation
- **Action distribution**: Gaussian with learnable standard deviation

## References

- [CleanRL PPO Implementation](https://docs.cleanrl.dev/rl-algorithms/ppo/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
