# Getting Started with PPO Training

## Quick Start (3 steps)

### 1. Test the environment
```bash
cd ppo
python test_env.py
```

This will verify that everything is working correctly.

### 2. Start training
```bash
python quick_train.py
```

Or with custom parameters:
```bash
python train_ppo.py --total-timesteps 5000000 --num-envs 8
```

### 3. Test the trained agent
```bash
python test_trained_agent.py --model-path runs/RobotCorridor-v0__ppo_robot_corridor__1__TIMESTAMP/ppo_robot_corridor.pth --episodes 10
```

## What to Expect

### During Training
- **First 100k steps**: Agent learns to balance and not fall immediately
- **100k-500k steps**: Agent starts moving forward consistently
- **500k-1M steps**: Agent learns to avoid obstacles and navigate better
- **1M-2M steps**: Performance stabilizes, success rate improves

### Training Output
```
global_step=8192, episodic_return=12.45, episodic_length=234
global_step=16384, episodic_return=25.67, episodic_length=456
...
Iteration 244/244, SPS: 1234
```

- `episodic_return`: Total reward for the episode (higher is better)
- `episodic_length`: Number of steps before termination
- `SPS`: Steps per second (training speed)

### Success Metrics
- **Good progress**: episodic_return > 50, distance > 20m
- **Great progress**: episodic_return > 100, distance > 50m
- **Success**: episodic_return > 150, distance = 100m (reached goal!)

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs
```

Open http://localhost:6006 to see:
- Episode returns over time
- Policy and value losses
- Learning rate schedule
- Final positions reached

### Key Metrics to Watch
1. **charts/episodic_return**: Should increase over time
2. **charts/final_x_position**: Should increase (robot going further)
3. **losses/policy_loss**: Should decrease and stabilize
4. **losses/value_loss**: Should decrease

## Troubleshooting

### "ModuleNotFoundError: No module named 'robot_corridor_env'"
Make sure you're in the `ppo/` directory when running scripts.

### Agent doesn't learn (returns stay low)
- Increase training time: `--total-timesteps 5000000`
- More exploration: `--ent-coef 0.02`
- More parallel envs: `--num-envs 8`

### Training is too slow
- Use more parallel environments: `--num-envs 8` or `--num-envs 16`
- Use GPU if available (automatic)
- Reduce `--num-steps 1024` (but may hurt performance)

### Agent falls immediately
- This is normal at the start!
- Wait for ~100k steps for the agent to learn balance
- Check that the environment works: `python test_env.py`

## Next Steps

After training:

1. **Test performance**:
   ```bash
   python test_trained_agent.py --model-path MODEL_PATH --episodes 20
   ```

2. **Visualize behavior**:
   ```bash
   python visualize.py --model-path MODEL_PATH
   ```

3. **Train longer for better results**:
   ```bash
   python train_ppo.py --total-timesteps 10000000
   ```

4. **Experiment with hyperparameters**:
   - Learning rate: `--learning-rate 1e-4`
   - Entropy coefficient: `--ent-coef 0.02`
   - Clip coefficient: `--clip-coef 0.3`

## Files Overview

- `robot_corridor_env.py`: Gymnasium environment
- `train_ppo.py`: Main training script
- `test_trained_agent.py`: Evaluate trained models
- `visualize.py`: Real-time visualization
- `test_env.py`: Verify environment works
- `quick_train.py`: One-command training

## Tips for Best Results

1. **Be patient**: Good results need 1-2M timesteps minimum
2. **Use parallel environments**: `--num-envs 8` speeds up training
3. **Monitor progress**: Use TensorBoard to track learning
4. **Save checkpoints**: Models are auto-saved in `runs/`
5. **Test regularly**: Check progress every 500k steps

## Expected Timeline

- **500k steps**: ~10 minutes (4 parallel envs, CPU)
- **1M steps**: ~20 minutes
- **2M steps**: ~40 minutes
- **5M steps**: ~1.5 hours

With GPU: 2-3x faster

Good luck with training! 🚀
