# Robot Corridor PPO Training

Train a 4-wheel robot to navigate a 100m corridor using PPO (Proximal Policy Optimization).

## 🚀 Quick Start

```bash
# 1. Check installation
python check_installation.py

# 2. Test environment
python test_env.py

# 3. Start training
python quick_train.py

# 4. Monitor progress
tensorboard --logdir runs

# 5. Test trained agent
python test_trained_agent.py --model-path runs/LATEST/ppo_robot_corridor.pth
```

## 📁 Files

| File | Description |
|------|-------------|
| `robot_corridor_env.py` | Gymnasium environment for the corridor |
| `train_ppo.py` | Main PPO training script |
| `test_trained_agent.py` | Evaluate trained models |
| `visualize.py` | Real-time visualization |
| `test_env.py` | Environment verification |
| `quick_train.py` | One-command training |
| `check_installation.py` | Verify dependencies |

## 📚 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Quick start guide
- **[README_TRAINING.md](README_TRAINING.md)**: Complete training guide
- **[SUMMARY.md](SUMMARY.md)**: Implementation details

## 🎯 Goal

Train an agent to navigate a 4-wheel robot from x=0m to x=100m through a corridor with obstacles.

## 🏆 Success Criteria

- **Reach the goal**: x ≥ 100m → +100 reward
- **Avoid falling**: z > 0.1m
- **Move forward**: Positive progress → positive reward

## 📊 Expected Results

| Training Steps | Avg Distance | Success Rate |
|----------------|--------------|--------------|
| 500k | 10-20m | 0-5% |
| 1M | 20-40m | 5-15% |
| 2M | 40-60m | 15-30% |
| 5M | 60-90m | 30-60% |

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install gymnasium torch numpy mujoco tensorboard tyro matplotlib
```

## 💡 Tips

1. **Be patient**: Good results need 1-2M timesteps
2. **Use parallel envs**: `--num-envs 8` for faster training
3. **Monitor with TensorBoard**: Track progress in real-time
4. **Test regularly**: Check performance every 500k steps

## 🐛 Troubleshooting

**Agent doesn't learn?**
- Increase training time: `--total-timesteps 5000000`
- More exploration: `--ent-coef 0.02`

**Training too slow?**
- More parallel envs: `--num-envs 8`
- Use GPU (automatic if available)

**Agent falls immediately?**
- Normal at start! Wait ~100k steps for balance learning

## 📖 Learn More

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed instructions.

## 🎓 Based On

- Notebook Questions 10-14 (RL API concepts)
- CleanRL PPO implementation
- Gymnasium environment API

---

**Ready to train?** Run `python check_installation.py` to get started! 🚀
