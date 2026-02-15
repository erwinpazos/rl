# PPO No Steer - 4 Independent Wheels Control

Version with direct control of 4 independent wheels (action space: 4 dimensions).

**Location**: `mujoco/workspace/ppo_no_steer/`

## Table of Contents

- [Overview](#overview)
- [Available Scripts](#available-scripts)
- [Training Pipeline](#training-pipeline)
- [Network Architecture](#network-architecture)
- [Configuration](#configuration)

---

## Overview

This version directly controls the 4 robot wheels:
- **Action space**: `Box(-1.0, 1.0, (4,))` 
- **Actions**: `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]`
- Each value represents the corresponding wheel speed
- More freedom but harder to learn

**Prerequisites**: Docker environment launched (see [main README](../../../README.md))

---

## Available Scripts

### 1. train_ppo.py - Training

Main training script with PPO.

**Usage:**
```bash
# In Docker environment (http://localhost:6080)
cd ~/workspace/ppo_no_steer
python3 train_ppo.py [OPTIONS]
```

**Arguments:**
- `--config PATH`: YAML configuration file (default: `config.yaml`)
- `--timesteps N`: Override total timesteps (default: 8,000,000)
- `--num-envs N`: Override number of parallel environments (default: 30)
- `--num-steps N`: Override steps per rollout (default: 1024)
- `--lr FLOAT`: Override learning rate (default: 0.0004)
- `--seed N`: Override seed for reproducibility (default: 1)
- `--fresh-start`: Force fresh start (ignore existing checkpoints)
- `--rollback`: Enable automatic rollback if performance regresses

**Examples:**
```bash
# In Docker environment
cd ~/workspace/ppo_no_steer

# Standard training with rollback
python3 train_ppo.py --rollback

# Fresh start with 16 environments
python3 train_ppo.py --fresh-start --num-envs 16

# Override learning rate and seed
python3 train_ppo.py --lr 0.0003 --seed 42
```

**Features:**
- Parallel training with AsyncVectorEnv
- Automatic saving every N iterations
- Progressive curriculum learning (4 phases)
- Automatic rollback on performance regression (with `--rollback`)
- Metric plotting
- Detailed logs per iteration and episode
- Automatic resume from latest checkpoint

**Generated files:**
- `models/ppo_corridor_{iteration}.pth`: Saved checkpoints
- `models/training_metrics.csv`: Training metrics
- `models/training_curves_{iteration}.png`: Iteration plots
- `models/training_curves.png`: Final plot
- `models/episodes_log.txt`: Detailed episode log
- `models/iteration_summary.json`: Latest iteration summary

---

### 2. test_ppo.py - Model Testing

Tests a trained model on N episodes.

**Usage:**
```bash
# In Docker environment
cd ~/workspace/ppo_no_steer
python3 test_ppo.py [OPTIONS]
```

**Arguments:**
- `--model PATH`: Path to checkpoint (default: latest checkpoint found)
- `--config PATH`: YAML configuration file (default: `config.yaml`)
- `--num-episodes N`: Number of episodes to test (default: 10)
- `--render`: Display MuJoCo 3D rendering
- `--show-vision`: Display CNN vision in real-time
- `--corridor PATH`: Use specific corridor (XML)
- `--bump-ratio FLOAT`: Bump ratio (0.0 to 1.0, default: from config)

**Examples:**
```bash
# In Docker environment
cd ~/workspace/ppo_no_steer

# Simple test (10 episodes, no rendering)
python3 test_ppo.py

# Test with 3D rendering and CNN vision
python3 test_ppo.py --render --show-vision --num-episodes 5

# Test on specific corridor
python3 test_ppo.py --render --corridor corridor_yguel.xml

# Test with 100% bumps
python3 test_ppo.py --render --bump-ratio 1.0 --num-episodes 3

# Test specific checkpoint
python3 test_ppo.py --model models/ppo_corridor_50.pth --render
```

**Display:**
- Statistics per episode (reward, distance, termination reason)
- Final summary (mean ± std, best distance)
- CNN vision in real-time (if `--show-vision`)
- MuJoCo 3D rendering (if `--render`)

---

### 3. visualize_corridor_map.py - CNN Visualization

Visualizes exactly what the CNN receives as input (2-channel grid).

**Usage:**
```bash
python3 visualize_corridor_map.py [OPTIONS]
```

**Arguments:**
- `--corridor PATH`: Corridor XML file (default: random generation)
- `--x FLOAT`: Robot X position (default: random)
- `--y FLOAT`: Robot Y position (default: random)
- `--angle FLOAT`: Robot angle in degrees (default: random)
- `--seed N`: Seed for random generation
- `--bump-ratio FLOAT`: Bump ratio (0.0 to 1.0, default: 0.5)
- `--render`: Open MuJoCo 3D rendering after visualization

**Examples:**
```bash
# Visualization with random corridor
python3 visualize_corridor_map.py

# Visualization of specific corridor
python3 visualize_corridor_map.py --corridor corridor_yguel.xml

# Specific robot position
python3 visualize_corridor_map.py --x 50.0 --y 0.5 --angle 15

# With 3D rendering
python3 visualize_corridor_map.py --render

# Random corridor with fixed seed
python3 visualize_corridor_map.py --seed 42 --bump-ratio 0.8
```

**Display:**
- Robot state (position, velocity, angle)
- Simplified history (8 frames × 6 values)
- Channel 0: Obstacles (bumps + lateral walls)
- Channel 1: Holes (holes + exterior)
- Combined view (red=obstacle, blue=hole, white=ground)
- Channel statistics

---

### 4. corridor_env.py - Gymnasium Environment

Custom environment for robot in corridor.

**Features:**
- Gymnasium compatible (gym.Env)
- Observation: robot state + history + CNN grid (2 channels)
- Action: 4 wheel speeds `[-1, 1]`
- Rewards: progress, collision, success/failure
- Termination: fell, flipped, no_progress, success

---

### 5. corridor_generator_similar.py - Corridor Generator

Generates random corridors with holes and bumps.

**Features:**
- Procedural generation with seed
- Bump ratio control (bump_ratio)
- Always 100% holes + X% bumps
- Save as MuJoCo XML

---

## Complete Training Pipeline

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ load_config()    │ ← config.yaml
                    │ Load YAML        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Create Agent     │ ← CNN+MLP architecture
                    │ + Optimizer      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ find_latest_     │
                    │ checkpoint()     │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
            ┌──────────────┐   ┌──────────────┐
            │ Checkpoint   │   │ No           │
            │ found        │   │ checkpoint   │
            └──────┬───────┘   └──────┬───────┘
                   │                  │
                   ▼                  ▼
         ┌──────────────────┐  ┌──────────────┐
         │ load_checkpoint()│  │ Start        │
         │ Restore state    │  │ from scratch │
         └──────┬───────────┘  └──────┬───────┘
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
         ┌─────────────────────────────────┐
         │ Create parallel environments    │
         │ AsyncVectorEnv (30 envs)        │
         └─────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                               │
│                    (260 iterations × 30,720 steps)               │
└──────────────────────────────────────────────────────────────────┘

                       │
                       ▼
         ┌─────────────────────────────┐
         │ START ITERATION             │
         │ iteration_tracker.reset()   │ ← Reset iteration stats
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ get_curriculum_state()      │ ← Determine current phase
         │ - Phase (1-4)               │
         │ - bump_ratio (0.5 → 1.0)    │
         │ - max_steps (curriculum)    │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ COLLECT ROLLOUT             │
         │ (30 envs × 1024 steps)      │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ For each step:              │
         │ 1. agent.get_action(obs)    │ ← Forward pass
         │ 2. envs.step(actions)       │ ← Parallel simulation
         │ 3. Store (obs, action,      │
         │    reward, done, value)     │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ If episode done:            │
         │ 1. iteration_tracker.add()  │ ← Iteration stats
         │ 2. save_episode_to_temp()   │ ← Log episode
         │ 3. Increment episode_num    │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ COMPUTE ADVANTAGES          │
         │ compute_gae()               │ ← GAE (λ=0.98, γ=0.995)
         │ - Advantages                │
         │ - Returns                   │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ PPO UPDATE                  │
         │ (10 epochs × 32 minibatches)│
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ For each epoch:             │
         │ 1. Shuffle indices          │
         │ 2. For each minibatch:      │
         │    - Forward pass           │
         │    - Compute losses         │
         │    - Backward + optimize    │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ BATCH METRICS               │
         │ compute_batch_metrics()     │
         │ - mean_return               │
         │ - mean_distance             │
         │ - mean_survival             │
         │ - success_rate              │
         │ - termination_counts        │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ save_temp_batch_to_csv()    │ ← Save to temp CSV
         │ + curriculum fields         │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ ITERATION LOGS DISPLAY      │
         │ - Current iteration stats   │
         │ - Curriculum phase          │
         │ - Termination counts        │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ iteration % save_interval?  │
         └─────────────┬───────────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
         ┌─────────┐      ┌─────────┐
         │   YES   │      │   NO    │
         └────┬────┘      └────┬────┘
              │                │
              ▼                │
┌──────────────────────────┐  │
│   SAVE CHECK             │  │
└──────────────────────────┘  │
              │                │
              ▼                │
┌──────────────────────────┐  │
│ get_mean_distance_       │  │
│ from_temp()              │  │ ← Average distance this iteration
└──────────┬───────────────┘  │
           │                  │
           ▼                  │
┌──────────────────────────┐  │
│ load_last_iteration_     │  │
│ summary()                │  │ ← Distance from last save
└──────────┬───────────────┘  │
           │                  │
           ▼                  │
┌──────────────────────────┐  │
│ current >= last?         │  │
└──────────┬───────────────┘  │
           │                  │
    ┌──────┴──────┐           │
    │             │           │
    ▼             ▼           │
┌────────┐   ┌────────┐      │
│ ACCEPT │   │ REJECT │      │
└───┬────┘   └───┬────┘      │
    │            │           │
    │            ▼           │
    │   ┌────────────────┐  │
    │   │ --rollback?    │  │
    │   └────┬───────────┘  │
    │        │              │
    │   ┌────┴────┐         │
    │   │         │         │
    │   ▼         ▼         │
    │ ┌───┐   ┌───────┐    │
    │ │YES│   │  NO   │    │
    │ └─┬─┘   └───┬───┘    │
    │   │         │         │
    │   ▼         ▼         │
    │ ┌──────┐ ┌──────┐    │
    │ │Load  │ │Skip  │    │
    │ │last  │ │save  │    │
    │ │ckpt  │ │      │    │
    │ └──┬───┘ └──┬───┘    │
    │    │        │         │
    │    └────┬───┘         │
    │         │             │
    ▼         ▼             │
┌──────────────────────┐   │
│ SAVE                 │   │
└──────────────────────┘   │
    │                      │
    ▼                      │
┌──────────────────────┐   │
│ save_iteration_      │   │
│ summary()            │   │ ← JSON with distance
└──────┬───────────────┘   │
       │                   │
       ▼                   │
┌──────────────────────┐   │
│ flush_temp_to_main_  │   │
│ metrics()            │   │ ← Append temp → main CSV
└──────┬───────────────┘   │
       │                   │
       ▼                   │
┌──────────────────────┐   │
│ flush_temp_episode_  │   │
│ logs()               │   │ ← Append temp → main log
└──────┬───────────────┘   │
       │                   │
       ▼                   │
┌──────────────────────┐   │
│ get_last_batch_num() │   │ ← Reload batch_num
└──────┬───────────────┘   │
       │                   │
       ▼                   │
┌──────────────────────┐   │
│ plot_training_       │   │
│ curves(iteration)    │   │ ← Plot PNG
└──────┬───────────────┘   │
       │                   │
       ▼                   │
┌──────────────────────┐   │
│ save_checkpoint()    │   │ ← .pth with full state
│ - model_state_dict   │   │
│ - optimizer_state    │   │
│ - iteration          │   │
│ - global_step        │   │
│ - total_episodes     │   │
│ - metrics            │   │
│ - curriculum_state   │   │
└──────┬───────────────┘   │
       │                   │
       └───────────┬───────┘
                   │
                   ▼
         ┌─────────────────┐
         │ iteration += 1  │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ iteration <     │
         │ total_iters?    │
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ┌────────┐        ┌────────┐
    │  YES   │        │  NO    │
    │ (loop) │        │ (end)  │
    └────┬───┘        └────┬───┘
         │                 │
         └────────┬────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│           END TRAINING               │
└──────────────────────────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ plot_training_  │
         │ curves()        │ ← Final plot
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Display final   │
         │ stats           │
         └─────────────────┘
```

---

## Network Architecture

### Overview

```
OBSERVATION (7 + history_dim + grid_dim values)
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Robot   │    │ History  │    │ Grid     │    │ Grid         │
│ State   │    │ (48 vals)│    │Channel 0 │    │ Channel 1    │
│ (7 vals)│    │          │    │(obstacles│    │ (holes)      │
└────┬────┘    └─────┬────┘    └─────┬────┘    └──────┬───────┘
     │               │               │                 │
     ▼               ▼               └────────┬────────┘
┌─────────┐    ┌──────────┐                  │
│ MLP     │    │ MLP      │                  ▼
│ 7→32    │    │ 48→64→32 │         ┌─────────────────┐
└────┬────┘    └─────┬────┘         │ CNN 2 channels  │
     │               │              │ Conv 2→32       │
     │               │              │ Conv 32→64      │
     │               │              │ Flatten         │
     │               │              │ Linear→64       │
     │               │              └────────┬────────┘
     │               │                       │
     └───────┬───────┴───────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │ Concatenate     │
    │ (32+32+64=128)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Backbone MLP    │
    │ 128→64          │
    └────────┬────────┘
             │
             ├─────────────────┬─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Actor Mean  │   │ Actor LogStd│   │ Critic      │
    │ 64→4        │   │ (learnable) │   │ 64→1        │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                  │
           └────────┬────────┘                  │
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐         ┌─────────────┐
           │ Normal(μ, σ)    │         │ Value       │
           │ Sample action   │         │ Estimate    │
           └─────────────────┘         └─────────────┘
```

### Component Details

**1. Robot State MLP (7 → 32)**
- Input: `[x, y, z, vx, vy, vz, theta]`
- Architecture: `Linear(7, 32) + Tanh`
- Role: Encode instantaneous robot state

**2. History MLP (48 → 64 → 32)**
- Input: 8 frames × 6 values (positions + relative velocities)
- Architecture: `Linear(48, 64) + Tanh + Linear(64, 32) + Tanh`
- Role: Encode history for anticipation

**3. CNN (2 channels → 64)**
- Input: Grid `[2, grid_rows, grid_cols]`
  - Channel 0: Obstacles (bumps + walls)
  - Channel 1: Holes (holes + exterior)
- Architecture:
  ```
  Conv2d(2, 32, kernel=3, stride=2, padding=1) + ReLU
  Conv2d(32, 64, kernel=3, stride=2, padding=1) + ReLU
  Flatten
  Linear(64 × conv_rows × conv_cols, 64) + Tanh
  ```
- Role: Extract spatial features from vision

**4. Backbone (128 → 64)**
- Input: Concatenation of 3 encoders (32 + 32 + 64 = 128)
- Architecture: `Linear(128, 64) + Tanh`
- Role: Merge information

**5. Actor Head (64 → 4)**
- Mean: `Linear(64, 4)` (init std=0.01)
- LogStd: `Parameter(zeros(1, 4))` (learnable)
- Distribution: `Normal(mean, exp(logstd))`
- Output: 4 continuous actions `[-1, 1]` for 4 wheels

**6. Critic Head (64 → 1)**
- Architecture: `Linear(64, 1)` (init std=1.0)
- Output: Value estimation for state

---

## Configuration (config.yaml)

### Complete Structure

```yaml
training:
  total_timesteps: 8000000  # Total training steps
  num_envs: 30             # Parallel environments
  num_steps: 1024          # Steps per rollout
  num_minibatches: 32      # Minibatches for update
  update_epochs: 10        # Optimization epochs
  seed: 1                  # Reproducibility seed

ppo:
  lr: 0.0004              # Learning rate
  gamma: 0.995            # Discount factor
  gae_lambda: 0.98        # GAE lambda
  clip_coef: 0.2          # PPO clip coefficient
  ent_coef: 0.01          # Entropy coefficient
  vf_coef: 0.5            # Value function coefficient
  max_grad_norm: 0.5      # Gradient clipping

optimizer:
  eps: 0.00001            # Adam epsilon

environment:
  max_steps: 7000                    # Max steps per episode
  use_random_corridor: true          # Random corridors
  corridor_xml: "corridor_yguel.xml" # If use_random=false

network:
  robot_net_hidden: [32]           # MLP robot: 7→32
  history_net_hidden: [64, 32]     # MLP history: 48→64→32
  cnn_channels: [32, 64]           # CNN: 2→32→64
  cnn_kernel_size: 3               # Kernel 3×3
  cnn_stride: 2                    # Stride 2
  backbone_hidden: [64]            # Backbone: 128→64

logging:
  log_interval: 1          # Log every N iterations
  save_interval: 5         # Save every N iterations
  render_interval: 10      # Render every N iterations
  batch_size_metrics: 20   # Batch size for metrics

curriculum:
  enabled: true            # Enable curriculum
  stabilization_steps: 20  # Initial stabilization steps
  
  bump_ratio_schedule:
    - phase: 1
      bump_ratio: 0.5
      distance_threshold: 10
    - phase: 2
      bump_ratio: 0.65
      distance_threshold: 12
    - phase: 3
      bump_ratio: 0.75
      distance_threshold: 65
    - phase: 4
      bump_ratio: 1.0
      distance_threshold: null

robot:
  max_steering_angle: 30.0  # Max steering angle (°)
  max_speed: 1.0           # Max speed (m/s)
  spawn_angle_max: 30.0    # Max spawn angle (°)

vision:
  cell_size: 0.2           # Grid cell size (m)
  vision_front: 5          # Front vision (m)
  vision_behind: 2         # Rear vision (m)
  vision_left: 2           # Left vision (m)
  vision_right: 2          # Right vision (m)

history:
  history_interval: 20     # Save position every N steps
  history_length: 8        # Number of past positions

corridor:
  corridor_length: 200.0   # Corridor length (m)
  corridor_width: 3.0      # Corridor width (m)
  success_distance: 100.0  # Success distance (m)

rewards:
  success_reward: 50.0      # Success reward
  failure_penalty: -5.0     # Failure penalty
  progress_multiplier: 5.0  # Progress multiplier
  collision_penalty: -0.01  # Collision penalty
  fell_threshold: 0.15      # Fall threshold (m)
  no_progress_check_interval: 750  # Check progress every N steps
  no_progress_min_distance: 0.3    # Min distance required (m)
  no_progress_penalty: -4.0        # No progress penalty
```

### Parameter Explanations

#### Training Section
- `total_timesteps`: Total training steps (8M = ~260 iterations with 30 envs and 1024 steps)
- `num_envs`: Number of parallel environments (more = faster but more RAM/VRAM)
- `num_steps`: Steps per rollout before PPO update (more = more stable but fewer updates)
- `num_minibatches`: Batch division for optimization (32 = 960 steps per minibatch)
- `update_epochs`: Number of passes over collected data (10 = good compromise)
- `seed`: Seed for reproducibility (change to vary training)

#### PPO Section
- `lr`: Learning rate (0.0004 = good start, reduce if unstable)
- `gamma`: Discount factor (0.995 = long-term horizon, close to 1 = more patient)
- `gae_lambda`: GAE lambda for advantage estimation (0.98 = good bias/variance tradeoff)
- `clip_coef`: PPO clip coefficient (0.2 = standard, limits policy changes)
- `ent_coef`: Entropy coefficient (0.01 = encourage exploration, increase if stuck)
- `vf_coef`: Value function coefficient (0.5 = weight of critic loss)
- `max_grad_norm`: Gradient clipping (0.5 = avoid gradient explosions)

#### Optimizer Section
- `eps`: Adam epsilon (1e-5 = numerical stability)

#### Environment Section
- `max_steps`: Maximum steps per episode (7000 = ~70s at 100Hz)
- `use_random_corridor`: true = random generation, false = use corridor_xml
- `corridor_xml`: XML file if use_random_corridor=false

#### Network Section
- `robot_net_hidden`: MLP layers for robot state [32] = 7→32
- `history_net_hidden`: MLP layers for history [64,32] = 48→64→32
- `cnn_channels`: CNN channels [32,64] = 2→32→64
- `cnn_kernel_size`: Convolution kernel size (3 = 3×3)
- `cnn_stride`: Convolution stride (2 = reduce dimensions by 2)
- `backbone_hidden`: Fusion MLP layers [64] = 128→64

#### Logging Section
- `log_interval`: Display logs every N iterations (1 = every iteration)
- `save_interval`: Save checkpoint every N iterations (5 = every 5)
- `render_interval`: Render every N iterations (10 = rarely, expensive)
- `batch_size_metrics`: Batch size for metrics calculation (20 episodes)

#### Curriculum Section
- `enabled`: Enable curriculum learning (true recommended)
- `stabilization_steps`: Steps before first curriculum check (20 iterations)
- `bump_ratio_schedule`: List of phases with:
  - `phase`: Phase number (1, 2, 3, 4)
  - `bump_ratio`: Bump ratio (0.5 = 50%, 1.0 = 100%)
  - `distance_threshold`: Average distance to advance to next phase (null = final phase)

#### Robot Section
- `max_steering_angle`: Maximum steering angle in degrees (30° = realistic for car)
- `max_speed`: Maximum speed in m/s (1.0 = ~3.6 km/h, slow but stable)
- `spawn_angle_max`: Maximum spawn angle in degrees (30° = initial variation)

#### Vision Section
- `cell_size`: Grid cell size in meters (0.2 = 20cm)
- `vision_front`: Front vision distance in meters (5 = see far ahead)
- `vision_behind`: Rear vision distance in meters (2 = rear context)
- `vision_left`: Left vision distance in meters (2 = detect walls)
- `vision_right`: Right vision distance in meters (2 = detect walls)

#### History Section
- `history_interval`: Save position every N steps (20 = ~0.2s at 100Hz)
- `history_length`: Number of past positions (8 = 8 frames × 6 values = 48)

#### Corridor Section
- `corridor_length`: Corridor length in meters (200 = long)
- `corridor_width`: Corridor width in meters (3 = narrow)
- `success_distance`: Distance for success in meters (100 = objective)

#### Rewards Section
- `success_reward`: Success reward (50 = big reward)
- `failure_penalty`: Failure penalty (-5 = moderate penalty)
- `progress_multiplier`: Progress multiplier (5.0 = encourage moving forward)
- `collision_penalty`: Collision penalty per step (-0.01 = small continuous penalty)
- `fell_threshold`: Fall threshold in meters (0.15 = robot fell if z < 0.15m)
- `no_progress_check_interval`: Check progress every N steps (750 = ~7.5s)
- `no_progress_min_distance`: Minimum distance required in meters (0.3 = must advance)
- `no_progress_penalty`: No progress penalty (-4.0 = strong penalty)

### Tuning Advice

**To speed up training:**
- Increase `num_envs` (if RAM/VRAM sufficient)
- Increase `num_steps` (more stable but fewer updates)
- Reduce `update_epochs` (fewer passes over data)

**If training is unstable:**
- Reduce `lr` (0.0003 or 0.0002)
- Increase `ent_coef` (0.02 or 0.03 for more exploration)
- Reduce `num_steps` (more frequent updates)

**If robot doesn't explore enough:**
- Increase `ent_coef` (0.02 or more)
- Reduce `clip_coef` (0.1 for more changes)
- Adjust curriculum (start easier)

**If robot is too cautious:**
- Reduce `collision_penalty` (less punishing)
- Increase `progress_multiplier` (encourage speed)
- Increase `max_speed` (allow faster movement)

**If robot is too aggressive:**
- Increase `collision_penalty` (more punishing)
- Reduce `progress_multiplier` (less rushed)
- Increase `failure_penalty` (more fear of failing)

---

## Tracked Metrics

### Per Batch (training_metrics.csv)

- `batch_num`: Batch number
- `episode_end`: Last episode in batch
- `episodes_range`: Episode range (ex: "1-20")
- `global_step`: Total steps since start
- `mean_return`: Average return for batch
- `mean_distance`: Average distance for batch
- `mean_survival`: Average survival for batch
- `success_rate`: Success rate for batch
- `current_phase`: Current curriculum phase
- `random_percentage`: % random corridors (fixed at 1.0)
- `bump_ratio`: Current bump ratio

### Per Iteration (console logs)

- Return: Recent (mean ± std) | Max
- Distance: Recent (mean ± std) | Max
- Survival: Recent (mean ± std)
- Terminations: fell, flipped, no_progress counts

### Per Episode (episodes_log.txt)

```
Episode 123: fell | Steps: 456 | Distance: 12.34m | Reward: 45.6 | Corridor: holes+50%bumps+random | Seed: 7890
```

---

## Curriculum Learning

### Phase Progression

```
Phase 1: holes + 50% bumps
  ↓ (distance >= 10m)
Phase 2: holes + 65% bumps
  ↓ (distance >= 12m)
Phase 3: holes + 75% bumps
  ↓ (distance >= 65m)
Phase 4: holes + 100% bumps
  (no threshold, final phase)
```

### Check Each Iteration

```
CURRICULUM CHECK
Iteration mean distance: 11.5m
Current phase: 1 (threshold: 10.0m)
✓ Distance >= threshold → Ready for next phase
```

### Automatic Rollback (--rollback)

If performance regresses (distance < last save):
1. Load latest checkpoint
2. Clean temp files
3. Continue training

Without `--rollback`: continue without saving.

---

## Rollback and Curriculum Learning

### Rollback Save System (`--rollback`)

The rollback system allows you to **avoid infinite loops** during curriculum learning phase changes.

**Problem without rollback:**
```
Iteration 15: Save Phase 1, distance: 45m
Iteration 18: Move to Phase 2 (threshold: 50m) ✓, distance: 52m
Iteration 19-20: Slight performance drop (normal post-phase) → distance: 44m
Iteration 25: Distance 44m < saved (45m) → REJECT, rollback to iteration 15
Iteration 26+: Restart from iteration 15 in infinite loop...
```

**Solution with rollback + phase_changed_since_save:**

The system detects a phase change between saves via `phase_changed_since_save` boolean:

1. **On phase change**: `phase_changed_since_save = True`
2. **At save check**:
   - If `phase_changed_since_save == True`: **Force save** (bypass distance comparison)
   - Treat save as first save (no previous baseline to compare)
3. **After save**: `phase_changed_since_save = False`

**Correct flow:**
```
Iteration 15: Save Phase 1, distance: 45m
Iteration 18: Move to Phase 2 (threshold: 50m) ✓
             → phase_changed_since_save = True
Iteration 19-20: Distance: 44m (< 45m normally rejected)
             → But phase_changed_since_save = True
             → FORCE save as new baseline
             → phase_changed_since_save = False
Iteration 25: Normal comparison resumes with baseline 44m
```

### Activation and Behavior

**Activation:**
```bash
python3 train_ppo.py --rollback
```

**Behavior:**
- Save every N iterations
- At each save check, compare current average distance vs last saved distance
- If `current_distance < last_distance` AND no recent phase change:
  - **REJECT**: Rollback to previous checkpoint
  - Reload model, optimizer, and training state
  - Clear temporary metrics
  - Restart iteration from checkpoint
- If `current_distance >= last_distance` OR phase change:
  - **ACCEPT**: Save new checkpoint

**State Restored on Rollback:**
- Model weights and optimizer state
- Iteration and global_step
- Batch tracking (last_batch_episode, episode history)
- Curriculum state (phase_distance_history)

### Progressive 4-Phase Curriculum Learning

| Phase | Description | Example bump_ratio | Distance Threshold |
|-------|-------------|-------------------|-------------------|
| 1 | Simple obstacles | 50% | 50m |
| 2 | Increase bumps | 65% | 70m |
| 3 | High difficulty | 75% | 100m |
| 4 | Max difficulty | 100% | N/A (final) |

Each phase gradually increases difficulty (more bumps) once distance threshold is reached.

### Rollback Configuration Recommendation

**IMPORTANT**: To use rollback effectively, it is **strongly recommended** to configure `save_interval: 1` in `config.yaml`:

```yaml
logging:
  save_interval: 1  # Save every iteration
```

**Why this is critical:**
- Without `save_interval: 1`, saves are spaced out (ex: every 5 iterations)
- If rollback triggers between saves, several iterations might stagnate
- With `save_interval: 1`, rollback can immediately reject a bad iteration → **continuous progress**
- Without this, risk of **infinite loops** or **blocked progress**

**Example problem with save_interval > 1:**
```
Iteration 45: Save ✓ (distance: 45m)
Iteration 46: Performance drops (41m) → Continue (no save check)
Iteration 47: Continues getting worse (38m) → Continue
Iteration 48: Continues (36m) → Continue
Iteration 50: Save check → distance 36m < 45m → REJECT globally
             → Rollback to iteration 45, 4 iterations lost
```

**With save_interval: 1:**
```
Iteration 45: Save ✓ (distance: 45m)
Iteration 46: Performance drops (41m) → Save check → distance < 45m
             → Immediately REJECTED and rollback to iteration 45
             → Minimal loss, training continues productively
```

---

## Generated Files

```
models/
├── ppo_corridor_5.pth          # Checkpoint iteration 5
├── ppo_corridor_10.pth         # Checkpoint iteration 10
├── ...
├── training_metrics.csv        # Complete metrics
├── training_curves_5.png       # Iteration 5 plots
├── training_curves_10.png      # Iteration 10 plots
├── training_curves.png         # Final plots
├── episodes_log.txt            # All episodes log
├── iteration_summary.json      # Latest iteration saved
├── temp_training_metrics.csv   # Temp metrics (flush)
└── temp_episodes_log.txt       # Temp logs (flush)
```

---

## Usage Examples

### Complete Training

```bash
# In Docker environment
cd ~/workspace/ppo_no_steer

# Standard training with rollback
python3 train_ppo.py --rollback
```

### Testing and Evaluation

```bash
# In Docker environment
cd ~/workspace/ppo_no_steer

# Quick test without rendering
python3 test_ppo.py --num-episodes 20

# Test with full visualization
python3 test_ppo.py --render --show-vision --num-episodes 5

# Test on difficult corridor
python3 test_ppo.py --render --bump-ratio 1.0 --num-episodes 10

# Test specific checkpoint
python3 test_ppo.py --model models/ppo_corridor_50.pth --render
```

### Visualization and Debug

```bash
# In Docker environment
cd ~/workspace/ppo_no_steer

# Visualize CNN vision
python3 visualize_corridor_map.py --render

# Specific position
python3 visualize_corridor_map.py --x 50 --y 0.5 --angle 15

# Corridor with fixed seed
python3 visualize_corridor_map.py --seed 42 --bump-ratio 0.8
```

---

## Troubleshooting

### Problem: tkinter not found

```bash
sudo apt update
sudo apt install python3-tk python3-pil.imagetk
pip install pillow
```

### Problem: CUDA out of memory

Reduce `num_envs` in config.yaml:
```yaml
training:
  num_envs: 16  # Instead of 30
```

### Problem: Training doesn't progress

1. Check curriculum: phase too difficult?
2. Reduce learning rate: `--lr 0.0002`
3. Increase entropy: `ent_coef: 0.02` in config
4. Use `--fresh-start` to restart

---

## Important Notes

- Checkpoints include full state (model, optimizer, iteration, metrics)
- Resume is automatic from latest checkpoint
- Temp files flushed on each save
- Batch numbering continues after flush
- Iteration stats reset each iteration
- Curriculum progresses automatically based on average distance

---
