# PPO Steer - Steering Angle and Speed Control

Version with high-level control via steering angle and speed (action space: 2 dimensions).

**Location**: `mujoco/workspace/ppo_steer/`

## Table of Contents

- [Overview](#overview)
- [Available Scripts](#available-scripts)
- [Training Pipeline](#training-pipeline)
- [Network Architecture](#network-architecture)
- [Configuration](#configuration)

---

## Overview

This version controls the robot like a car:
- **Action space**: `Box(-1.0, 1.0, (2,))` 
- **Actions**: `[steering_angle, speed]`
  - `steering_angle`: Normalized steering wheel angle [-1, 1] → [-30°, +30°]
  - `speed`: Normalized speed [-1, 1] → [-max_speed, +max_speed]
- Automatic conversion to 4 wheel speeds via `steer_angle_to_wheel_speeds()`
- More natural and simpler to learn

**Prerequisites**: Docker environment launched (see [main README](../../../README.md))

---

## Steering → Wheel Speeds Conversion

The `steer_angle_to_wheel_speeds()` function converts high-level commands to wheel speeds:

```python
def steer_angle_to_wheel_speeds(steering_angle, speed, wheelbase, track_width):
    """
    steering_angle: angle in radians [-π/6, +π/6] (±30°)
    speed: linear speed in m/s
    wheelbase: distance between axles (m)
    track_width: width between wheels (m)
    
    Returns: [v_FL, v_FR, v_RL, v_RR] (angular speeds rad/s)
    """
```

**Principle:**
1. Calculate steering radius: `R = wheelbase / tan(steering_angle)`
2. Calculate linear speeds of left/right wheels
3. Convert to angular speeds: `ω = v / wheel_radius`

---

## Available Scripts

### 1. train_ppo.py - Training

Main training script with PPO and steering control.

**Usage:**
```bash
# In Docker environment (http://localhost:6080)
cd ~/workspace/ppo_steer
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
cd ~/workspace/ppo_steer

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
- 2D action space: `[steering_angle, speed]`

**Generated files:**
- `models/ppo_corridor_{iteration}.pth`: Saved checkpoints
- `models/training_metrics.csv`: Training metrics
- `models/training_curves_{iteration}.png`: Iteration plots
- `models/training_curves.png`: Final plot
- `models/episodes_log.txt`: Detailed episode log
- `models/iteration_summary.json`: Latest iteration summary

---

### 2. test_ppo.py - Model Testing

Tests a trained model on N episodes with steering control.

**Usage:**
```bash
# In Docker environment
cd ~/workspace/ppo_steer
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
cd ~/workspace/ppo_steer

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
- Steering actions displayed: `[angle, speed]`

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

Custom environment for robot in corridor with steering control.

**Features:**
- Gymnasium compatible (gym.Env)
- Observation: robot state + history + CNN grid (2 channels)
- Action: 2 values `[steering_angle, speed]` in `[-1, 1]`
- Rewards: progress, collision, success/failure
- Termination: fell, flipped, no_progress, success

**Action space and conversion:**

```python
# Action space
self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

# In step()
def step(self, action):
    steering_angle_normalized = action[0]  # [-1, 1]
    speed_normalized = action[1]           # [-1, 1]
    
    # Denormalize
    steering_angle = steering_angle_normalized * self.max_steering_angle_rad
    speed = speed_normalized * self.max_speed
    
    # Convert to wheel speeds
    wheel_speeds = steer_angle_to_wheel_speeds(
        steering_angle, speed, 
        self.wheelbase_length, self.track_width
    )
    
    # Apply to MuJoCo actuators
    self.data.ctrl[:] = wheel_speeds
```

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
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Create Agent     │ ← Actor: 64→2 (steering+speed)
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
                   └──────────┬───────┘
                              │
                              ▼
         ┌─────────────────────────────────┐
         │ Create parallel environments    │
         │ AsyncVectorEnv (30 envs)        │
         │ Action space: Box(-1,1,(2,))    │
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
         │ COLLECT ROLLOUT             │
         │ (30 envs × 1024 steps)      │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ For each step:              │
         │ 1. agent.get_action(obs)    │ ← Forward: obs → [steering, speed]
         │ 2. envs.step(actions)       │ ← Conversion: [s,v] → [4 wheels]
         │ 3. Store transitions        │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ COMPUTE ADVANTAGES          │
         │ compute_gae()               │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ PPO UPDATE                  │
         │ (10 epochs × 32 minibatches)│
         │ Actor: 2 actions            │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ METRICS + SAVE              │
         │ (identical to no_steer)     │
         └─────────────────────────────┘
```

### Pipeline Details

**Initialization:**
1. Load configuration from `config.yaml`
2. Create agent with CNN+MLP architecture (actor head: 64→2)
3. Find latest checkpoint for automatic resume
4. Create 30 parallel environments with 2D action space

**Training loop (260 iterations):**
1. Reset iteration stats
2. Determine curriculum phase (bump_ratio, max_steps)
3. Collect rollout (30 envs × 1024 steps):
   - Forward pass: obs → `[steering, speed]`
   - Automatic conversion: `[steering, speed]` → 4 wheel speeds
   - MuJoCo simulation
   - Store transitions
4. Compute advantages with GAE
5. PPO update (10 epochs × 32 minibatches)
6. Compute batch metrics
7. Save to temp CSV
8. Every N iterations:
   - Check performance (rollback if regression)
   - Save checkpoint + metrics
   - Generate plots
   - Flush temp → main files

**Key characteristics:**
- 2D action space simplifies learning
- Steering→wheels conversion in environment
- Progressive curriculum (4 phases)
- Optional automatic rollback

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
│ State   │    │ (48 vals)│    │ Channel 0│    │ Channel 1    │
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
    │ 64→2        │   │ (learnable) │   │ 64→1        │
    │ [steer,spd] │   │             │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                  │
           └────────┬────────┘                  │
                    │                           │
                    ▼                           ▼
           ┌─────────────────┐         ┌─────────────┐
           │ Normal(μ, σ)    │         │ Value       │
           │ Sample action   │         │ Estimate    │
           │ [steering,speed]│         │             │
           └─────────────────┘         └─────────────┘
```

**Difference with no_steer:**
- Actor Mean: `Linear(64, 2)` instead of `Linear(64, 4)`
- Actor LogStd: `Parameter(zeros(1, 2))` instead of `(1, 4)`
- Output: `[steering_angle, speed]` instead of `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]`

Rest of architecture is identical.

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

**Specific to steering control:**
- Adjust `max_steering_angle` if robot turns too much/too little (20-45° typical)
- Check physical parameters (wheelbase, track_width) in corridor_env.py
- Steering control typically converges faster than 4 independent wheels

---

## Action Interpretation

### Action Space

```python
action = [steering_angle_normalized, speed_normalized]
# Each value in [-1, 1]
```

### Action Examples

| Action | Interpretation | Result |
|--------|---------------|--------|
| `[0.0, 0.8]` | Straight, 80% speed | Go straight |
| `[1.0, 0.8]` | Max left (+30°), 80% speed | Turn left |
| `[-1.0, 0.8]` | Max right (-30°), 80% speed | Turn right |
| `[0.5, 0.5]` | Left 15°, 50% speed | Soft left turn |
| `[0.0, -0.5]` | Straight, reverse 50% | Reverse straight |
| `[0.0, 0.0]` | No movement | Stop |

### Conversion to Wheel Speeds

```python
# Example: action = [0.5, 0.8]
steering_angle = 0.5 * 30° = 15° = 0.262 rad
speed = 0.8 * 1.0 m/s = 0.8 m/s

# Calculate steering radius
R = wheelbase / tan(steering_angle)
R = 0.5 / tan(0.262) = 1.88 m

# Linear speeds
v_left = speed * (R - track_width/2) / R
v_right = speed * (R + track_width/2) / R

# Angular speeds (rad/s)
ω_left = v_left / wheel_radius
ω_right = v_right / wheel_radius

# Result: [ω_FL, ω_FR, ω_RL, ω_RR]
```

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

## Usage Examples

### Training

```bash
# In Docker environment
cd ~/workspace/ppo_steer

# Standard training with rollback
python3 train_ppo.py --rollback

# Fresh start
python3 train_ppo.py --fresh-start

# Override parameters
python3 train_ppo.py --lr 0.0003 --num-envs 16 --seed 42
```

### Testing

```bash
# In Docker environment
cd ~/workspace/ppo_steer

# Test with full visualization
python3 test_ppo.py --render --show-vision --num-episodes 5

# Test on difficult corridor
python3 test_ppo.py --render --bump-ratio 1.0

# Test specific checkpoint
python3 test_ppo.py --model models/ppo_corridor_50.pth --render
```

### Visualization

```bash
# In Docker environment
cd ~/workspace/ppo_steer

# Visualize CNN vision
python3 visualize_corridor_map.py --render

# Specific position
python3 visualize_corridor_map.py --x 50 --y 0.5 --angle 15
```

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

## Advantages of Steering Control

### Advantages

1. **Smaller action space**: 2D instead of 4D
   - Fewer dimensions to explore
   - Potentially faster convergence
   - Less variance in gradients

2. **More natural control**: Like a real car
   - Steering + speed = human intuition
   - Physical constraints automatically respected
   - Behavior more predictable

3. **Fewer degrees of freedom**:
   - Can't make "impossible" movements
   - More coherent and stable behavior
   - Less risk of divergence

4. **Facilitated learning**:
   - More structured action space
   - Natural correlations between actions
   - Better generalization

### Disadvantages

1. **Less flexibility**:
   - Can't turn in place
   - Can't make lateral movements
   - Minimum turning radius

2. **Dependency on conversion**:
   - Quality depends on `steer_angle_to_wheel_speeds()`
   - Physical parameters must be correct (wheelbase, track_width)
   - Modeling errors can affect performance

3. **Mechanical constraints**:
   - Limited by max_steering_angle (30°)
   - Low-speed behavior can be unstable
   - Less suitable for complex maneuvers

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

### Problem: Robot spins in circles

1. Check physical parameters (wheelbase, track_width)
2. Check max_steering_angle (30° default)
3. Increase entropy for more exploration
4. Verify steering→wheels conversion is correct

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

## Important Notes

- Action space: 2 dimensions `[steering_angle, speed]` in `[-1, 1]`
- Automatic conversion to 4 wheel speeds in environment
- Checkpoints contain full state (model, optimizer, iteration, metrics)
- Automatic resume from latest checkpoint
- Temp files flushed on each save
- Curriculum progresses automatically based on average distance
- Critical physical parameters: wheelbase, track_width, wheel_radius

---

## Comparison with ppo_no_steer

### Main Differences

| Aspect | ppo_steer | ppo_no_steer |
|--------|-----------|--------------|
| Action space | 2D `[steering, speed]` | 4D `[wheel_FL, FR, RL, RR]` |
| Actor head | `Linear(64, 2)` | `Linear(64, 4)` |
| Control | High level (car) | Low level (wheels) |
| Conversion | Automatic in env | No conversion |
| Flexibility | Limited (physical constraints) | Maximum |
| Learning | Simpler (2D) | More complex (4D) |
| Checkpoints | Not compatible | Not compatible |

### When to Use ppo_steer?

- Car-like behavior desired
- Faster learning priority
- Realistic physical constraints important
- Reduced action space preferred

### When to Use ppo_no_steer?

- Complex maneuvers needed
- Fine wheel control required
- Maximum flexibility priority
- Unconventional movements possible

---

## References

- Complete configuration: `config.yaml`
- Environment: `corridor_env.py`
- PPO agent: `train_ppo.py`
- Steering conversion: `steer_angle_to_wheel_speeds()` function in `corridor_env.py`
- Corridor generator: `corridor_generator_similar.py`

---
