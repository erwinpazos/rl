# Basic Robot Control Training

This folder contains the basic locomotion learning system where the robot learns to control its 4 wheels from scratch on flat ground.

## Purpose

Before learning complex navigation with obstacles, the robot must first master basic locomotion:
- **Forward movement**: Coordinating all 4 wheels to move straight
- **Turning**: Learning differential drive (left wheels vs right wheels)
- **Stability**: Staying upright and not spinning out of control
- **Target reaching**: Moving towards specific coordinates

## Files

- `robot_basic_env.py`: Flat ground environment with target reaching
- `train_basic_control.py`: PPO training for basic locomotion
- `test_basic_control.py`: Test trained locomotion models
- `four_wheels_robot.xml`: Robot model (copied from ppo/)

## Key Differences from Navigation Training

### Environment
- **Flat ground only**: No obstacles, holes, or ramps
- **Random targets**: Robot must reach randomly placed targets
- **Raw wheel control**: 4 independent wheel torques (no pre-made steering)

### Rewards
- **Forward progress**: +2.0 * delta_x for moving forward
- **Target reaching**: +20.0 for reaching targets (< 1m distance)
- **Stability**: +0.2 for staying upright (not tilted)
- **Anti-spinning**: -0.1 * |angular_velocity| to discourage spinning
- **Proximity**: -0.01 * distance_to_target to encourage approach

### Observations (13 values)
- Position: x, y, z (3)
- Orientation: quaternion w, x, y, z (4) 
- Linear velocity: vx, vy, vz (3)
- Angular velocity: wx, wy, wz (3)

### Actions (4 values)
- Raw wheel torques: [front_left, front_right, rear_left, rear_right]
- Range: [-1, 1] scaled to [-20, 20] Nm

## Usage

### Training
```bash
cd basic_control
python train_basic_control.py
```

### Testing
```bash
# Test latest model
python test_basic_control.py --render

# Test specific model
python test_basic_control.py --model-path models/basic_robot_control_1_1234567890/basic_control.pth --render
```

## Expected Learning Progression

1. **Random flailing** (0-100k steps): Robot spins and falls randomly
2. **Basic stability** (100k-500k steps): Learns to stay upright
3. **Forward movement** (500k-1M steps): Discovers how to move forward
4. **Turning discovery** (1M-1.5M steps): Learns differential drive
5. **Target reaching** (1.5M-2M steps): Combines movement + turning for navigation

## Success Metrics

- **Stability**: Episodes lasting full 2000 steps without falling
- **Movement**: Consistent forward progress (positive delta_x)
- **Turning**: Ability to change direction (varying delta_y)
- **Target reaching**: Successfully hitting 1+ targets per episode

Once the robot masters basic control here, it can be transferred to the complex corridor navigation task with pre-trained locomotion skills.