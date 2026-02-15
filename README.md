# PPO Robot Navigation - Corridor Environment

Training a 4-wheel robot to navigate a corridor with obstacles (holes and bumps) using the PPO (Proximal Policy Optimization) algorithm.

This project uses Docker with NVIDIA GPU support for training and MuJoCo for physical simulation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Launching the Environment](#launching-the-environment)
- [Quick Start](#quick-start)
- [Differences between ppo_no_steer and ppo_steer](#differences-between-ppo_no_steer-and-ppo_steer)

---

## Prerequisites (either one)

### Windows

- Windows 10/11 (version 21H2 or higher)
- NVIDIA GPU with drivers installed (recommended for training)
- 8 GB RAM minimum (16 GB recommended)
- WSL2 (Windows Subsystem for Linux)
- Docker Desktop

### Linux

- Ubuntu 20.04+ or compatible distribution
- NVIDIA GPU with drivers installed (recommended)
- 8 GB RAM minimum (16 GB recommended)
- Docker with GPU support (NVIDIA Container Toolkit)

---

## Installation

### 1. Clone the Repository

The repository should be cloned in a specific folder according to your operating system:

**Windows (in PowerShell):**
```powershell
# Location: C:\Users\YOUR_USERNAME
```

**Linux / WSL:**
```bash
# Location: /home/YOUR_USERNAME
```

**Clone the repository:**
```bash
cd $env:USERPROFILE
git clone https://github.com/erwinpazos/rl.git .

# Linux / WSL
cd ~
git clone https://github.com/erwinpazos/rl.git .
```

The repository contains:
- `launch_scripts/`: Docker launch scripts (start.bat, start.sh)
- `mujoco/workspace/`: Project source code (ppo_no_steer, ppo_steer, etc.)

### 2. Install WSL2 and Docker

#### Windows

**For complete WSL2, Docker Desktop installation and GPU configuration on Windows, see:**

📖 **[Complete Windows Installation Guide in launch_scripts/README.md](launch_scripts/README.md)**

This guide contains:
- WSL2 Installation
- Docker Desktop Installation
- Docker Configuration for WSL2
- GPU Verification
- Windows Troubleshooting

#### Linux

**Install Docker:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in to apply group changes
```

**Install NVIDIA Container Toolkit (for GPU):**
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the GPU
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Installation

**Windows (in WSL):**
```bash
wsl
nvidia-smi
```

**Linux:**
```bash
nvidia-smi
```

You should see your NVIDIA GPU information.

**Test Docker with GPU:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## Project Structure

```
rl/                                    # Root folder (created by scripts)
├── launch_scripts/                    # Launch scripts
│   ├── start.bat                      # Windows launch
│   ├── start.sh                       # Linux/WSL launch
│   ├── README.md                      # Detailed Docker documentation
│   └── *.png                          # Documentation images
│
└── mujoco/                            # MuJoCo environment
    └── workspace/                     # Source code (shared with Docker)
        ├── README.md                  # This file
        │
        ├── ppo_no_steer/              # 4 independent wheels version
        │   ├── README.md              # Detailed documentation
        │   ├── config.yaml            # Complete configuration
        │   ├── train_ppo.py           # Training script
        │   ├── test_ppo.py            # Test script
        │   ├── corridor_env.py        # Gymnasium environment
        │   ├── *.xml                  # MuJoCo models
        │   ├── models/                # Checkpoints and metrics
        │   └── utils/                 # Utility modules
        │
        ├── ppo_steer/                 # Steering control version
        │   ├── README.md              # Detailed documentation
        │   ├── config.yaml            # Complete configuration
        │   ├── train_ppo.py           # Training script
        │   ├── test_ppo.py            # Test script
        │   ├── corridor_env.py        # Gymnasium environment (steering)
        │   ├── *.xml                  # MuJoCo models
        │   ├── models/                # Checkpoints and metrics
        │   └── utils/                 # Utility modules
        │
        ├── ppo_final/                 # Final version (reference)
        ├── corridor_creation/         # Corridor creation tools
        └── notebooks/                 # Experimentation notebooks
```

---

## Launching the Environment

The project uses Docker to provide a complete environment with:
- MuJoCo Desktop (graphical interface via noVNC)
- Jupyter Notebook
- CUDA GPU support for PyTorch
- All necessary Python packages

### Option 1: Launch via WSL (Windows)

**Advantages:**
- ✅ Functional CUDA GPU (PyTorch/TensorFlow)
- ✅ Jupyter works correctly
- ✅ Optimal performance for training

**Launch:**
```powershell
# In PowerShell
wsl

# In WSL
cd ~/rl/launch_scripts
./start.sh
```

**Shared folder:**
```
Linux (WSL): /home/YOUR_USERNAME/rl/mujoco/workspace
Docker:      /home/student/workspace
Windows:     \\wsl.localhost\Ubuntu\home\YOUR_USERNAME\rl\mujoco\workspace
```

### Option 2: Direct launch from Windows

**Advantages:**
- ✅ Simple, double-click on start.bat
- ✅ Native Windows folder (easy to access)

**Disadvantages:**
- ⚠️ CUDA GPU may not work
- ⚠️ Jupyter may have issues

**Launch:**
```powershell
# Double-click on start.bat or in PowerShell:
cd C:\Users\YOUR_USERNAME\rl\launch_scripts
.\start.bat
```

**Shared folder:**
```
Windows: C:\Users\YOUR_USERNAME\rl\mujoco\workspace
Docker:  /home/student/workspace
```

### Option 3: Native Linux launch

**Launch:**
```bash
cd ~/rl/launch_scripts
./start.sh
```

**Shared folder:**
```
Linux:  /home/YOUR_USERNAME/rl/mujoco/workspace
Docker: /home/student/workspace
```

### Launch Options

```bash
# Custom resolution
./start.sh --resolution 2560x1440

# RAM saving mode
./start.sh --small_ram

# Custom RAM
./start.sh --ram 2g

# Display quality
./start.sh --quality medium  # or low

# Local mode (without checking updates)
./start.sh --local

# Without GPU (force software rendering)
./start.sh --no_gpu
```

### Accessing the Environment

Once launched, open in your browser:

- **Desktop noVNC**: http://localhost:6080
- **Jupyter Notebook**: http://localhost:8888

### Stopping the Environment

In the terminal where the environment is running:
```
Ctrl+C
```

---

## Quick Start

### In the Docker Environment

Once the environment is launched (see previous section), access the Desktop via http://localhost:6080

### Choose a Version

The project offers two versions with different control methods:

#### ppo_no_steer - 4 Independent Wheels Control
- Action space: 4 dimensions `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]`
- Direct control of each wheel
- More freedom but harder to learn

📖 **[Complete documentation: mujoco/workspace/ppo_no_steer/README.md](mujoco/workspace/ppo_no_steer/README.md)**

#### ppo_steer - Steering Angle and Speed Control
- Action space: 2 dimensions `[steering_angle, speed]`
- Control like a car (more natural)
- Simpler to learn

📖 **[Complete documentation: mujoco/workspace/ppo_steer/README.md](mujoco/workspace/ppo_steer/README.md)**

### Edit Files

**From Windows:**
- Via WSL: `\\wsl.localhost\Ubuntu\home\YOUR_USERNAME\rl\mujoco\workspace`
- Via Windows: `C:\Users\YOUR_USERNAME\rl\mujoco\workspace` (if launched with start.bat)

**From Linux:**
- `~/rl/mujoco/workspace`

**With VSCode:**
- Open the workspace folder directly
- Changes are synchronized in real-time with Docker

---

## Verify GPU Support

In Jupyter (http://localhost:8888) or in the Docker terminal:

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available - training will be slower")
```

Expected result with GPU:
```
Device: cuda:0
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.x
```

---

## Comparison of Launch Methods

| Criteria | WSL (start.sh) | Windows (start.bat) | Linux (start.sh) |
|----------|----------------|---------------------|------------------|
| CUDA GPU | ✅ Works | ✅ Works | ✅ Works |
| Jupyter | ✅ Works | ⚠️ Possible issues | ✅ Works |
| Shared folder | `/home/USERNAME/rl/...` | `C:\Users\USERNAME\rl\...` | `/home/USERNAME/rl/...` |
| Simplicity | ⚠️ WSL terminal | ✅ Double-click | ✅ Terminal |
| Performance | ✅ Optimal | ⚠️ Average | ✅ Optimal |
| Recommended for | GPU training | Quick tests | GPU training |

---

## Objective

The robot must learn to navigate a 100m corridor with:
- **Holes**: Areas to avoid (falling = failure)
- **Bumps**: Obstacles that slow down and penalize
- **Lateral walls**: Corridor limits (3m wide)

The robot uses:
- **Ego-centric vision**: 2-channel CNN grid (obstacles, holes)
- **Position history**: 8 past frames for anticipation
- **Robot state**: Position, velocity, orientation

---

## Training Metrics

The following metrics are tracked:
- **Average return**: Cumulative reward per episode
- **Average distance**: Distance traveled (objective: 100m)
- **Success rate**: % of episodes reaching 100m
- **Average survival**: Number of steps before termination
- **Termination reasons**: fell, flipped, no_progress, success

---

## Curriculum Learning

Training uses a progressive curriculum:

1. **Phase 1**: Holes + 50% bumps (threshold: 10m)
2. **Phase 2**: Holes + 65% bumps (threshold: 12m)
3. **Phase 3**: Holes + 75% bumps (threshold: 65m)
4. **Phase 4**: Holes + 100% bumps (no threshold)

Progression to the next phase happens automatically when the average distance of the iteration exceeds the threshold.

---

## Configuration

All parameters are configurable via `config.yaml` in each folder (ppo_no_steer, ppo_steer):
- PPO hyperparameters (learning rate, gamma, etc.)
- Network architecture (CNN, MLP)
- Environment parameters (max_steps, vision)
- Curriculum learning (phases, thresholds)
- Reward system

See specific READMEs for more details.

---

## Troubleshooting

### Docker won't start

**Windows:**
- Verify WSL2 is installed: `wsl --status`
- Verify Docker Desktop is running
- Check WSL2 integration in Docker Desktop Settings

**Linux:**
- Verify Docker is running: `sudo systemctl status docker`
- Verify you're in the docker group: `groups`
- If not: `sudo usermod -aG docker $USER` then log out/in

### GPU not detected

**Check NVIDIA driver:**
```bash
nvidia-smi
```

**Check Docker GPU support:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**If it doesn't work:**
- Install NVIDIA Container Toolkit (see Installation section)
- Restart Docker: `sudo systemctl restart docker`
- Use `./start.sh` from WSL (not start.bat)

### Jupyter won't start

- Use `./start.sh` from WSL instead of `start.bat`
- Check Docker logs
- Try with `--small_ram` if low on memory

### Files not synchronized

- Verify you're editing in the correct folder
- WSL: `\\wsl.localhost\Ubuntu\home\USERNAME\rl\mujoco\workspace`
- Windows: `C:\Users\USERNAME\rl\mujoco\workspace`
- Changes should appear immediately in Docker

### Slow performance

- Verify GPU is being used (see "Verify GPU Support" section)
- Reduce `num_envs` in config.yaml if low on RAM
- Use `--small_ram` when launching
- Close resource-heavy applications

---

## Important Notes

- The `rl/` folder is created automatically by the launch scripts (will use your cloned rl folder)
- Checkpoints and metrics are saved in `models/` of each project
- File changes are synchronized in real-time with Docker
- CUDA GPU is essential for fast training (CPU = very slow)
- WSL2 + start.sh is the recommended method on Windows for GPU

---

## License

MIT

---

## 👥 Authors

**Erwin PAZOS** - [GitHub](https://github.com/erwinpazos)

**Docker Image** provided by [Manuel Yguel](https://github.com/yguel)
