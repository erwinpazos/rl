# MuJoCo Desktop Environment with NVIDIA GPU

Installation and usage guide for MuJoCo environment with GPU support for reinforcement learning.


## Prerequisites

- Windows 10/11 (version 21H2 or higher)
- NVIDIA GPU with drivers installed
- 8 GB RAM minimum (16 GB recommended)

## Installation

### 1. Install WSL2 (Windows Subsystem for Linux)

**Important**: WSL2 must be installed **before** Docker Desktop.

Open **PowerShell as administrator** and run:

```powershell
wsl --install
wsl --update
```

Restart your computer if prompted.

### 2. First WSL Launch

In PowerShell (no admin needed):

```powershell
wsl
```

On first launch, WSL will ask for:
- A **username**
- A **password** (remember it!)

### 3. Install Docker Desktop

1. Download Docker Desktop for Windows: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
2. Run the `.exe` installer
3. Follow installation instructions (Docker will automatically detect WSL2)
4. Restart your computer if prompted
5. Launch Docker Desktop

### 4. Configure Docker Desktop for WSL2

1. Open **Docker Desktop**

2. **Check WSL2 integration in General:**
   - Go to **Settings → General**
   - Verify that **"Use the WSL 2 based engine"** is checked
   
   ![General Configuration](general.png)

3. **Enable integration with your WSL distribution:**
   - Go to **Settings → Resources → WSL Integration**
   - Enable:
     - ✅ "Enable integration with my default WSL distro"
     - ✅ Your Ubuntu distribution (or your distro name)
   
   ![Resources Configuration](ressources.png)

4. Click **"Apply & Restart"**

5. **Verify integration is active** after restart:
   - Return to **Settings → Resources → WSL Integration**
   - Confirm checkboxes are still checked

### 4. Verify GPU Installation in WSL

In WSL (type `wsl` in PowerShell):

```bash
nvidia-smi
```

You should see your NVIDIA GPU information.

If not, install drivers:

https://www.nvidia.com/en-us/drivers/

Test Docker with GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```


## Launching the Environment

---

## PART 1: Launch via WSL (Recommended for GPU)

### Launch command:

Open PowerShell and launch WSL:

```powershell
wsl
```

Then in WSL, navigate to the folder with start.sh:

For example:
```bash
cd /mnt/c/Users/YOUR_USERNAME/Document/appr_renf
./start.sh
```

### Shared folder:

**Linux (WSL)** ↔️ **Docker**
```
/home/YOUR_USERNAME/rl/mujoco/workspace  ←→  /home/student/workspace
```

### Edit files from Windows:

```
\\wsl.localhost\Ubuntu\home\YOUR_USERNAME\rl\mujoco\workspace
```


### Advantages:
- CUDA GPU functional (PyTorch/TensorFlow)
- Jupyter works correctly
- Better GPU detection
- Optimal performance for training

### Disadvantages:
- Different folder from Windows launch
- Requires opening WSL

### Access:
- Desktop: http://localhost:6080
- Jupyter: http://localhost:8888

---

## PART 2: Direct Launch from Windows

### Launch command:

Double-click `start.bat` or in PowerShell:

```powershell
cd C:\Users\YOUR_USERNAME\Documents\appr_renf
.\start.bat
```

### Shared folder:

**Windows** ↔️ **Docker**
```
C:\Users\YOUR_USERNAME\rl\mujoco\workspace  ←→  /home/student/workspace
```

### Edit files from Windows:

**VSCode directly:**
```
Open folder: C:\Users\YOUR_USERNAME\rl\mujoco\workspace
```

**Windows Explorer:**
```
C:\Users\YOUR_USERNAME\rl\mujoco\workspace
```

### Advantages:
- Simple, double-click on start.bat
- Native Windows folder (easy access)
- No need to open WSL

### Disadvantages:
- CUDA GPU may not work (less reliable detection)
- Jupyter may have permission issues
- Software rendering for OpenGL

### Access:
- Desktop: http://localhost:6080
- Jupyter: http://localhost:8888 (may not start)

---

## Quick Comparison:

| Criteria | WSL (start.sh) | Windows (start.bat) |
|---------|----------------|---------------------|
| CUDA GPU | ✅ Works | ❌ May fail |
| Jupyter | ✅ Works | ⚠️ Possible issues |
| Shared folder | `/home/USERNAME/...` | `C:\Users\USERNAME\...` |
| Simplicity | ⚠️ WSL Terminal | ✅ Double-click |
| Performance | ✅ Optimal | ⚠️ Average |

---


## Accessing the Environment

Once launched, open in your browser:

- **Desktop noVNC**: http://localhost:6080
- **Jupyter Notebook**: http://localhost:8888


## Verify GPU Support

In Jupyter (http://localhost:8888), create a new notebook and test:

```python
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU available")
```

Expected result:
```
Device: cuda:0
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 1x.x
```

## Stop the Environment

In the terminal where the environment is running:

```
Ctrl+C
```

The container will stop cleanly.

## Launch Options

### Custom resolution

```bash
./start.sh --resolution 2560x1440
```

### RAM saving mode

```bash
./start.sh --small_ram
```

### Custom RAM

```bash
./start.sh --ram 2g
```

### Display quality

```bash
./start.sh --quality medium  # or low
```

### Local mode (without checking for updates)

```bash
./start.sh --local
```

## CUDA vs OpenGL Support

### CUDA (GPU Computing)
- PyTorch, TensorFlow
- Neural network training
- **Works with this setup**

### OpenGL (3D Display)
- MuJoCo visualization
- Software rendering (CPU)
- Slightly slower but functional

**Note**: RL training mainly uses CUDA (fast), 3D display is mostly for visualization/debugging.

## Resources

- MuJoCo: https://mujoco.org/
- Docker Desktop: https://www.docker.com/products/docker-desktop
- WSL Documentation: https://docs.microsoft.com/windows/wsl/
