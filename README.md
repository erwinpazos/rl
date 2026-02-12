# PPO Robot Navigation - Corridor Environment

Entraînement d'un robot à 4 roues pour naviguer dans un corridor avec obstacles (trous et bosses) en utilisant l'algorithme PPO (Proximal Policy Optimization).

Ce projet utilise Docker avec support GPU NVIDIA pour l'entraînement et MuJoCo pour la simulation physique.

## 📋 Table des matières

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Lancement de l'environnement](#lancement-de-lenvironnement)
- [Utilisation rapide](#utilisation-rapide)
- [Différences entre ppo_no_steer et ppo_steer](#différences-entre-ppo_no_steer-et-ppo_steer)

---

## 🔧 Prérequis (l'un ou l'autre)

### Windows

- Windows 10/11 (version 21H2 ou supérieure)
- GPU NVIDIA avec drivers installés (recommandé pour l'entraînement)
- 8 GB RAM minimum (16 GB recommandé)
- WSL2 (Windows Subsystem for Linux)
- Docker Desktop

### Linux

- Ubuntu 20.04+ ou distribution compatible
- GPU NVIDIA avec drivers installés (recommandé)
- 8 GB RAM minimum (16 GB recommandé)
- Docker avec support GPU (NVIDIA Container Toolkit)

---

## 🚀 Installation

### 1. Cloner le dépôt

Le dépôt doit être cloné dans un dossier spécifique selon votre système d'exploitation:

**Windows (dans PowerShell):**
```powershell
# Emplacement: C:\Users\VOTRE_USERNAME
```

**Linux / WSL:**
```bash
# Emplacement: /home/VOTRE_USERNAME
```

**Cloner le dépôt:**
```bash
cd $env:USERPROFILE
git clone https://github.com/erwinpazos/rl.git .

# Linux / WSL
cd ~
git clone https://github.com/erwinpazos/rl.git .
```

Le dépôt contient:
- `launch_scripts/`: Scripts de lancement Docker (start.bat, start.sh)
- `mujoco/workspace/`: Code source du projet (ppo_no_steer, ppo_steer, etc.)

### 2. Installer WSL2 et Docker

#### Windows

**Pour l'installation complète de WSL2, Docker Desktop et la configuration GPU sur Windows, voir:**

📖 **[Guide d'installation Windows complet dans launch_scripts/README.md](launch_scripts/README.md)**

Ce guide contient:
- Installation WSL2
- Installation Docker Desktop
- Configuration Docker pour WSL2
- Vérification GPU
- Troubleshooting Windows

#### Linux

**Installer Docker:**
```bash
# Installer Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Ajouter votre utilisateur au groupe docker
sudo usermod -aG docker $USER

# Se déconnecter et reconnecter pour appliquer les changements de groupe
```

**Installer NVIDIA Container Toolkit (pour GPU):**
```bash
# Ajouter le repository NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Installer le toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configurer Docker pour utiliser le GPU
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Vérifier l'installation GPU

**Windows (dans WSL):**
```bash
wsl
nvidia-smi
```

**Linux:**
```bash
nvidia-smi
```

Vous devriez voir les informations de votre GPU NVIDIA.

**Tester Docker avec GPU:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## 📁 Structure du projet

```
rl/                                    # Dossier racine (créé par les scripts)
├── launch_scripts/                    # Scripts de lancement
│   ├── start.bat                      # Lancement Windows
│   ├── start.sh                       # Lancement Linux/WSL
│   ├── README.md                      # Documentation détaillée Docker
│   └── *.png                          # Images documentation
│
└── mujoco/                            # Environnement MuJoCo
    └── workspace/                     # Code source (partagé avec Docker)
        ├── README.md                  # Ce fichier
        │
        ├── ppo_no_steer/              # Version 4 roues indépendantes
        │   ├── README.md              # Documentation détaillée
        │   ├── config.yaml            # Configuration complète
        │   ├── train_ppo.py           # Script d'entraînement
        │   ├── test_ppo.py            # Script de test
        │   ├── corridor_env.py        # Environnement Gymnasium
        │   ├── *.xml                  # Modèles MuJoCo
        │   ├── models/                # Checkpoints et métriques
        │   └── utils/                 # Modules utilitaires
        │
        ├── ppo_steer/                 # Version contrôle par volant
        │   ├── README.md              # Documentation détaillée
        │   ├── config.yaml            # Configuration complète
        │   ├── train_ppo.py           # Script d'entraînement
        │   ├── test_ppo.py            # Script de test
        │   ├── corridor_env.py        # Environnement Gymnasium (steering)
        │   ├── *.xml                  # Modèles MuJoCo
        │   ├── models/                # Checkpoints et métriques
        │   └── utils/                 # Modules utilitaires
        │
        ├── ppo_final/                 # Version finale (référence)
        ├── corridor_creation/         # Outils de création de corridors
        └── notebooks/                 # Notebooks d'expérimentation
```

---

## 🐳 Lancement de l'environnement

Le projet utilise Docker pour fournir un environnement complet avec:
- MuJoCo Desktop (interface graphique via noVNC)
- Jupyter Notebook
- Support GPU CUDA pour PyTorch
- Tous les packages Python nécessaires

### Option 1: Lancement via WSL (Windows)

**Avantages:**
- ✅ GPU CUDA fonctionnel (PyTorch/TensorFlow)
- ✅ Jupyter fonctionne correctement
- ✅ Performance optimale pour l'entraînement

**Lancement:**
```powershell
# Dans PowerShell
wsl

# Dans WSL
cd ~/rl/launch_scripts
./start.sh
```

**Dossier partagé:**
```
Linux (WSL): /home/VOTRE_USERNAME/rl/mujoco/workspace
Docker:      /home/student/workspace
Windows:     \\wsl.localhost\Ubuntu\home\VOTRE_USERNAME\rl\mujoco\workspace
```

### Option 2: Lancement direct depuis Windows

**Avantages:**
- ✅ Simple, un double-clic sur start.bat
- ✅ Dossier Windows natif (facile d'accès)

**Inconvénients:**
- ⚠️ GPU CUDA peut ne pas fonctionner
- ⚠️ Jupyter peut avoir des problèmes

**Lancement:**
```powershell
# Double-cliquer sur start.bat ou dans PowerShell:
cd C:\Users\VOTRE_USERNAME\rl\launch_scripts
.\start.bat
```

**Dossier partagé:**
```
Windows: C:\Users\VOTRE_USERNAME\rl\mujoco\workspace
Docker:  /home/student/workspace
```

### Option 3: Lancement Linux natif

**Lancement:**
```bash
cd ~/rl/launch_scripts
./start.sh
```

**Dossier partagé:**
```
Linux:  /home/VOTRE_USERNAME/rl/mujoco/workspace
Docker: /home/student/workspace
```

### Options de lancement

```bash
# Résolution personnalisée
./start.sh --resolution 2560x1440

# Mode économie de RAM
./start.sh --small_ram

# RAM personnalisée
./start.sh --ram 2g

# Qualité d'affichage
./start.sh --quality medium  # ou low

# Mode local (sans vérifier les mises à jour)
./start.sh --local

# Sans GPU (forcer software rendering)
./start.sh --no_gpu
```

### Accès à l'environnement

Une fois lancé, ouvrir dans votre navigateur:

- **Desktop noVNC**: http://localhost:6080
- **Jupyter Notebook**: http://localhost:8888

### Arrêter l'environnement

Dans le terminal où l'environnement tourne:
```
Ctrl+C
```

---

## ⚡ Utilisation rapide

### Dans l'environnement Docker

Une fois l'environnement lancé (voir section précédente), accéder au Desktop via http://localhost:6080

### Choisir une version

Le projet propose deux versions avec des méthodes de contrôle différentes:

#### ppo_no_steer - Contrôle 4 roues indépendantes
- Action space: 4 dimensions `[wheel_FL, wheel_FR, wheel_RL, wheel_RR]`
- Contrôle direct de chaque roue
- Plus de liberté mais plus difficile à apprendre

📖 **[Documentation complète: mujoco/workspace/ppo_no_steer/README.md](mujoco/workspace/ppo_no_steer/README.md)**

#### ppo_steer - Contrôle par volant et vitesse
- Action space: 2 dimensions `[steering_angle, speed]`
- Contrôle comme une voiture (plus naturel)
- Plus simple à apprendre

📖 **[Documentation complète: mujoco/workspace/ppo_steer/README.md](mujoco/workspace/ppo_steer/README.md)**

### Éditer les fichiers

**Depuis Windows:**
- Via WSL: `\\wsl.localhost\Ubuntu\home\VOTRE_USERNAME\rl\mujoco\workspace`
- Via Windows: `C:\Users\VOTRE_USERNAME\rl\mujoco\workspace` (si lancé avec start.bat)

**Depuis Linux:**
- `~/rl/mujoco/workspace`

**Avec VSCode:**
- Ouvrir le dossier workspace directement
- Les modifications sont synchronisées en temps réel avec Docker

---

## 🔍 Vérifier le support GPU

Dans Jupyter (http://localhost:8888) ou dans le terminal Docker:

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

Résultat attendu avec GPU:
```
Device: cuda:0
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.x
```

---

## 📊 Comparaison des méthodes de lancement

| Critère | WSL (start.sh) | Windows (start.bat) | Linux (start.sh) |
|---------|----------------|---------------------|------------------|
| GPU CUDA | ✅ Fonctionne | ❌ Peut échouer | ✅ Fonctionne |
| Jupyter | ✅ Fonctionne | ⚠️ Problèmes possibles | ✅ Fonctionne |
| Dossier partagé | `/home/USERNAME/rl/...` | `C:\Users\USERNAME\rl\...` | `/home/USERNAME/rl/...` |
| Simplicité | ⚠️ Terminal WSL | ✅ Double-clic | ✅ Terminal |
| Performance | ✅ Optimale | ⚠️ Moyenne | ✅ Optimale |
| Recommandé pour | Entraînement GPU | Tests rapides | Entraînement GPU |

---

## 🎯 Objectif

Le robot doit apprendre à naviguer dans un corridor de 100m avec:
- **Trous** (holes): Zones à éviter (chute = échec)
- **Bosses** (bumps): Obstacles qui ralentissent et pénalisent
- **Murs latéraux**: Limites du corridor (3m de large)

Le robot utilise:
- **Vision ego-centrique**: Grille CNN 2 canaux (obstacles, trous)
- **Historique de positions**: 8 frames passées pour anticipation
- **État du robot**: Position, vitesse, orientation

---

## 📊 Métriques d'entraînement

Les métriques suivantes sont trackées:
- **Return moyen**: Récompense cumulée par épisode
- **Distance moyenne**: Distance parcourue (objectif: 100m)
- **Taux de succès**: % d'épisodes atteignant 100m
- **Survie moyenne**: Nombre de steps avant terminaison
- **Raisons de terminaison**: fell, flipped, no_progress, success

---

## 🎓 Curriculum Learning

L'entraînement utilise un curriculum progressif:

1. **Phase 1**: Trous + 50% bosses (seuil: 10m)
2. **Phase 2**: Trous + 65% bosses (seuil: 12m)
3. **Phase 3**: Trous + 75% bosses (seuil: 65m)
4. **Phase 4**: Trous + 100% bosses (pas de seuil)

Le passage à la phase suivante se fait automatiquement quand la distance moyenne de l'itération dépasse le seuil.

---

## 🛠️ Configuration

Tous les paramètres sont configurables via `config.yaml` dans chaque dossier (ppo_no_steer, ppo_steer):
- Hyperparamètres PPO (learning rate, gamma, etc.)
- Architecture du réseau (CNN, MLP)
- Paramètres d'environnement (max_steps, vision)
- Curriculum learning (phases, seuils)
- Système de récompenses

Voir les READMEs spécifiques pour plus de détails.

---

## 🐛 Troubleshooting

### Docker ne démarre pas

**Windows:**
- Vérifier que WSL2 est installé: `wsl --status`
- Vérifier que Docker Desktop est lancé
- Vérifier l'intégration WSL2 dans Docker Desktop Settings

**Linux:**
- Vérifier que Docker est lancé: `sudo systemctl status docker`
- Vérifier que vous êtes dans le groupe docker: `groups`
- Si non: `sudo usermod -aG docker $USER` puis se déconnecter/reconnecter

### GPU non détecté

**Vérifier le driver NVIDIA:**
```bash
nvidia-smi
```

**Vérifier Docker GPU support:**
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

**Si ça ne fonctionne pas:**
- Installer NVIDIA Container Toolkit (voir section Installation)
- Redémarrer Docker: `sudo systemctl restart docker`
- Utiliser `./start.sh` depuis WSL (pas start.bat)

### Jupyter ne démarre pas

- Utiliser `./start.sh` depuis WSL au lieu de `start.bat`
- Vérifier les logs Docker
- Essayer avec `--small_ram` si peu de mémoire

### Fichiers non synchronisés

- Vérifier que vous éditez dans le bon dossier
- WSL: `\\wsl.localhost\Ubuntu\home\USERNAME\rl\mujoco\workspace`
- Windows: `C:\Users\USERNAME\rl\mujoco\workspace`
- Les modifications doivent apparaître immédiatement dans Docker

### Performance lente

- Vérifier que le GPU est bien utilisé (voir section "Vérifier le support GPU")
- Réduire `num_envs` dans config.yaml si manque de RAM
- Utiliser `--small_ram` au lancement
- Fermer les applications gourmandes en ressources

---

## 📝 Notes importantes

- Le dossier `rl/` est créé automatiquement par les scripts de lancement (va utiliser votre dossier rl cloné)
- Les checkpoints et métriques sont sauvegardés dans `models/` de chaque projet
- Les modifications de fichiers sont synchronisées en temps réel avec Docker
- Le GPU CUDA est essentiel pour un entraînement rapide (CPU = très lent)
- WSL2 + start.sh est la méthode recommandée sur Windows pour le GPU

---

## 📝 Licence

MIT

---

## 👥 Auteurs

Erwin PAZOS
