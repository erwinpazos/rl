"""
Contrôle manuel du robot avec les flèches du clavier.
Utilise le même environnement que l'IA pour tester manuellement.
"""
import numpy as np
import mujoco
from mujoco import viewer
import time
import argparse
import queue
import tkinter as tk
from PIL import Image, ImageTk
from corridor_env import CorridorEnv


class ManualController:
    def __init__(self):
        self.steering_angle = 0.0  # Angle de volant actuel
        self.speed = 0.0           # Vitesse actuelle
        self.max_steering = 30.0   # Angle max en degrés
        self.max_speed = 1.0       # Vitesse max en m/s
        
        # États des touches (comme dans ppo/simulation.py)
        self.is_turning_left = False
        self.is_turning_right = False
        self.is_accelerating = False
        self.is_braking = False
        
        # Paramètres de contrôle
        self.steering_increment = 30.0  # Incrément d'angle par pression (angle max en 1 clic)
        self.speed_increment = 0.2     # Incrément de vitesse par pression
        self.steering_decay = 0.70     # Décroissance automatique du volant (plus lente pour garder l'angle)
        self.speed_decay = 0.98        # Décroissance automatique de la vitesse
        
        print("🎮 CONTRÔLES:")
        print("  ↑ : Accélérer (toggle)")
        print("  ↓ : Freiner/Reculer (toggle)")
        print("  ← : Tourner à gauche (toggle ON/OFF)")
        print("  → : Tourner à droite (toggle ON/OFF)")
        print("  R : Reset environnement")
        print("  ESC : Quitter")
        print("  ESPACE : Arrêt d'urgence (reset tous les états)")
    
    def process_key_press(self, key):
        """Traiter les pressions de touches (comme ppo/simulation.py)."""
        if key == 265:  # Flèche haut
            self.is_accelerating = True
            self.is_braking = False
            return True
        elif key == 264:  # Flèche bas
            self.is_braking = True
            self.is_accelerating = False
            return True
        elif key == 263:  # Flèche gauche
            if self.is_turning_left:
                # Si déjà en train de tourner à gauche, arrêter
                self.is_turning_left = False
            else:
                # Sinon, commencer à tourner à gauche
                self.is_turning_left = True
                self.is_turning_right = False
            return True
        elif key == 262:  # Flèche droite
            if self.is_turning_right:
                # Si déjà en train de tourner à droite, arrêter
                self.is_turning_right = False
            else:
                # Sinon, commencer à tourner à droite
                self.is_turning_right = True
                self.is_turning_left = False
            return True
        elif key == 32:  # Espace - arrêt d'urgence
            self.is_accelerating = False
            self.is_braking = False
            self.is_turning_left = False
            self.is_turning_right = False
            self.speed = 0.0
            self.steering_angle = 0.0
            return True
        elif key == 82 or key == 114:  # R - reset
            return 'reset'
        elif key == 256:  # ESC - quitter
            return 'quit'
        
        return False
    
    def update(self):
        """Mise à jour continue basée sur les états (comme ppo/simulation.py)."""
        # Steering basé sur l'état des touches
        if self.is_turning_left:
            self.steering_angle = self.max_steering  # Action +1.0 continue
        elif self.is_turning_right:
            self.steering_angle = -self.max_steering  # Action -1.0 continue
        else:
            self.steering_angle = 0.0  # Pas de rotation
        
        # Vitesse basée sur l'état des touches
        if self.is_accelerating:
            self.speed = min(self.max_speed, self.speed + self.speed_increment * 0.1)
        elif self.is_braking:
            self.speed = max(-self.max_speed, self.speed - self.speed_increment * 0.1)
        else:
            # Décroissance naturelle
            if abs(self.speed) > 0.05:
                self.speed *= self.speed_decay
            else:
                self.speed = 0.0
    
    def get_action(self):
        """Convertir les commandes en action pour l'environnement."""
        # Normaliser pour l'environnement (±1.0)
        steering_normalized = self.steering_angle / self.max_steering
        speed_normalized = self.speed / self.max_speed
        
        return np.array([steering_normalized, speed_normalized], dtype=np.float32)
    
    def get_status(self):
        """Obtenir le statut actuel pour affichage."""
        states = []
        if self.is_turning_left: states.append("←")
        if self.is_turning_right: states.append("→")
        if self.is_accelerating: states.append("↑")
        if self.is_braking: states.append("↓")
        state_str = "".join(states) if states else "○"
        
        return f"État: {state_str} | Volant: {self.steering_angle:+5.1f}° | Vitesse: {self.speed:+5.2f} m/s"
    
    def get_action(self):
        """Convertir les commandes en action pour l'environnement."""
        # Normaliser pour l'environnement (±1.0)
        steering_normalized = self.steering_angle / self.max_steering
        speed_normalized = self.speed / self.max_speed
        
        return np.array([steering_normalized, speed_normalized], dtype=np.float32)
    
    def get_status(self):
        """Obtenir le statut actuel pour affichage."""
        return f"Volant: {self.steering_angle:+5.1f}° | Vitesse: {self.speed:+5.2f} m/s"


def load_config(config_path="config.yaml"):
    """Charge la configuration depuis un fichier YAML."""
    import yaml
    import os
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config file {config_path} not found, using default values")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return None


class VisionWindow:
    """Fenêtre tkinter pour afficher la vision CNN en temps réel."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Vision CNN - Contrôle Manuel')
        self.root.geometry('1200x650')
        
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les 3 vues (en haut)
        vision_frame = tk.Frame(main_frame)
        vision_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        
        # 3 colonnes pour les 3 vues
        self.frames = []
        self.labels = []
        self.titles = ['Canal 0 - Obstacles', 'Canal 1 - Trous', 'Vue Combinée']
        
        for i, title in enumerate(self.titles):
            frame = tk.Frame(vision_frame)
            frame.grid(row=0, column=i, padx=10, pady=10)
            
            title_label = tk.Label(frame, text=title, font=('Arial', 12, 'bold'))
            title_label.pack()
            
            img_label = tk.Label(frame)
            img_label.pack()
            
            self.frames.append(frame)
            self.labels.append(img_label)
        
        # Frame pour les logs (en bas)
        log_frame = tk.Frame(main_frame)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        log_title = tk.Label(log_frame, text='Status & Controls', font=('Arial', 12, 'bold'))
        log_title.pack()
        
        # Zone de texte avec scrollbar
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=8, yscrollcommand=scrollbar.set, 
                               font=('Courier', 9), bg='black', fg='lime')
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
    
    def add_log(self, message):
        """Ajoute un message au log."""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)  # Auto-scroll vers le bas
    
    def display_grid(self, grid, env):
        """Affiche la grille de vision."""
        rows, cols = grid.shape[0], grid.shape[1]
        robot_row = env.robot_row_in_grid
        robot_col = env.robot_col_in_grid
        
        # Canal 0 - Obstacles (rouge)
        img0 = self.grid_to_image(grid[:, :, 0], robot_row, robot_col, (255, 0, 0))
        photo0 = ImageTk.PhotoImage(img0.resize((380, 380), Image.NEAREST))
        self.labels[0].config(image=photo0)
        self.labels[0].image = photo0  # Garder référence
        
        # Canal 1 - Trous (bleu)
        img1 = self.grid_to_image(grid[:, :, 1], robot_row, robot_col, (0, 0, 255))
        photo1 = ImageTk.PhotoImage(img1.resize((380, 380), Image.NEAREST))
        self.labels[1].config(image=photo1)
        self.labels[1].image = photo1
        
        # Vue combinée
        img2 = self.grid_combined_to_image(grid, robot_row, robot_col)
        photo2 = ImageTk.PhotoImage(img2.resize((380, 380), Image.NEAREST))
        self.labels[2].config(image=photo2)
        self.labels[2].image = photo2
    
    def grid_to_image(self, channel, robot_row, robot_col, color):
        """Convertit un canal en image."""
        rows, cols = channel.shape
        img_data = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        # Obstacles en couleur
        mask = channel > 0.5
        img_data[mask] = color
        img_data[~mask] = [255, 255, 255]
        
        # Robot en vert
        if 0 <= robot_row < rows and 0 <= robot_col < cols:
            img_data[robot_row, robot_col] = [0, 255, 0]
        
        return Image.fromarray(img_data, 'RGB')
    
    def grid_combined_to_image(self, grid, robot_row, robot_col):
        """Crée une vue combinée des deux canaux."""
        rows, cols = grid.shape[0], grid.shape[1]
        img_data = np.ones((rows, cols, 3), dtype=np.uint8) * 255
        
        for i in range(rows):
            for j in range(cols):
                obstacle = grid[i, j, 0]
                hole = grid[i, j, 1]
                
                if obstacle > 0.5 and hole > 0.5:
                    img_data[i, j] = [128, 0, 128]  # Purple
                elif obstacle > 0.5:
                    img_data[i, j] = [255, 0, 0]  # Rouge
                elif hole > 0.5:
                    img_data[i, j] = [0, 0, 255]  # Bleu
        
        # Robot en vert
        if 0 <= robot_row < rows and 0 <= robot_col < cols:
            img_data[robot_row, robot_col] = [0, 255, 0]
        
        return Image.fromarray(img_data, 'RGB')
    
    def update(self):
        """Met à jour la fenêtre."""
        self.root.update()
    
    def is_running(self):
        """Vérifie si la fenêtre est toujours ouverte."""
        try:
            self.root.winfo_exists()
            return True
        except:
            return False


def load_config(config_path="config.yaml"):
    """Charge la configuration depuis un fichier YAML."""
    import yaml
    import os
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config file {config_path} not found, using default values")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return None


def main():
    # Arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Contrôle manuel du robot dans le corridor")
    parser.add_argument("--seed", type=int, help="Seed pour la génération aléatoire (force random corridor)")
    parser.add_argument("--fixed", action="store_true", help="Utiliser un corridor fixe au lieu d'aléatoire")
    parser.add_argument("--bump", type=float, default=None, help="Ratio de bumps (0.0 à 1.0, ex: 1.0 = 100%% bumps)")
    
    args = parser.parse_args()
    
    print("🚗 CONTRÔLE MANUEL DU ROBOT")
    print("=" * 50)
    
    # Charger la configuration
    config = load_config("config.yaml")
    
    # Déterminer les paramètres depuis le config
    if config and 'environment' in config:
        env_config = config['environment']
        max_steps = env_config.get('max_steps', 10000)
        use_random = not args.fixed  # Par défaut random, sauf si --fixed
    else:
        max_steps = 10000
        use_random = not args.fixed
    
    # Déterminer le bump_ratio
    if args.bump is not None:
        bump_ratio = args.bump
        print(f"💥 Bump ratio: {bump_ratio*100:.0f}% (depuis --bump)")
    elif config and 'curriculum' in config:
        # Utiliser le bump_ratio de la phase 1 du curriculum
        bump_schedule = config['curriculum'].get('bump_ratio_schedule', [])
        bump_ratio = 1.0  # Défaut
        for phase in bump_schedule:
            if phase['phase'] == 1:
                bump_ratio = phase['bump_ratio']
                break
        print(f"💥 Bump ratio: {bump_ratio*100:.0f}% (depuis config.yaml phase 1)")
    else:
        bump_ratio = 1.0
        print(f"💥 Bump ratio: {bump_ratio*100:.0f}% (défaut)")
    
    # Créer environnement avec corridor aléatoire (comme dans l'entraînement)
    if use_random:
        print("🎲 Mode: Corridor ALÉATOIRE (comme l'entraînement)")
        if args.seed is not None:
            print(f"   Seed forcé: {args.seed}")
        env = CorridorEnv(
            max_steps=max_steps, 
            corridor_xml=None,  # None = génération aléatoire
            use_fixed_seed=(args.seed is not None)
        )
        # Configurer le bump_ratio
        env.bump_ratio = bump_ratio
        
        # Si seed fourni, forcer la régénération avec ce seed
        if args.seed is not None:
            env.env_random = np.random.RandomState(args.seed)
            env.reset()
    else:
        print("📋 Mode: Corridor FIXE")
        env = CorridorEnv(
            max_steps=max_steps,
            corridor_xml=None,
            use_fixed_seed=True
        )
        # Configurer le bump_ratio
        env.bump_ratio = bump_ratio
    
    controller = ManualController()
    
    # Afficher les paramètres depuis le config
    if config:
        if 'robot' in config:
            robot_config = config['robot']
            controller.max_steering = robot_config.get('max_steering_angle', 30.0)
            controller.max_speed = robot_config.get('max_speed', 1.0)
            print(f"🎮 Contrôles: Steering max={controller.max_steering}°, Speed max={controller.max_speed} m/s")
        
        if 'corridor' in config:
            corridor_config = config['corridor']
            print(f"🏁 Corridor: {corridor_config.get('corridor_length', 110)}m × {corridor_config.get('corridor_width', 3)}m")
            print(f"   Objectif: {corridor_config.get('success_distance', 100)}m")
    
    # Reset initial
    obs, info = env.reset()
    
    # Variables de suivi
    total_reward = 0.0
    step_count = 0
    
    # Créer la fenêtre de vision
    vision_window = VisionWindow()
    vision_window.add_log("🚗 CONTRÔLE MANUEL DU ROBOT")
    vision_window.add_log("=" * 50)
    vision_window.add_log("🎮 CONTRÔLES:")
    vision_window.add_log("  ↑ : Accélérer (toggle)")
    vision_window.add_log("  ↓ : Freiner/Reculer (toggle)")
    vision_window.add_log("  ← : Tourner à gauche (toggle)")
    vision_window.add_log("  → : Tourner à droite (toggle)")
    vision_window.add_log("  R : Reset environnement")
    vision_window.add_log("  ESC : Quitter")
    vision_window.add_log("  ESPACE : Arrêt d'urgence")
    vision_window.add_log("=" * 50)
    vision_window.add_log(f"Position initiale: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}")
    
    # Variables pour les touches (approche simplifiée comme ppo/simulation.py)
    keys_pressed = set()
    reset_requested = False
    quit_requested = False
    
    def key_callback(keycode):
        """Callback pour les touches pressées."""
        nonlocal reset_requested, quit_requested
        keys_pressed.add(keycode)
        
        if keycode == 82 or keycode == 114:  # R
            reset_requested = True
        elif keycode == 256:  # ESC
            quit_requested = True
    
    try:
        # Essayer avec key_callback d'abord
        try:
            v = viewer.launch_passive(env.model, env.data, key_callback=key_callback)
        except TypeError:
            # Si key_callback n'est pas supporté, utiliser sans
            vision_window.add_log("⚠️ Callbacks clavier non supportés")
            v = viewer.launch_passive(env.model, env.data)
        
        with v:
            # Configuration caméra
            robot_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = robot_id
            v.cam.azimuth = 180
            v.cam.elevation = -20
            v.cam.distance = 8
            
            vision_window.add_log("✅ Environnement prêt !")
            
            last_key_time = {}
            
            while v.is_running() and vision_window.is_running():
                # Traiter les touches du callback (si disponible)
                action_taken = False
                current_time = time.time()
                
                # Traiter reset et quit
                if reset_requested:
                    vision_window.add_log("\n🔄 RESET ENVIRONNEMENT")
                    obs, info = env.reset()
                    total_reward = 0.0
                    step_count = 0
                    controller.speed = 0.0
                    controller.steering_angle = 0.0
                    vision_window.add_log(f"Nouvelle position: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}")
                    reset_requested = False
                    continue
                
                # Traiter les pressions de touches (comme ppo/simulation.py)
                for key_code in list(keys_pressed):
                    if key_code in last_key_time:
                        if current_time - last_key_time[key_code] < 0.1:
                            continue
                    
                    result = controller.process_key_press(key_code)
                    if result:
                        last_key_time[key_code] = current_time
                        action_taken = True
                
                keys_pressed.clear()
                
                # Mise à jour continue des états (comme ppo/simulation.py)
                controller.update()
                
                # Obtenir action et faire un step
                action = controller.get_action()
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # Mettre à jour la vision CNN toutes les 5 steps
                if step_count % 5 == 0:
                    try:
                        grid = obs[7+env.history_dim:].reshape(env.grid_rows, env.grid_cols, 2)
                        vision_window.display_grid(grid, env)
                        vision_window.update()
                    except Exception as e:
                        vision_window.add_log(f"⚠️ Vision error: {e}")
                
                # Affichage périodique dans la fenêtre
                if step_count % 25 == 0 or action_taken:
                    x, y, z = env.data.qpos[:3]
                    status = controller.get_status()
                    log_msg = f"Step {step_count:4d} | Pos: ({x:5.2f}, {y:5.2f}, {z:5.2f}) | {status} | Reward: {total_reward:6.1f}"
                    vision_window.add_log(log_msg)
                
                # Vérifier fin d'épisode
                if terminated or truncated:
                    reason = info.get('reason', 'unknown')
                    final_x = env.data.qpos[0]
                    
                    vision_window.add_log(f"\n🏁 EPISODE ENDED: {reason}")
                    vision_window.add_log(f"   Distance: {final_x:.2f}m")
                    vision_window.add_log(f"   Steps: {step_count}")
                    vision_window.add_log(f"   Total reward: {total_reward:.1f}")
                    vision_window.add_log("   Press 'r' to restart or 'q' to quit")
                    
                    # Attendre une action de l'utilisateur
                    while v.is_running() and vision_window.is_running() and not reset_requested and not quit_requested:
                        v.sync()
                        vision_window.update()
                        time.sleep(0.01)
                    
                    if quit_requested:
                        vision_window.add_log("\n👋 Au revoir !")
                        return
                
                # Synchroniser le viewer
                v.sync()
                time.sleep(0.02)  # 50 FPS
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption clavier - Arrêt du programme")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n👋 Au revoir !")


if __name__ == "__main__":
    main()