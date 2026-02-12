"""
Utilitaires pour l'affichage de la vision du robot.
"""
import sys
import subprocess
import numpy as np


def check_and_install_display_dependencies():
    """Vérifie et installe tkinter et PIL si nécessaire."""
    missing_packages = []
    
    # Vérifier tkinter
    try:
        import tkinter as tk
        tk.Tk().destroy()  # Test rapide
    except (ImportError, Exception):
        missing_packages.append('tkinter')
    
    # Vérifier PIL/Pillow
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append('PIL')
    
    if not missing_packages:
        print("✓ Display dependencies OK (tkinter, PIL)")
        return True
    
    print(f"⚠ Missing display packages: {', '.join(missing_packages)}")
    print("Installing required packages...")
    
    try:
        # Installer les packages système
        print("Running: sudo apt update")
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        
        print("Running: sudo apt install python3-tk python3-pil.imagetk")
        subprocess.run(['sudo', 'apt', 'install', '-y', 'python3-tk', 'python3-pil.imagetk'], check=True)
        
        # Installer via pip
        print("Running: pip install pillow")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pillow'], check=True)
        
        print("✓ Display dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        print("Please install manually:")
        print("  sudo apt update")
        print("  sudo apt install python3-tk python3-pil.imagetk")
        print("  pip install pillow")
        return False
    except Exception as e:
        print(f"✗ Error during installation: {e}")
        return False


# Import après vérification
try:
    import tkinter as tk
    from PIL import Image, ImageTk
except ImportError:
    # Les imports échoueront mais la fonction check_and_install_display_dependencies() peut les installer
    pass


class VisionWindow:
    """Fenêtre tkinter pour afficher la vision CNN en temps réel."""
    
    def __init__(self, title='Vision CNN', vision_queue=None):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('1200x650')
        self.vision_queue = vision_queue
        
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
        
        log_title = tk.Label(log_frame, text='Episode Progress', font=('Arial', 12, 'bold'))
        log_title.pack()
        
        # Zone de texte avec scrollbar
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=8, yscrollcommand=scrollbar.set, 
                               font=('Courier', 9), bg='black', fg='lime')
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # Si une queue est fournie, démarrer le timer pour la checker
        if self.vision_queue is not None:
            self.check_queue()
    
    def add_log(self, message):
        """Ajoute un message au log."""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)  # Auto-scroll vers le bas
    
    def check_queue(self):
        """Vérifie la queue pour les mises à jour (si queue fournie)."""
        if self.vision_queue is None:
            return
        
        import queue
        try:
            data = self.vision_queue.get_nowait()
            if data is None:
                self.root.quit()
                return
            
            # Distinguer entre vision data et log message
            if isinstance(data, str):
                # C'est un message de log
                self.add_log(data)
            else:
                # C'est des données de vision
                grid, env_data = data
                self.display_grid(grid, env_data)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.check_queue)  # Check toutes les 50ms
    
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


def display_vision(obs, step, ret, env):
    """Afficher vision robot dans terminal."""
    print("\033[2J\033[H", end="")
    
    # Décoder observation: pos(3) + vel(3) + angle(1) + history + grid
    robot_state = obs[:7]  # pos(3) + vel(3) + angle(1)
    history_start = 7
    history = obs[history_start:history_start+env.history_dim].reshape(env.history_length, 6)
    grid = obs[history_start+env.history_dim:].reshape(env.grid_rows, env.grid_cols, 2)
    
    print("=" * 60)
    print(f"Step: {step} | Return: {ret:.1f}")
    print(f"Position: x={robot_state[0]:.2f}m, y={robot_state[1]:.2f}m, z={robot_state[2]:.2f}m")
    print(f"Velocity: vx={robot_state[3]:.2f}, vy={robot_state[4]:.2f}, vz={robot_state[5]:.2f}")
    print(f"Angle: {robot_state[6]:.2f}rad ({np.degrees(robot_state[6]):.1f}°)")
    print("=" * 60)
    print(f"\nVision {env.grid_rows}×{env.grid_cols}×2 EGO-CENTRIQUE (robot à ligne {env.robot_row_in_grid}):")
    print("Canal 0=Obstacles, Canal 1=Trous")
    print("-" * 50)
    
    # Afficher grille combinée (20 premières lignes max)
    display_rows = min(env.grid_rows, 20)
    for i in range(display_rows):
        relative_dist = (i - env.robot_row_in_grid) * env.cell_size
        line = f"{relative_dist:+.1f}m: "
        for j in range(min(env.grid_cols, 40)):  # Limiter colonnes
            # Combiner les 2 canaux pour affichage
            obstacle = grid[i, j, 0]
            trou = grid[i, j, 1]
            
            if obstacle > 0.5:
                line += '#'  # Obstacle (bump)
            elif trou > 0.5:
                line += '.'  # Trou
            else:
                line += '/'  # Sol navigable
        print(line)
    
    print("-" * 50)
    print("Légende: /=sol navigable  #=obstacle/bump  .=trou")
    print("2 canaux binaires: [obstacles, trous]")
    print(f"Vision: {env.vision_length:.1f}m x {env.vision_width:.1f}m, cellules de {env.cell_size}m")
    print("=" * 60)
