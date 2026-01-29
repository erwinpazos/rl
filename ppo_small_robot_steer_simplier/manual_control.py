"""
Contrôle manuel du robot avec les flèches du clavier.
Utilise le même environnement que l'IA pour tester manuellement.
"""
import numpy as np
import mujoco
from mujoco import viewer
import time
import argparse
from corridor_env import CorridorEnv

# Import du générateur pour les corridors aléatoires
try:
    from corridor_generator_similar import CorridorGenerator
    CorridorGenerator = CorridorGenerator
except ImportError:
    print("⚠️  Générateur de corridor non disponible!")
    CorridorGenerator = None


def generate_random_corridor(seed=None):
    """Génère un corridor aléatoire temporaire."""
    if CorridorGenerator is None:
        print("❌ Générateur de corridor non disponible!")
        return None
    
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    print(f"🎲 Génération d'un corridor aléatoire (seed={seed})...")
    
    generator = CorridorGenerator()
    temp_filename = f"temp_manual_corridor_{seed}.xml"
    
    try:
        # Générer avec des paramètres variés
        length = np.random.uniform(80.0, 120.0)
        width = np.random.uniform(2.5, 3.5)
        
        generator.save_corridor(temp_filename, length, width, seed)
        
        # Statistiques
        bumps = generator.generate_bump_pattern(length, seed)
        holes = generator.generate_hole_pattern(length, seed)
        
        print(f"✅ Corridor généré: {length:.1f}m × {width:.1f}m")
        print(f"   {len(bumps)} bumps, {len(holes)} trous")
        
        return temp_filename
        
    except Exception as e:
        print(f"❌ Erreur génération corridor: {e}")
        return None


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


def main():
    # Arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Contrôle manuel du robot dans le corridor")
    parser.add_argument("--random", action="store_true", help="Générer un corridor aléatoire")
    parser.add_argument("--seed", type=int, help="Seed pour la génération aléatoire")
    parser.add_argument("--corridor", type=str, default="corridor_3x100_no_full_obstacles.xml", 
                       help="Fichier XML du corridor à utiliser")
    
    args = parser.parse_args()
    
    print("🚗 CONTRÔLE MANUEL DU ROBOT")
    print("=" * 50)
    
    # Déterminer le corridor à utiliser
    corridor_xml = args.corridor
    temp_file = None
    
    # Si --random OU --seed est fourni, générer un corridor
    if args.random or args.seed is not None:
        temp_file = generate_random_corridor(args.seed)
        if temp_file:
            corridor_xml = temp_file
        else:
            print("⚠️  Utilisation du corridor par défaut à la place")
    
    # Créer environnement avec le corridor choisi
    env = CorridorEnv(max_steps=10000, corridor_xml=corridor_xml)  # Pas de limite de temps
    controller = ManualController()
    
    print(f"🏁 Corridor utilisé: {corridor_xml}")
    
    # Reset initial
    obs, info = env.reset()
    
    # Variables de suivi
    total_reward = 0.0
    step_count = 0
    
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
            print("⚠️ Callbacks clavier non supportés - utilisez la console pour les commandes")
            v = viewer.launch_passive(env.model, env.data)
        
        with v:
            # Configuration caméra
            robot_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'robot')
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = robot_id
            v.cam.azimuth = 180
            v.cam.elevation = -20
            v.cam.distance = 8
            
            print(f"Position initiale: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}")
            print("Environnement prêt !")
            print("COMMANDES CONSOLE: w=avant, s=arrière, a=gauche, d=droite, r=reset, q=quit")
            
            last_key_time = {}
            
            while v.is_running():
                # Traiter les touches du callback (si disponible)
                action_taken = False
                current_time = time.time()
                
                # Traiter reset et quit
                if reset_requested:
                    print("\n🔄 RESET ENVIRONNEMENT")
                    obs, info = env.reset()
                    total_reward = 0.0
                    step_count = 0
                    controller.speed = 0.0
                    controller.steering_angle = 0.0
                    print(f"Nouvelle position: x={env.data.qpos[0]:.2f}, y={env.data.qpos[1]:.2f}")
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

                
                # Contrôle console alternatif (non-bloquant)
                try:
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        key = sys.stdin.read(1).lower()
                        if key == 'w':
                            controller.speed = min(controller.max_speed, controller.speed + controller.speed_increment)
                            action_taken = True
                        elif key == 's':
                            controller.speed = max(-controller.max_speed, controller.speed - controller.speed_increment)
                            action_taken = True
                        elif key == 'a':
                            controller.steering_angle = min(controller.max_steering, controller.steering_angle + controller.steering_increment)
                            action_taken = True
                        elif key == 'd':
                            controller.steering_angle = max(-controller.max_steering, controller.steering_angle - controller.steering_increment)
                            action_taken = True
                        elif key == ' ':
                            controller.speed = 0.0
                            controller.steering_angle = 0.0
                            action_taken = True
                        elif key == 'r':
                            reset_requested = True
                        elif key == 'q':
                            quit_requested = True
                except:
                    pass  # select non disponible sur Windows
                
                # Mise à jour automatique du contrôleur
                controller.update()
                
                # Obtenir action et faire un step
                action = controller.get_action()
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # Affichage périodique
                if step_count % 50 == 0 or action_taken:
                    x, y, z = env.data.qpos[:3]
                    status = controller.get_status()
                    print(f"Step {step_count:4d} | Pos: ({x:5.2f}, {y:5.2f}, {z:5.2f}) | {status} | Reward: {total_reward:6.1f}")
                
                # Vérifier fin d'épisode
                if terminated or truncated:
                    reason = info.get('reason', 'unknown')
                    final_x = env.data.qpos[0]
                    
                    print(f"\n🏁 ÉPISODE TERMINÉ: {reason}")
                    print(f"   Distance parcourue: {final_x:.2f}m")
                    print(f"   Steps: {step_count}")
                    print(f"   Reward total: {total_reward:.1f}")
                    print("   Tapez 'r' pour recommencer ou 'q' pour quitter")
                    
                    # Attendre une action de l'utilisateur
                    while v.is_running() and not reset_requested and not quit_requested:
                        v.sync()
                        time.sleep(0.01)
                    
                    if quit_requested:
                        print("\n👋 Au revoir !")
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
        
        # Nettoyer le fichier temporaire si créé
        if temp_file:
            try:
                import os
                os.remove(temp_file)
                print(f"🗑️  Fichier temporaire supprimé: {temp_file}")
            except Exception as e:
                print(f"⚠️  Impossible de supprimer {temp_file}: {e}")


if __name__ == "__main__":
    main()