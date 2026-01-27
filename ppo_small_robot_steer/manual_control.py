"""
Contrôle manuel du robot avec les flèches du clavier.
Utilise le même environnement que l'IA pour tester manuellement.
"""
import numpy as np
import mujoco
from mujoco import viewer
import time
from corridor_env import CorridorEnv


class ManualController:
    def __init__(self):
        self.steering_angle = 0.0  # Angle de volant actuel
        self.speed = 0.0           # Vitesse actuelle
        self.max_steering = 30.0   # Angle max en degrés
        self.max_speed = 1.0       # Vitesse max en m/s
        
        # Paramètres de contrôle
        self.steering_increment = 30.0  # Incrément d'angle par pression (angle max en 1 clic)
        self.speed_increment = 0.2     # Incrément de vitesse par pression
        self.steering_decay = 0.70     # Décroissance automatique du volant (plus lente pour garder l'angle)
        self.speed_decay = 0.98        # Décroissance automatique de la vitesse
        
        print("🎮 CONTRÔLES:")
        print("  ↑ : Accélérer")
        print("  ↓ : Freiner/Reculer")
        print("  ← : Tourner à gauche")
        print("  → : Tourner à droite")
        print("  R : Reset environnement")
        print("  ESC : Quitter")
        print("  ESPACE : Arrêt d'urgence")
    
    def process_key(self, key):
        """Traiter les touches pressées."""
        if key == 265:  # Flèche haut
            self.speed = min(self.max_speed, self.speed + self.speed_increment)
            return True
        elif key == 264:  # Flèche bas
            self.speed = max(-self.max_speed, self.speed - self.speed_increment)
            return True
        elif key == 263:  # Flèche gauche
            self.steering_angle = self.max_steering  # Action maximale directement (comme l'IA)
            return True
        elif key == 262:  # Flèche droite
            self.steering_angle = -self.max_steering  # Action maximale directement (comme l'IA)
            return True
        elif key == 32:  # Espace - arrêt d'urgence
            self.speed = 0.0
            self.steering_angle = 0.0
            return True
        elif key == 82 or key == 114:  # R - reset
            return 'reset'
        elif key == 256:  # ESC - quitter
            return 'quit'
        
        return False
    
    def update(self):
        """Mise à jour automatique (décroissance)."""
        # Décroissance automatique du volant vers 0
        if abs(self.steering_angle) > 0.1:
            self.steering_angle *= self.steering_decay
        else:
            self.steering_angle = 0.0
        
        # Décroissance automatique de la vitesse vers 0
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
        return f"Volant: {self.steering_angle:+5.1f}° | Vitesse: {self.speed:+5.2f} m/s"


def main():
    print("🚗 CONTRÔLE MANUEL DU ROBOT")
    print("=" * 50)
    
    # Créer environnement
    env = CorridorEnv(max_steps=10000)  # Pas de limite de temps
    controller = ManualController()
    
    # Reset initial
    obs, info = env.reset()
    
    # Variables de suivi
    total_reward = 0.0
    step_count = 0
    
    # Variables pour les touches
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
                
                if quit_requested:
                    print("\n👋 Au revoir !")
                    return
                
                # Traiter les touches de contrôle du callback
                for key_code in list(keys_pressed):
                    if key_code in last_key_time:
                        if current_time - last_key_time[key_code] < 0.1:
                            continue
                    
                    result = controller.process_key(key_code)
                    if result:
                        last_key_time[key_code] = current_time
                        action_taken = True
                
                keys_pressed.clear()
                
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


if __name__ == "__main__":
    main()