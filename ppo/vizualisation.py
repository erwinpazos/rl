import mujoco
from mujoco import viewer
import numpy as np
import xml.etree.ElementTree as ET
import time
import json

import numpy as np
import random

class CinematicCamera:
    """
    Gère une caméra cinématique "drone" avec une séquence d'intro
    et des modes de prise de vue aléatoires.
    """
    def __init__(self, robot_body_id, alpha=0.05):
        self.robot_body_id = robot_body_id
        
        # --- Paramètres de lissage ---
        # (0.02 = très lent, 0.1 = réactif. 0.05 est un bon équilibre)
        self.alpha = alpha  
        self.prev_pos = None
        self.prev_lookat = None
        
        # --- Définition de l'intro (basée sur ton code) ---
        self.intro_phase1_duration = 6.5  # Durée du zoom (130 * 0.05s)
        self.intro_phase2_duration = 1.8  # Durée du travelling (90 * 0.02s)
        self.intro_total_duration = self.intro_phase1_duration + self.intro_phase2_duration # 8.3s
        
        # --- Machine d'état pour les modes ---
        self.random_modes = ['CHASE_CLOSE', 'CHASE_FAR', 'HIGH_ANGLE_CHASE', 'SIDE_STRAFE']
        
        # On COMMENCE TOUJOURS par le mode 'INTRO'
        self.current_mode = 'INTRO'
        self.time_in_mode = 0.0
        self.current_mode_duration = self.intro_total_duration
        
        print(f"Caméra Cinématique: Mode initial = INTRO (durée {self.intro_total_duration:.1f}s)")

    def _select_new_mode(self):
        """Choisit un nouveau mode aléatoire (après l'intro)."""
        # Choisit un mode dans la liste des modes aléatoires
        self.current_mode = random.choice(self.random_modes)
        self.time_in_mode = 0.0
        self.current_mode_duration = random.uniform(8.0, 15.0)
        print(f"\nCaméra Cinématique: Changement -> {self.current_mode} (pour {self.current_mode_duration:.1f}s)")

    def get_params(self, d, timestep):
        """
        Calcule et retourne les paramètres de la caméra MuJoCo pour ce pas de temps.
        """
        # --- 1. Mise à jour de l'état ---
        self.time_in_mode += timestep
        if self.time_in_mode > self.current_mode_duration:
            # Si le mode est fini (soit l'intro, soit un mode aléatoire)
            # on en choisit un nouveau.
            self._select_new_mode()

        # --- 2. Obtenir la position/orientation du robot (Base) ---
        robot_pos = d.xpos[self.robot_body_id].copy()
        robot_mat = d.xmat[self.robot_body_id].reshape(3, 3)
        forward_vec = robot_mat[:, 0] # Axe X local (Avant)
        right_vec   = robot_mat[:, 1] # Axe Y local (Droite)
        up_vec      = robot_mat[:, 2] # Axe Z local (Haut)

        # --- 3. Définir les CIBLES (pos & lookat) selon le mode ---
        
        target_cam_pos = robot_pos # Initialisation
        target_lookat = robot_pos  # Initialisation
        if self.current_mode == 'INTRO':
            # Séquence d'intro en 2 phases
            if self.time_in_mode < self.intro_phase1_duration:
                # --- Phase 1: Long zoom avant (6.5s) ---
                progress = self.time_in_mode / self.intro_phase1_duration
                
                # Interpolation de la distance (de 135m à 5m)
                current_dist = 135.0 * (1.0 - progress) + 5.0 * progress
                
                # --- CORRECTION : On commence de plus haut ---
                current_az_rad = np.radians(180.0)  # Toujours en face
                current_el_rad = np.radians(2.0) # <--- CHANGÉ (était -1.0)
                # --- FIN CORRECTION ---
                
                target_lookat = robot_pos  # Regarde le robot
            
            else:
                # --- Phase 2: Travelling "Grue" (1.8s) ---
                t_phase2 = self.time_in_mode - self.intro_phase1_duration
                progress = t_phase2 / self.intro_phase2_duration
                
                # --- CORRECTION : Transition douce de -15 à -20 ---
                # Azimuth: 180 -> 0
                current_az_rad = np.radians(180.0 * (1.0 - progress)) 
                # Elevation: -15 -> -20
                current_el_rad = np.radians(2* (1.0 - progress) + 4 * progress) # <--- CHANGÉ (était -1.0)
                # Distance: 5 -> 8
                current_dist = 5.0 * (1.0 - progress) + 8.0 * progress
                # --- FIN CORRECTION ---
                
                # Le point de visée bouge aussi, de "sur le robot" à "devant le robot"
                target_lookat = robot_pos + (progress * 2.0) * forward_vec

            # Calcul de la position CIBLE de la caméra (inchangé)
            cam_pos_local = np.array([
                np.cos(current_az_rad) * np.cos(current_el_rad),
                np.sin(current_az_rad) * np.cos(current_el_rad),
                -np.sin(current_el_rad)
            ])
            cam_pos_local *= -current_dist
            target_cam_pos = robot_pos + robot_mat.dot(cam_pos_local)
        elif self.current_mode == 'CHASE_CLOSE':
            cam_offset = -3.0 * forward_vec + 1.2 * up_vec
            target_cam_pos = robot_pos + cam_offset
            target_lookat = robot_pos + 2.0 * forward_vec
        
        elif self.current_mode == 'CHASE_FAR':
            cam_offset = -8.0 * forward_vec + 3.0 * up_vec
            target_cam_pos = robot_pos + cam_offset
            target_lookat = robot_pos + 3.0 * forward_vec
        elif self.current_mode == 'HIGH_ANGLE_CHASE':
            # Nouveau mode: Poursuite de loin et en hauteur (type drone)
            cam_offset = -10.0 * forward_vec + 7.0 * up_vec # 10m derrière, 7m au-dessus
            target_cam_pos = robot_pos + cam_offset
            target_lookat = robot_pos + 3.0 * forward_vec # Regarde 3m devant le robot
        elif self.current_mode == 'TOP_DOWN':
            cam_offset = 7.0 * up_vec - 1.0 * forward_vec
            target_cam_pos = robot_pos + cam_offset
            target_lookat = robot_pos - 0.5 * forward_vec

        elif self.current_mode == 'SIDE_STRAFE':
            cam_offset = 5.0 * right_vec + 1.5 * up_vec # 5m à droite
            target_cam_pos = robot_pos + cam_offset
            target_lookat = robot_pos

        # --- 4. Lissage (le cœur de l'effet "drone") ---
        if self.prev_pos is None:
            self.prev_pos = target_cam_pos
            self.prev_lookat = target_lookat

        # La position/lookat "réelle" de la caméra suit la "cible" avec un amorti
        smooth_cam_pos = self.prev_pos * (1 - self.alpha) + target_cam_pos * self.alpha
        smooth_lookat = self.prev_lookat * (1 - self.alpha) + target_lookat * self.alpha
        
        self.prev_pos = smooth_cam_pos
        self.prev_lookat = smooth_lookat

        # --- 5. Conversion en paramètres MuJoCo ---
        vec = smooth_lookat - smooth_cam_pos
        distance = np.linalg.norm(vec)
        if distance < 1e-6: 
            distance = 1e-6
            
        azimuth = np.degrees(np.arctan2(vec[1], vec[0]))
        elevation = np.degrees(np.arcsin(vec[2] / distance))

        return distance, azimuth, elevation, smooth_lookat 
command_log = [] 

# Global viewer reference for keyboard callback
viewer_instance = None

def extract_corridor_from_xml(xml_file_path):
    print(f"Extracting corridor components from {xml_file_path}...")
    # Parse the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract useful components
    components = {
        'compiler': None,
        'option': None,
        'default': None,
        'asset': None,
        'corridor_geom': None,
        'actuators': None
    }
    
    # Find and extract each component
    for child in root:
        if child.tag == 'compiler':
            components['compiler'] = child
        elif child.tag == 'option':
            components['option'] = child
        elif child.tag == 'default':
            components['default'] = child
        elif child.tag == 'asset':
            components['asset'] = child
        elif child.tag == 'worldbody':
            components['corridor_geom'] = child
        elif child.tag == 'actuator':
            components['actuators'] = child
    
    return components

def extract_robot_from_xml(xml_file_path):
    """
    Extract robot components from existing XML file.
    This teaches students how to parse and reuse XML components.
    """
    print(f"Extracting robot components from {xml_file_path}...")
    
    # Parse the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract useful components
    components = {
        'compiler': None,
        'option': None,
        'default': None,
        'asset': None,
        'robot_body': None,
        'actuators': None,
        'visual':None
    }
    
    # Find and extract each component
    for child in root:
        if child.tag == 'compiler':
            components['compiler'] = child
        elif child.tag == 'option':
            components['option'] = child
        elif child.tag == 'default':
            components['default'] = child
        elif child.tag == 'asset':
            components['asset'] = child
        elif child.tag == 'worldbody':
            # Extract the robot body from worldbody
            for body in child:
                if body.get('name') == 'robot':
                    components['robot_body'] = body
        elif child.tag == 'actuator':
            components['actuators'] = child
        elif child.tag == 'visual':
            components['visual'] = child
    return components

def build_combined_model(robot_components, corridor_components,  robot_height=1.0):
    print("Building combined model...")
    print(f"Robot starting height: {robot_height}m above floor")
    
    # Create root mujoco element
    root = ET.Element('mujoco')
    root.set('model', 'robot_with_programmatic_floor')
    
    # Add compiler settings from robot XML
    if robot_components['compiler'] is not None:
        root.append(robot_components['compiler'])
    
    # CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS (override robot's settings)
    option = ET.Element('option')
    option.set('timestep', '0.01')
    ## Ex. 01: Add gravity
    option.set('gravity', '0 0 -9.81')  # Standard Earth gravity
    # CORRECT
    option.set('solver', 'Newton')      # Garde Newton pour les contacts
    option.set('integrator', 'RK4')   # Ajoute RK4 pour l'intégration du temps
    option.set('iterations', '50')
    root.append(option)
    print("  Environment physics: gravity enabled, timestep=0.01s")
    
    # Add size settings
    size = ET.Element('size')
    size.set('njmax', '1000')
    size.set('nconmax', '500')
    root.append(size)
    
    if robot_components['default'] is not None:
        root.append(robot_components['default'])
    if robot_components['visual'] is not None:
        root.append(robot_components['visual'])
    
    # Create asset section with textures and enhanced materials
    asset = ET.Element('asset')
    
    added_material_names = set()
    
    # Also keep any original materials from robot XML if they exist
    if robot_components['asset'] is not None:
        for original_material in robot_components['asset']:
            # Only add if it's not already in our enhanced materials
            material_name = original_material.get('name', '')
            if material_name not in added_material_names:
                asset.append(original_material)
                added_material_names.add(material_name)
    if corridor_components['asset'] is not None:
        for original_material in corridor_components['asset']:
            # Only add if it's not already in our enhanced materials
            material_name = original_material.get('name', '')
            if material_name not in added_material_names:
                asset.append(original_material)
                added_material_names.add(material_name)
    
    root.append(asset)
    
    # Create worldbody with floor and robot
    worldbody = ET.Element('worldbody')


    # Add robot body with enhanced visuals and adjusted height
    if robot_components['robot_body'] is not None:
        # Adjust robot starting height (floor is at -0.1, so robot center should be at robot_height - 0.1)
        robot_z_position = robot_height - 0.1  # Floor offset
        current_pos = robot_components['robot_body'].get('pos', '0 0 0.2')
        pos_parts = current_pos.split()
        if len(pos_parts) == 3:
            new_pos = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
            robot_components['robot_body'].set('pos', new_pos)
            print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")
        
        worldbody.append(robot_components['robot_body'])

    if corridor_components['corridor_geom'] is not None:
        for geom in corridor_components['corridor_geom']:
             worldbody.append(geom)

    root.append(worldbody)
    
    # Add actuators
    if robot_components['actuators'] is not None:
        root.append(robot_components['actuators'])
    
    # Convert to XML string
    xml_string = ET.tostring(root, encoding='unicode')
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# Extract robot from existing XML file
robot_components = extract_robot_from_xml("four_wheels_robot.xml")

corridor_components = extract_corridor_from_xml("corridor_3x100.xml")

h = 0.45  # Robot will start 2 meters above floor and fall down

print(f"=== PHYSICS DEMONSTRATION ===")
print(f"Robot will start {h}m above the floor")
print("Watch it fall due to gravity - this proves environment physics work!")
print()

# Build combined model with enhanced visuals and physics
m_combined = build_combined_model(robot_components, corridor_components, robot_height=h)
d = mujoco.MjData(m_combined)

# For convenience, use shorter variable name in the rest of the code
m = m_combined

# Important : re-synchroniser
mujoco.mj_forward(m, d)

# Initialize simulation with zero velocities for stability
d.qvel[:] = 0  # Zero all velocities
d.qacc[:] = 0  # Zero all accelerations

# See what DOF 0 represents:
print("\n=== MODEL DEBUG INFO ===")
print(f"Total DOFs: {m.nv}")
print(f"Joint names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}")
print(f"DOF names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_DOF, i) for i in range(m.nv)]}")
print(f"First few qpos: {d.qpos[:7]}")  # First 7 DOFs (free joint = 7: 3 pos + 4 quat)
print(f"First few qvel: {d.qvel[:6]}")  # First 6 velocities (free joint = 6: 3 linear + 3 angular)

print("\n=== GRAVITY & PHYSICS DEBUG ===")
print(f"Gravity setting: {m.opt.gravity}")
print(f"Timestep: {m.opt.timestep}")
print(f"Solver: {m.opt.solver}")

print("\n=== GEOMETRY DEBUG ===")
print(f"Total geometries: {m.ngeom}")
for i in range(m.ngeom):
    geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    geom_type = m.geom_type[i]
    geom_size = m.geom_size[i]
    geom_pos = m.geom_pos[i]
    print(f"  Geom {i}: {geom_name}, type={geom_type}, size={geom_size}, pos={geom_pos}")
print("\n")

# Check if wheels are touching the floor:

print("=== CONTACT DEBUG ===")
print(f"Number of contacts: {d.ncon}")
for i in range(d.ncon):
    contact = d.contact[i]
    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    print(f"Contact {i}: {geom1_name} <-> {geom2_name}")
print("\n")

def set_wheel_speeds(d, w_fl, w_fr, w_rl, w_rr, key=None):
    """Mode terrestre - roues actives"""
    max_speed = 100
    
    # Mettre TOUT à zéro d'abord
    d.ctrl[:] = 0
    
    # Activer SEULEMENT les roues (indices 0-3)
    d.ctrl[0] = np.clip(w_fl, -max_speed, max_speed)
    d.ctrl[1] = np.clip(w_fr, -max_speed, max_speed)
    d.ctrl[2] = np.clip(w_rl, -max_speed, max_speed)
    d.ctrl[3] = np.clip(w_rr, -max_speed, max_speed)

set_wheel_speeds(d, 0, 0, 0, 0)
robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')

def restore_initial_state(d, state):
    d.qpos[:] = np.array(state["qpos"])
    d.qvel[:] = np.array(state["qvel"])
    if state.get("act"):
        d.act[:] = np.array(state["act"])
    mujoco.mj_forward(m, d)

with open("commands_log.json", "r") as f:
    payload = json.load(f)

command_log = payload["commands"]
initial_state = payload["initial_state"]
timestep = payload.get("timestep", m.opt.timestep)

# 🎬 Replay avec rendu vidéo

import imageio.v3 as iio
import sys


real_duration = payload["real_time"]
duration = command_log[-1]["t"] + 1

print(f"Temps simulé : {duration:.2f}s")
print(f"Temps réel   : {real_duration:.2f}s")

# --- paramètres ---
FPS = 60                         # cadence finale de la vidéo (10 fps)
speedup = duration / real_duration
height, width = 720, 1280
print(f"⏩ Accélération ×{speedup:.2f} — sortie à {FPS} fps ({real_duration:.1f}s)")

# --- Caméra MuJoCo ---
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

frames = []
idx = 0
next_capture_t = 0.0              # prochain instant simulé où capturer une image
dt_capture = duration / (real_duration * FPS)  # intervalle simulé entre deux frames
print(f"Chaque frame représente {dt_capture:.3f} s simulées.")

restore_initial_state(d, initial_state)

prev_lookat, prev_pos = None, None

cinematic_cam = CinematicCamera(robot_body_id, alpha=0.02)
# --- Rendu ---
with mujoco.Renderer(m, height, width) as renderer:
    while d.time < duration:
        mujoco.mj_step(m, d)

        while (idx < len(command_log) and
               d.time >= (command_log[idx]["t"])):
            
            w = command_log[idx]["wheels"]
            set_wheel_speeds(d, *w)
            idx += 1 # On passe à la commande suivante dans le log
        
        # Caméra cinématique fluide
        dist, az, el, lookat = cinematic_cam.get_params(d, m.opt.timestep)
        # capturer quand on a dépassé le temps prévu
        if d.time >= next_capture_t:
            cam.distance = dist
            cam.azimuth = az
            cam.elevation = el
            cam.lookat = lookat
            renderer.update_scene(d, cam)
            frame = renderer.render()
            frames.append(frame)

            # mise à jour de la barre de progression
            progress = min(d.time / duration, 1.0)
            bar = "█" * int(progress * 40) + "-" * int((1 - progress) * 40)
            sys.stdout.write(f"\rRendering: |{bar}| {progress*100:5.1f}% ({d.time:.2f}/{duration:.2f}s)")
            sys.stdout.flush()

            next_capture_t += dt_capture  # prochaine frame à capturer

print(f"\n✅ Simulation complète ({duration:.2f}s simulées)")
print(f"🎞️ {len(frames)} frames enregistrées pour une vidéo de {real_duration:.2f}s à {FPS} fps")

# --- Sauvegarde ---
output_path = "replay.mp4"
iio.imwrite(output_path, frames, fps=FPS)
print(f"🎬 Vidéo sauvegardée → {output_path}")


import numpy as np
import cv2
import os  # <-- 1. Importer le module os

# Définir le nom du fichier
video_filename = "replay.mp4"

# Construire le chemin complet
script_dir = os.path.dirname(os.path.realpath(__file__))
video_path = os.path.join(script_dir, video_filename)

print(f"Tentative d'ouverture de : {video_path}")

# Vérifier si le fichier existe
if not os.path.exists(video_path):
    print(f"Erreur : Le fichier '{video_filename}' est introuvable.")
else:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur: OpenCV n'a pas pu ouvrir le fichier vidéo.")
    else:
        print("Vidéo ouverte. Lancement de la lecture...")
        
        # --- LA CORRECTION EST ICI ---

        # 1. Récupérer le vrai FPS de la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 2. Calculer le délai en millisecondes
        # S'il ne trouve pas le FPS (fps=0), on met 30 par défaut
        if fps > 0:
            delay_ms = int(1000 / fps)
        else:
            delay_ms = int(1000 / 30) # 30 FPS par défaut

        print(f"Vidéo à {fps:.2f} FPS, pause de {delay_ms} ms entre les images.")
        print("Appuyez sur 'q' pour quitter.")
        
        # --- FIN DE LA CORRECTION ---

        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                cv2.imshow("Lecture de 'replay.mp4'", frame)
                
                # 3. Utiliser le bon délai
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    break
            else:
                print("Fin de la vidéo.")
                break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()