import mujoco
from mujoco import viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
import json

command_log = []
last_cmd = None  # pour ne logger que quand ça change
initial_state_saved = False

# Global viewer reference for keyboard callback
viewer_instance = None
display_camera_info = False


quit_requested = False

last_command_received = None
def set_wheel_speeds(d, w_fl, w_fr, w_rl, w_rr, key=None):
    max_speed = 100 
    d.ctrl[:] = 0
    d.ctrl[0] = np.clip(w_fl, -max_speed, max_speed)
    d.ctrl[1] = np.clip(w_fr, -max_speed, max_speed)
    d.ctrl[2] = np.clip(w_rl, -max_speed, max_speed)
    d.ctrl[3] = np.clip(w_rr, -max_speed, max_speed)



def save_initial_state_if_needed(d):
    global initial_state_saved, initial_state
    if not initial_state_saved:
        initial_state = {
            "qpos": d.qpos.copy().tolist(),
            "qvel": d.qvel.copy().tolist(),
            "act":  d.act.copy().tolist() if d.act.size else [],
        }
        initial_state_saved = True

def restore_initial_state(d, state):
    d.qpos[:] = np.array(state["qpos"])
    d.qvel[:] = np.array(state["qvel"])
    if state.get("act"):
        d.act[:] = np.array(state["act"])
    # Important pour que MuJoCo recalcule les dérivés cohérents
    mujoco.mj_forward(m, d)

speed_gear = 1

is_moving = 0
def custom_key_callback(keycode):
    global v, viewer_instance, display_camera_info, quit_requested, d, m, drone_mode, speed_gear
    global robot_body_id,is_moving,last_command_received
                
    backward_speed = 5
    linear_speed = 5
    turn_speed = 4
    print(keycode)
    # --- NOUVELLE LOGIQUE "STATELESS" ---
    # On n'utilise PLUS JAMAIS check_wheel_spinning() pour calculer une commande
    
    s = speed_gear * linear_speed  # Vitesse d'avance de base
    turn = turn_speed * 3          # Force de virage (3*8 = 24)

    if keycode == ord('c') or keycode == ord('C'):
        display_camera_info = True
    elif keycode == ord('q') or keycode == ord('Q'):
        quit_requested = True
    elif keycode == 256:  # ESC
        quit_requested = True
        
    # --- COMMANDES DE MOUVEMENT (STATELESS) ---
    
    elif keycode == 265:  # Flèche haut
        print("🚗 Roulage avant")
        w_fl = s
        w_fr = s
        w_rl = s
        w_rr = s
        is_moving = 1
    
    elif keycode == 264:  # Flèche bas
        print("🚗 Roulage arrière")
        w_fl = -backward_speed
        w_fr = -backward_speed
        w_rl = -backward_speed
        w_rr = -backward_speed
        is_moving = 1
    
    elif keycode == 263:  # Flèche gauche
        print("🚗 Virage à gauche (en avançant)")
        w_fl = s*is_moving - turn
        w_fr = s*is_moving + turn
        w_rl = s*is_moving - turn
        w_rr = s*is_moving + turn

    elif keycode == 262:  # Flèche droite
        print("🚗 Virage à droite (en avançant)")
        w_fl = s*is_moving + turn
        w_fr = s*is_moving - turn
        w_rl = s*is_moving + turn
        w_rr = s*is_moving - turn

    elif keycode == ord(' '): # Stop
        print("🚗 STOP")
        is_moving = 0
        set_wheel_speeds(d, 0, 0, 0, 0, keycode)

    elif keycode == 49: 
        speed_gear = 1
        print("Changement de vitesse : 1")
    elif keycode == 50: 
        speed_gear = 1.5
        print("Changement de vitesse : 2")
    elif keycode == 51: 
        speed_gear = 2
        print("Changement de vitesse : 3")
    elif keycode == 52: 
        speed_gear = 2.5
        print("Changement de vitesse : 3")
    elif keycode == 53: 
        speed_gear = 3
        print("Changement de vitesse : 5")
    # Stop
    elif keycode == ord(' '):
        set_wheel_speeds(d, 0, 0, 0, 0, keycode)
    if (keycode == 262 or keycode == 263 or keycode == 264 or keycode == 265):
        last_command_received = {
            "keycode": keycode,
            "wheels": [w_fl, w_fr, w_rl, w_rr]
        }



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


def build_combined_model(robot_components, corridor_components, robot_height=1.0):
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

    option.set('solver', 'Newton')    
    option.set('integrator', 'RK4')
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
    
    # Add textures first
    # textures = create_textures()
    # for texture in textures:
    #     asset.append(texture)

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
    
    # Create and return MuJoCo model
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# Extract robot from existing XML file
robot_components = extract_robot_from_xml("four_wheels_robot.xml")

corridor_components = extract_corridor_from_xml("corridor_3x100_no_full_obstacles.xml")

# Choose robot starting height
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
def check_wheel_spinning(d,m):
    # Add this in the simulation loop to monitor wheel speeds:
    wheel_joints = ['hinge_fl', 'hinge_fr', 'hinge_rl', 'hinge_rr']
    speed = []
    for i, joint_name in enumerate(wheel_joints):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            # qvel index for hinge joints starts after the free joint (6 DOFs)
            qvel_addr = m.jnt_dofadr[joint_id]
            print(f"{joint_name}: vel={d.qvel[qvel_addr]:.3f} rad/s, ctrl={d.ctrl[i]:.3f}")
            speed.append(d.qvel[qvel_addr])
    return speed

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

set_wheel_speeds(d, 0, 0, 0, 0)
robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')

def reset_camera_to_tracking(cam, body_id):
    """Réapplique les paramètres de suivi MuJoCo."""
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.azimuth = 0
    cam.elevation = -20
    cam.distance = 8

def start_camera(cam, body_id):
    """Réapplique les paramètres de suivi MuJoCo."""
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.azimuth = 180
    cam.elevation = -1
    cam.distance = 135
    for i in range(130):
        time.sleep(0.05)
        cam.distance -= 1
    print(cam.distance)
    steps = 90  # nombre total d'étapes de l’animation
    for i in range(steps):
        time.sleep(0.02)

        # azimuth : de 0 à -180° sur 90 étapes
        cam.azimuth -= 2  

        # elevation : sur 19/90 ≈ 0.21° par étape (donc sur 90 steps total → -19°)
        cam.elevation -= 19 / steps  

        # zoom (distance) : +0.3 total sur 90 steps → +1/300 par step
        cam.distance += (3 / 90)  # équivaut à ton 3*10 boucles à +1/10 total
    
    print(cam.azimuth)
    print(cam.elevation)
    print(cam.distance)
with viewer.launch_passive(m, d, key_callback=custom_key_callback) as v:
    # Configure camera for a good view of the robot
    cam = v.cam
    start_camera(cam, robot_body_id)
    reset_camera_to_tracking(cam, robot_body_id)
    start_time = time.time()
    step_count = 0

    target_timestep = m.opt.timestep
    # Le rendu peut tourner à 60Hz (pour être fluide)
    render_interval = 1.0 / 60.0
    # 2. Initialiser les horloges
    last_render_time = time.time()
    last_step_time = time.time()
    time_accumulator = 0.0
    while v.is_running():
        current_time = time.time()
        # 3. Calculer le temps réel écoulé et l'accumuler
        real_time_elapsed = current_time - last_step_time
        last_step_time = current_time
        time_accumulator += real_time_elapsed
        # Sauver l'état initial (une seule fois, juste après le premier forward)
        if not initial_state_saved:
            save_initial_state_if_needed(d)

        while time_accumulator >= target_timestep:
            mujoco.mj_step(m, d)
            step_count += 1
            if last_command_received is not None:
                cmd = last_command_received
                
                command_log.append({
                    "t": float(d.time),
                    "key": str(cmd["keycode"]),
                    "wheels": cmd["wheels"]
                })
                
                w = cmd["wheels"]
                set_wheel_speeds(d, w[0], w[1], w[2], w[3], key=None)
                
                last_command_received = None  # Consommée
            # --- Code qui doit tourner à 100Hz ---
            if not initial_state_saved:
                save_initial_state_if_needed(d)
            
            if np.any(np.isnan(d.qacc)) or np.any(np.isinf(d.qacc)):
                print(f"Simulation became unstable at step {step_count}")
                quit_requested = True # Forcer la sortie
                break
            # --- Fin code 100Hz ---

            # On "consomme" le temps qu'on vient de simuler
            time_accumulator -= target_timestep

        # Check for instability and break if detected
        if np.any(np.isnan(d.qacc)) or np.any(np.isinf(d.qacc)):
            print(f"Simulation became unstable at step {step_count}")
            break
        # Handle keyboard requests
        if quit_requested:
            print("Quit requested via keyboard.")
            break
        elif display_camera_info:
            print(f"Camera position: {cam.pos}, lookat: {cam.lookat}, distance: {cam.distance}, azimuth: {cam.azimuth}, elevation: {cam.elevation}")
            display_camera_info = False

        # 6. Boucle de RENDU (Indépendante)
        # On ne dessine que si 1/60s s'est écoulé
        if current_time - last_render_time >= render_interval:
            # --- Code qui peut tourner à 60Hz ---
            reset_camera_to_tracking(cam, robot_body_id)
            v.sync()
            last_render_time = current_time
            
            # Print robot Z position continuously
            robot_z = d.qpos[2]
            robot_x = d.qpos[0]
            print(f"\rRobot position: x={robot_x:.2f}m, z={robot_z:.3f}m", end="", flush=True)
            # --- Fin code 60Hz ---
    if not v.is_running():
        exit()
    
print("Simulation completed successfully!")
real_duration = time.time() - start_time
duration = command_log[-1]["t"]
print(f"Temps simulé : {duration:.2f}s")
print(f"Temps réel   : {real_duration:.2f}s")

with open("commands_log.json", "w") as f:
    json.dump({
        "initial_state": initial_state,
        "commands": command_log,
        "timestep": float(m.opt.timestep),
        "real_time": real_duration
    }, f, indent=2)

print(f"✅ {len(command_log)} commandes enregistrées dans commands_log.json")