# Control with arrow keys, spacebar to stop and 'c' to center steering

import mujoco
from mujoco import viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
from enum import Enum,unique,auto

# Global viewer reference for keyboard callback
viewer_instance = None
display_camera_info = False
quit_requested = False
debug_mode = False
tracking_mode_switch_required = False
tracking_mode = False
fixed_camera_mode_switch_required = False
fixed_camera_mode = False
show_contacts = False

# Global robot control variables
max_speed = 314  # max wheel speed in rad/s
steering_angle = 0.0  # in degrees, positive is left turn
velocity = 0.0       # velocity, positive is forward
delta_v = 0.05# 0.05  # change in velocity per command
delta_angle = 1  # change in steering angle per command

def v_to_delta_v(v):
    """Delta_v definition: the bigger the velocity magnitude, the bigger the delta_v step."""
    delta_v = 0.1 * abs(v) + 0.025
    return delta_v


@unique
class MotionState(Enum):
    NONE = auto()
    STOP = auto()
    ACCELERATE_STRAIGHT = auto()
    ACCELERATE_BACKWARD = auto()
    STEER_LEFT = auto()
    STEER_RIGHT = auto()
    STEER_CENTER = auto()

motion_state = MotionState.STOP
motion_request = MotionState.STOP

def clip_steering_angle(angle, max_angle=30.0):
    """Clip steering angle to within +/- max_angle degrees."""
    if angle > max_angle:
        return max_angle
    elif angle < -max_angle:
        return -max_angle
    return angle

def steer_angle_to_wheel_speeds_hackerman(steering_angle, speed=0.5, wheelbase_length=0.8, wheel_radius=0.15, track_width=0.6, max_steering_angle=30.0 # in degrees
                                          ):
    """
    Convert a steering angle (in degrees) to individual wheel speeds for a 4-wheel robot.
    Positive steering_angle means turning left.
    base_speed is the forward speed when going straight.
    Returns (speed_fl, speed_fr, speed_rl, speed_rr) in rad/s.
    """
    print(f"Calculating wheel speeds for steering angle: {steering_angle} degrees, base speed: {speed} m/s")

    # Simple differential steering model
    steering_angle = clip_steering_angle(steering_angle, max_steering_angle)
    
    if steering_angle == 0.0:
        return (speed/wheel_radius, speed/wheel_radius, speed/wheel_radius, speed/wheel_radius)
    
    # Calculate turning radius
    turn_radius = wheelbase_length / np.tan(np.radians(steering_angle))  # scale factor based on wheelbase length
    print(f"  Turn radius: {turn_radius:.2f} m")
    
    track_width_half = track_width / 2.0
    pcorr_turn_radius = turn_radius + track_width_half
    ncorr_turn_radius = turn_radius - track_width_half

    # Calculate linear wheel speeds based on turn radius in m/s
    speed_fl_m_s = speed * (ncorr_turn_radius) / turn_radius  # front-left
    speed_fr_m_s = speed * (pcorr_turn_radius) / turn_radius  # front-right
    speed_rl_m_s = speed * (ncorr_turn_radius) / turn_radius  # rear-left
    speed_rr_m_s = speed * (pcorr_turn_radius) / turn_radius  # rear-right

    print(f"  Wheel linear speeds (m/s): FL={speed_fl_m_s:.2f}, FR={speed_fr_m_s:.2f}, RL={speed_rl_m_s:.2f}, RR={speed_rr_m_s:.2f}")

    # Convert linear speeds (m/s) to angular speeds (rad/s)
    speed_fl = speed_fl_m_s / wheel_radius
    speed_fr = speed_fr_m_s / wheel_radius
    speed_rl = speed_rl_m_s / wheel_radius
    speed_rr = speed_rr_m_s / wheel_radius

    print(f"  Wheel angular speeds (rad/s): FL={speed_fl:.2f}, FR={speed_fr:.2f}, RL={speed_rl:.2f}, RR={speed_rr:.2f}")
    
    return (speed_fl, speed_fr, speed_rl, speed_rr)


def steer_angle_to_wheel_speeds_spin(steering_angle_deg, speed=0.5,
                                     wheelbase_length=0.8, wheel_radius=0.15, track_width=0.6, max_steering_angle=30.0 # in degrees
                                     ):
    """
    Spin-on-steer mapping:
    - steering_angle > 0 (left): right wheels +, left wheels -
    - steering_angle < 0 (right): right wheels -, left wheels +
    - steering_angle == 0: all wheels same sign as speed
    speed is treated as a *magnitude* for turning (m/s).
    Returns wheel angular speeds (rad/s): (fl, fr, rl, rr)
    """
    steering_angle_deg = clip_steering_angle(steering_angle_deg, max_steering_angle)

    # Straight: pure translation
    if abs(steering_angle_deg) < 1e-9:
        w = speed / wheel_radius
        return (w, w, w, w)

    # Turn intensity scales with steering angle (0..1)
    turn_gain = abs(steering_angle_deg) / max_steering_angle

    # Use magnitude of speed as "how hard to spin"
    turn_v = abs(speed) * turn_gain  # m/s equivalent tangential command

    # Left turn => right +, left -
    if steering_angle_deg > 0:
        v_left, v_right = -turn_v, +turn_v
    else:  # right turn
        v_left, v_right = +turn_v, -turn_v

    w_left  = v_left  / wheel_radius
    w_right = v_right / wheel_radius

    return (w_left, w_right, w_left, w_right)

def yaw_rate_to_wheel_speeds(yaw_rate_rps: float,
                             speed: float,
                             track_width: float = 0.4,
                             wheel_radius: float = 0.15,
                             max_wheel_rad_s: float | None = None):
    """
    Arcade-style control: command (v, yaw_rate) -> 4 wheel angular speeds (rad/s)
    For a 4-wheel skid-steer robot:
      - yaw_rate > 0 turns left
      - v > 0 goes forward

    Parameters
    ----------
    yaw_rate_rps : float
        Desired yaw rate in radians per second.
    speed : float
        Desired forward speed in meters per second.
    track_width : float
        Distance between left and right wheels (meters).
    wheel_radius : float
        Radius of the wheels (meters).
    max_wheel_rad_s : float | None
        Optional maximum wheel angular speed (rad/s) for saturation.

    Returns
    -------
    (w_fl, w_fr, w_rl, w_rr) in rad/s
        4-tuple of wheel angular speeds in radians per second: (front-left, front-right, rear-left, rear-right)
    """
    # Linear speed for each side (m/s)
    v_left  = speed - yaw_rate_rps * (track_width / 2.0)
    v_right = speed + yaw_rate_rps * (track_width / 2.0)

    # Convert to wheel angular speeds (rad/s)
    w_left  = v_left / wheel_radius
    w_right = v_right / wheel_radius

    # Optional saturation (keeps arcade feel without exploding commands)
    if max_wheel_rad_s is not None:
        w_left  = float(np.clip(w_left,  -max_wheel_rad_s, max_wheel_rad_s))
        w_right = float(np.clip(w_right, -max_wheel_rad_s, max_wheel_rad_s))

    # FL, FR, RL, RR
    return (w_left, w_right, w_left, w_right)

def steer_angle_to_wheel_speeds(steering_angle, speed=0.5, wheelbase_length=0.8, wheel_radius=0.15, track_width=1.8, max_steering_angle=30.0):
    """
    Wrapper function to choose steering model.
    Currently uses the Hackerman model.
    """
    # return steer_angle_to_wheel_speeds_hackerman(steering_angle=steering_angle, speed=speed, wheelbase_length=wheelbase_length, wheel_radius=wheel_radius, track_width=track_width, max_steering_angle=max_steering_angle)
    # return steer_angle_to_wheel_speeds_spin(steering_angle_deg=steering_angle, speed=speed, wheelbase_length=wheelbase_length, wheel_radius=wheel_radius, track_width=track_width, max_steering_angle=max_steering_angle)
    # Set yaw rate as steering angle
    max_yaw_rate = np.radians(45.0)  # rad/s at max steering angle
    steering_angle_rad = np.radians(steering_angle)
    return yaw_rate_to_wheel_speeds(yaw_rate_rps=steering_angle_rad,
                                    speed=speed,
                                    track_width=track_width,
                                    wheel_radius=wheel_radius,
                                    max_wheel_rad_s=None)


def key_callback(keycode):
    global viewer_instance, display_camera_info, quit_requested, motion_request, tracking_mode_switch_required, fixed_camera_mode_switch_required, debug_mode, show_contacts
    # print (f"Key pressed: {keycode}")
    handled = False
    if keycode == ord('v') or keycode == ord('V'):
        print("\nCamera info requested:")
        display_camera_info = True
        handled = True
    elif keycode == ord('t') or keycode == ord('T'):
        print("\nCamera tracks robot")
        tracking_mode_switch_required = True
        handled = True
    elif keycode == ord('f') or keycode == ord('F'):
        print("\nCamera fixed toogle between fixed cameras")
        fixed_camera_mode_switch_required = True
        handled = True
    elif keycode == ord('q') or keycode == ord('Q'):
        print("\nQ pressed - Quitting...")
        quit_requested = True
        handled = True
    elif keycode == 256:  # ESC key
        print("\nESC pressed - Quitting...")
        quit_requested = True
        handled = True
    elif keycode == 265:  # Up arrow
        print("\tAcc. straight requested")
        motion_request = MotionState.ACCELERATE_STRAIGHT
        handled = True
    elif keycode == 263:  # Left arrow
        print("\tTurn left requested")
        motion_request = MotionState.STEER_LEFT
        handled = True
    elif keycode == 264:  # Down arrow
        print("\tAcc. backward requested")
        motion_request = MotionState.ACCELERATE_BACKWARD
        handled = True
    elif keycode == 262:  # Right arrow
        print("\tTurn right requested")
        # For simplicity, we can treat right turn as left turn with inverted speeds
        motion_request = MotionState.STEER_RIGHT
        handled = True
    elif keycode == ord('c') or keycode == ord('C'):
        print("\nCenter steering requested")
        motion_request = MotionState.STEER_CENTER
        handled = True
    elif keycode == ord('p') or keycode == ord('P'):
        if debug_mode:
            print("\nDEBUG MODE disabled")
        else:
            print("\nDEBUG MODE requested")
        debug_mode = not debug_mode
        handled = True
    elif keycode == ord('i') or keycode == ord('I'):
        if show_contacts:
            print("\nContact display disabled")
        else:
            print("\nContact display enabled")
        show_contacts = not show_contacts
        handled = True
    elif keycode == ord(' '):
        print("\nStop requested")
        motion_request = MotionState.STOP
        handled = True
    return handled


def print_camera_info(v):
    cam = v.cam
    print("=== Camera Parameters ===")
    print(f"Type: {cam.type}")
    print(f"Lookat: [{cam.lookat[0]:.2f}, {cam.lookat[1]:.2f}, {cam.lookat[2]:.2f}]")
    print(f"Azimuth: {cam.azimuth:.1f}°")
    print(f"Elevation: {cam.elevation:.1f}°")
    print(f"Distance: {cam.distance:.2f}")
    print("========================")

def setCam(in_cam, out_cam):
    out_cam.type = in_cam.type
    out_cam.fixedcamid = in_cam.fixedcamid or -1
    out_cam.trackbodyid = in_cam.trackbodyid or -1
    if -1 == in_cam.trackbodyid:
        out_cam.lookat[:] = in_cam.lookat
    out_cam.distance = in_cam.distance
    out_cam.azimuth = in_cam.azimuth
    out_cam.elevation = in_cam.elevation

def set_camera_from_pos_xyaxes(cam, pos, xyaxes):
    """
    Convert MuJoCo XML camera format (pos + xyaxes) to mjvCamera.
    pos: [x, y, z] - absolute camera position
    xyaxes: [x_axis_0, x_axis_1, x_axis_2, y_axis_0, y_axis_1, y_axis_2]
    """
    pos = np.array(pos)
    x_axis = np.array(xyaxes[:3])
    y_axis = np.array(xyaxes[3:])
    
    # z_axis is perpendicular to both x and y
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # The camera looks along -z_axis direction
    # Find a reasonable lookat point (typically near the center of the scene)
    lookat = pos - 5 * z_axis  # 5 meters forward from camera
    
    # Calculate distance and angles
    distance = np.linalg.norm(lookat - pos)
    
    # Calculate azimuth and elevation from the view direction
    view_dir = lookat - pos
    azimuth = np.arctan2(view_dir[1], view_dir[0]) * 180 / np.pi
    elevation = np.arcsin(view_dir[2] / np.linalg.norm(view_dir)) * 180 / np.pi
    
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

def extract_materials_from_xml(xml_file_path):
    """
    Extract material definitions from existing XML file.
    This teaches students how to parse and reuse XML components.
    """
    print(f"Extracting materials from {xml_file_path}...")
    
    # Parse the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    materials = []
    
    # Find and extract material definitions
    for child in root:
        if child.tag == 'asset':
            for asset_child in child:
                if asset_child.tag == 'material':
                    materials.append(asset_child)
    
    return materials
    
def extract_robot_from_xml(xml_file_path):
    """
    Extract robot components from existing XML file.
    This teaches students how to parse and reuse XML components.
    """
    print(f"  • Extracting robot components from {xml_file_path}...")
    
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
            # Extract the robot body from worldbody
            for body in child:
                if body.get('name') == 'robot':
                    components['robot_body'] = body
        elif child.tag == 'actuator':
            components['actuators'] = child
    
    return components

def extract_worldbody_children_from_xml(xml_file_path):
    """
    Extract components in worldbody from existing XML file.
    This teaches students how to parse and reuse XML components.
    """
    print(f"  • Extracting worldbody children from {xml_file_path}...")
    
    # Parse the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    components = {
        'asset': None,
        'scene_bodies': [],
    }
    
    worldbody_children = []
    
    # Find and extract worldbody
    for child in root:
        if child.tag == 'worldbody':
            worldbody_children = []
            for body in child:
                worldbody_children.append(body)
        elif child.tag == 'asset':
            components['asset'] = child
    components['scene_bodies'] = worldbody_children
    
    return components

floor_configs = {
    "standard": {
        "material": "mat_floor_normal",
        "friction": "1.0 0.005 0.0001",
        "rgba": "0.8 0.9 0.8 1",
        "description": "Standard floor with normal friction"
    },
    "ice": {
        "material": "mat_floor_ice", 
        "friction": "0.1 0.001 0.0001",
        "rgba": "0.9 0.95 1.0 1",
        "description": "Icy floor with very low friction"
    },
    "sand": {
        "material": "mat_floor_sand",
        "friction": "2.0 0.1 0.01", 
        "rgba": "0.9 0.8 0.6 1",
        "description": "Sandy floor with high friction"
    }
}

def create_enhanced_materials():
    """
    Create enhanced materials with textures and visual appeal.
    Returns a list of material elements.
    """
    materials = []
    
    # Enhanced floor materials with textures and proper scaling
    floor_materials = [
        {
            'name': 'mat_floor_normal',
            'texture': 'tex_grid',
            'rgba': '0.7 0.8 1.0 0.8',  # Faint blue with transparency
            'shininess': '0.1',
            'specular': '0.3',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        },
        {
            'name': 'mat_floor_ice', 
            'texture': 'tex_grid',
            'rgba': '0.6 0.7 1.0 0.7',  # Slightly more blue for ice
            'shininess': '0.8',
            'specular': '0.9',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        },
        {
            'name': 'mat_floor_sand',
            'texture': 'tex_grid',
            'rgba': '0.9 0.8 0.6 0.8',  # Keep sand yellowish
            'shininess': '0.1',
            'specular': '0.1',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        }
    ]
    
    # Enhanced robot materials
    robot_materials = [
        {
            'name': 'mat_chassis_beige',
            'rgba': '0.96 0.87 0.70 1',  # Nice beige color
            'shininess': '0.3',
            'specular': '0.5'
        },
        {
            'name': 'mat_wheel_black',
            'texture': 'tex_wheel_radius',
            'rgba': '0.1 0.1 0.1 1',
            'shininess': '0.6',
            'specular': '0.3'
        }
    ]
    
    # Create material elements
    all_materials = floor_materials + robot_materials
    for mat_config in all_materials:
        material = ET.Element('material')
        material.set('name', mat_config['name'])
        material.set('rgba', mat_config['rgba'])
        if 'texture' in mat_config:
            material.set('texture', mat_config['texture'])
        if 'shininess' in mat_config:
            material.set('shininess', mat_config['shininess'])
        if 'specular' in mat_config:
            material.set('specular', mat_config['specular'])
        if 'texrepeat' in mat_config:
            material.set('texrepeat', mat_config['texrepeat'])
        materials.append(material)
    
    return materials

def create_textures():
    """
    Create texture elements for enhanced visuals.
    Returns a list of texture elements.
    """
    textures = []
    
    # Grid texture for floor - 2x2 meter cells
    grid_texture = ET.Element('texture')
    grid_texture.set('name', 'tex_grid')
    grid_texture.set('type', '2d')
    grid_texture.set('builtin', 'checker')
    grid_texture.set('rgb1', '0.7 0.8 1.0')    # Faint blue base
    grid_texture.set('rgb2', '0.75 0.85 1.0')  # Slightly lighter blue alternate
    grid_texture.set('width', '100')           # Texture resolution
    grid_texture.set('height', '100')          # Texture resolution
    grid_texture.set('mark', 'edge')           # White grid lines
    grid_texture.set('markrgb', '1 1 1')       # Pure white grid lines
    textures.append(grid_texture)
    
    # Wheel radius texture to show rotation
    wheel_texture = ET.Element('texture')
    wheel_texture.set('name', 'tex_wheel_radius')
    wheel_texture.set('type', '2d')
    wheel_texture.set('builtin', 'checker')
    wheel_texture.set('rgb1', '0.8 0.2 0.2')  # Red spoke
    wheel_texture.set('rgb2', '0.1 0.1 0.1')  # Black tire
    wheel_texture.set('width', '32')
    wheel_texture.set('height', '32')
    wheel_texture.set('mark', 'random')
    wheel_texture.set('markrgb', '0.8 0.8 0.2')  # Yellow marks
    textures.append(wheel_texture)
    
    return textures

def enhance_robot_visuals(robot_body):
    """
    Enhance robot body with better materials and visual features.
    Also adjusts wheel positions to be outside the robot body.
    Modifies the robot body XML element in place.
    """
    if robot_body is None:
        return
    
    print("Enhancing robot visuals...")
    
    # Find and update chassis material
    for geom in robot_body.iter('geom'):
        if geom.get('name') == 'chassis':
            geom.set('material', 'mat_chassis_beige')
            print("  Updated chassis to beige color")
    
    # Find and update wheel materials (positions now defined in XML)
    wheel_count = 0
    
    for body in robot_body.iter('body'):
        body_name = body.get('name', '')
        if body_name.startswith('wheel_'):
            # Update wheel material
            for geom in body.iter('geom'):
                if geom.get('name', '').startswith('geom_'):
                    geom.set('material', 'mat_wheel_black')
                    wheel_count += 1
            
            # Log the wheel position (already set in XML)
            current_pos = body.get('pos', '0 0 0')
            print(f"  {body_name} at position: {current_pos}")
    
    print(f"  Updated {wheel_count} wheels with textured black material")
    print("  Wheel positions defined in XML (no runtime modification)")

def build_combined_model(scene_components, robot_components, floor_type="standard", robot_xy = [0.0,0.0], robot_height=1.0):
    """
    Build a complete MuJoCo model by combining robot components with programmatic floor.
    Environment controls physics settings (gravity, timestep, etc).
    
    Args:
        robot_components: Extracted robot parts from XML
        floor_type: Type of floor to create ("standard", "ice", "sand")  
        robot_height: Starting height of robot above floor (meters)
    """
    print("Building combined model...")
    print(f"Robot starting height: {robot_height}m above floor")
    
    # Create root mujoco element
    root = ET.Element('mujoco')
    root.set('model', 'robot_with_corridor')
    
    # Add compiler settings from robot XML
    if robot_components['compiler'] is not None:
        root.append(robot_components['compiler'])
    
    # CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS (override robot's settings)
    option = ET.Element('option')
    option.set('timestep', '0.001') # Smaller timestep for stability with gravity and contacts
    option.set('gravity', '0 0 -9.81')  # Environment controls gravity!
    option.set('solver', 'Newton') # You can also try "PGS"
    option.set('iterations', '30') # Fewer iterations for performance, more iterations for accuracy and stability
    root.append(option)
    print("  Environment physics: gravity enabled, timestep=0.01s")
    
    # Add size settings
    size = ET.Element('size')
    size.set('njmax', '1000')
    size.set('nconmax', '500')
    root.append(size)
    
    if robot_components['default'] is not None:
        default = ET.Element('default')
        for child in robot_components['default']:
            if child.tag == 'geom':
                # remove default color in default geom to use enhanced materials
                if 'rgba' in child.attrib:
                    del child.attrib['rgba']
            default.append(child)
        root.append(default)
    
    # Create asset section with textures and enhanced materials
    asset = ET.Element('asset')
    
    # Add textures first
    textures = create_textures()
    for texture in textures:
        asset.append(texture)
    
    # Create worldbody with corridor and robot
    worldbody = ET.Element('worldbody')
    
    # Add enhanced materials (including floor and robot materials)
    enhanced_materials = create_enhanced_materials()
    asset.extend(enhanced_materials)
    enhanced_names = [mat.get('name', '') for mat in enhanced_materials]

    # Also keep any original materials from robot XML if they exist
    if robot_components['asset'] is not None:
        for original_material in robot_components['asset']:
            # Only add if it's not already in our enhanced materials
            material_name = original_material.get('name', '')
            if material_name not in enhanced_names:
                asset.append(original_material)
                enhanced_names.append(material_name)

    # Also keep any original materials from scene XML if they exist
    if scene_components['asset'] is not None:
        for original_material in scene_components['asset']:
            # Only add if it's not already in our enhanced materials
            material_name = original_material.get('name', '')
            if material_name in enhanced_names:
                print(f"  Skipping duplicate material in scene already added from robot or enhanced materials: {material_name}")
            else:
                asset.append(original_material)
                enhanced_names.append(material_name)

    root.append(asset)

    # Add worldbody components
    for component in scene_components['scene_bodies']:
        worldbody.append(component)
    
    # Add robot body with enhanced visuals and adjusted height
    if robot_components['robot_body'] is not None:
        # Enhance the robot's visual appearance
        enhance_robot_visuals(robot_components['robot_body'])
        
        # Adjust robot starting height (floor is at -0.1, so robot center should be at robot_height - 0.1)
        robot_z_position = robot_height - 0.1  # Floor offset
        new_pos = f"{robot_xy[0]} {robot_xy[1]} {robot_z_position}"
        robot_components['robot_body'].set('pos', new_pos)
        print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")
        
        worldbody.append(robot_components['robot_body'])
    
    root.append(worldbody)
    
    # Add actuators
    if robot_components['actuators'] is not None:
        root.append(robot_components['actuators'])
    
    # Convert to XML string
    xml_string = ET.tostring(root, encoding='unicode')

    # Store XML for debugging
    with open("combined_robot_corridor_model.xml", "w") as f:
        f.write(xml_string)
    print("  Combined model XML saved to 'combined_robot_corridor_model.xml'")
    
    # Create and return MuJoCo model
    return mujoco.MjModel.from_xml_string(xml_string)

# LESSON: Show students how to compose models from existing components
print("=== LESSON: add robot into corridor ===")
print("1. Loading robot components from XML file...")
print("2. Loading corridor scene programmatically...")
print("3. Combining components into new model...")
print("4. Enable piloting robot with keyboard inside corridor...")

def demonstrate_visual_options():
    """
    Show students the different visual options available.
    """
    print("\n\n")
    print("=== AVAILABLE VISUAL OPTIONS ===")
    print("Floor features:")
    print("  • Faint blue color with transparency")
    print("  • White grid lines with 2x2 meter cells")
    print("  • Professional appearance with realistic lighting")
    print()
    print("Floor types:")
    print("  'standard' - Faint blue grid with normal friction")
    print("  'ice'      - Slightly more blue with low friction (slippery)")
    print("  'sand'     - Yellow tinted grid with high friction")
    print()
    print("Robot features:")
    print("  • Beige chassis (warm, natural color)")
    print("  • Black wheels with rotation indicators")
    print("  • Wheels positioned OUTSIDE robot body for better visibility")
    print("  • 15cm wheel extension from body sides")
    print()
    print("To change floor type, modify the 'floor_type' variable above!")
    print("=" * 50)
    print()

# Choose floor type
floor_type = "standard"  # Try: "standard", "ice", "sand"

# Choose robot starting height (for gravity demonstration)
robot_init_h = 2.0  # Robot will start 2 meters above floor and fall down
robot_xy = [1.0, 0.0]  # Robot starting XY position

# Show visual options to students
demonstrate_visual_options()

print(f"=== PHYSICS DEMONSTRATION ===")
print(f"Robot will start {robot_init_h}m above the floor")
print("Watch it fall due to gravity - this proves environment physics work!")
print()

# Extract robot from existing XML file
print("1. Loading robot components from XML file programmatically...")
robot_components = extract_robot_from_xml("four_wheel_robot.xml")

# Extract scene from existing corridor XML file
print("2. Loading corridor scene programmatically...")
scene_components = extract_worldbody_children_from_xml("corridor_3x100.xml")

# Build combined model with enhanced visuals and physics
print("3. Combining components into new model...")
m_combined = build_combined_model(scene_components, robot_components , floor_type, robot_xy=robot_xy, robot_height=robot_init_h)
 
d = mujoco.MjData(m_combined)

# For convenience, use shorter variable name in the rest of the code
m = m_combined

print("\n")
print("=== VISUAL ENHANCEMENTS COMPLETE ===")
print(f"✅ Created model with {floor_type} floor")
print("✅ Floor: Faint blue with white 2x2m grid cells")
print("✅ Robot chassis: Beige color")
print("✅ Robot wheels: Black with rotation indicators")
print("✅ Wheels positioned outside robot body (defined in XML)")
print("✅ No XML strings embedded in Python code!")
print()

wheel_joints_names = ['hinge_fl', 'hinge_fr', 'hinge_rl', 'hinge_rr']
wheel_joints = {
    'hinge_fl' : {
        'speed' : 0.0,
        'ctrl' : 0.0
    }, 
    'hinge_fr' : {
        'speed' : 0.0,
        'ctrl' : 0.0
    },
    'hinge_rl' : {
        'speed' : 0.0,
        'ctrl' : 0.0
    },
    'hinge_rr' : {
        'speed' : 0.0,
        'ctrl' : 0.0
    }
}

def set_wheel_speeds(d, w_fl, w_fr, w_rl, w_rr):
    # actuator order matches XML order above
    # Clamp values to safe range to prevent instability
    global max_speed
    d.ctrl[:] = [
        np.clip(w_fl, -max_speed, max_speed),
        np.clip(w_fr, -max_speed, max_speed), 
        np.clip(w_rl, -max_speed, max_speed),
        np.clip(w_rr, -max_speed, max_speed)
    ]
    for i, joint_name in enumerate(wheel_joints_names):
        wheel_joints[joint_name]['ctrl'] = d.ctrl[i]
    

def check_wheel_spinning(d,m, step_count):
    # Add this in the simulation loop to monitor wheel speeds:
    new_wheel_speeds = {}
    speed_changed = False
    for i, joint_name in enumerate(wheel_joints_names):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            # qvel index for hinge joints starts after the free joint (6 DOFs)
            qvel_addr = m.jnt_dofadr[joint_id]
            new_wheel_speeds[joint_name] = {
                'speed': d.qvel[qvel_addr],
                'ctrl': d.ctrl[i]
            }
            if abs(new_wheel_speeds[joint_name]['speed'] - wheel_joints[joint_name]['speed']) > 0.01:
                speed_changed = True
    if speed_changed:
        ss = "["
        for joint_name in wheel_joints_names:
            ss += f" {joint_name}: speed={new_wheel_speeds[joint_name]['speed']:.3f} rad/s, ctrl={new_wheel_speeds[joint_name]['ctrl']:.3f} |"
            wheel_joints[joint_name]['speed'] = new_wheel_speeds[joint_name]['speed']
        ss = ss[:-1] + " ]"
        print(f"Wheel speeds updated at step {step_count}: {ss}")

known_contacts = []

def contacts_differ(c1, c2):
    if len(c1) != len(c2):
        return True
    for i in range(len(c1)):
        if c1[i]['geom1'] != c2[i]['geom1'] or c1[i]['geom2'] != c2[i]['geom2'] or abs(c1[i]['penetration_depth'] - c2[i]['penetration_depth']) > 0.001:
            return True
    return False

def check_contact_and_penetration(d,m, step_count):
    global known_contacts
    new_contacts = []
    # Check for contacts and penetration depth
    if d.ncon > 0:
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            penetration_depth = contact.dist  # Negative value indicates penetration
            contact_info = {'geom1': geom1_name, 'geom2': geom2_name, 'penetration_depth': penetration_depth}
            new_contacts.append(contact_info)
    if contacts_differ(known_contacts, new_contacts):
        known_contacts = new_contacts
        print(f"Change of contacts at step {step_count}:")
        for i, c in enumerate(known_contacts):
            print(f"  Contact {i}: {c['geom1']} <-> {c['geom2']}, penetration depth = {c['penetration_depth']:.4f} m\n")


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

# Get robot body ID
print("=== ROBOT BODY ID ===")
robot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot')
print(f"Robot body ID: {robot_body_id}")
print(f"Total bodies in model: {m.nbody}")
print("All body names:")
for i in range(m.nbody):
    body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  Body {i}: {body_name}")
print()

set_wheel_speeds(d, 0, 0, 0, 0)

# USE that to debug (slower visualization)
SLOW_MOTION = False

#define a tracking camera behind the robot
tracking_cam = mujoco.MjvCamera()
tracking_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
tracking_cam.fixedcamid = -1
tracking_cam.trackbodyid = robot_body_id
tracking_cam.azimuth = 0.0    # 0° = behind the robot, 180° = in front
tracking_cam.elevation = -30.0   # 30° = looking down from above (increase for higher view)
tracking_cam.distance = 5.0     # Distance from robot (increase to zoom out)

#define fixed camera parameters
corridor_entry_cam = mujoco.MjvCamera()
corridor_entry_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
corridor_entry_cam.fixedcamid = -1
set_camera_from_pos_xyaxes(corridor_entry_cam, [-6.323, -0.256, 4.726], [0.003, -1.000, 0.000, 0.368, 0.001, 0.930])

with viewer.launch_passive(m, d, key_callback=key_callback) as v:
    # Configure camera for a good view of the robot
    cam = v.cam
    setCam(corridor_entry_cam, cam)

    
    print(f"\n=== SIMULATION START ===")
    print(f"Floor type: {floor_type}")
    print("Robot visualization started!")
    print("You should see the robot moving inside the corridor.")
    print("Notice: The robot is loaded from XML, the corridor is loaded also from XML.")
    print("Press 'C' to display camera info, 'Q' or 'ESC' to quit.")
    print("Press 'Up arrow' to go straight, 'Left arrow' to turn left, 'Right arrow' to turn right, 'Down arrow' to go backward or 'SPACE' to stop.")
    print()
    
    start_time = time.time()
    step_count = 0
    #
    render_interval = 2  # Render every 2 simulation steps (slows down visualization)
    
    while v.is_running():
        mujoco.mj_step(m, d)
        if not SLOW_MOTION:
            v.sync()
        step_count += 1
        
        if SLOW_MOTION:
            # Only sync (render) every render_interval steps to slow down visualization
            if step_count % render_interval == 0:
                v.sync()
                time.sleep(0.01)  # Add 10ms delay between renders for smoother slow-motion
                
                # Debug: print robot position every 1000 steps
                if step_count % 1000 == 0:
                    robot_xpos = d.xpos[1]  # Body 1 is usually the robot (body 0 is world)
                    print(f"Step {step_count}: Robot pos = {robot_xpos}, Height = {robot_xpos[2]:.2f}m")
        
        # Use that for debug
        if debug_mode:
            check_wheel_spinning(d,m, step_count)
            if show_contacts:
                check_contact_and_penetration(d,m, step_count)
        
        # Check for instability and break if detected
        if np.any(np.isnan(d.qacc)) or np.any(np.isinf(d.qacc)):
            print(f"Simulation became unstable at step {step_count}")
            break

        # Handle keyboard requests
        if quit_requested:
            print("Quit requested via keyboard.")
            break
        else:
            if display_camera_info:
                print_camera_info(v)
                display_camera_info = False
            if tracking_mode_switch_required:
                tracking_mode = not tracking_mode
                tracking_mode_switch_required = False
            if tracking_mode:
                # Set tracking camera
                setCam(tracking_cam, v.cam)
            if fixed_camera_mode_switch_required:
                fixed_camera_mode = not fixed_camera_mode
                fixed_camera_mode_switch_required = False
            if fixed_camera_mode:
                # Reset to free camera
                setCam(corridor_entry_cam, v.cam)
                # fixed_camera_mode_required = False
            if motion_request != MotionState.NONE:
                match motion_request:
                    case MotionState.ACCELERATE_STRAIGHT:
                        # speed = max(0.0, speed)  # Ensure starting from non-negative speed
                        speed += delta_v
                        delta_v = v_to_delta_v(speed)
                        print(f"\tAccelerate straight: new speed = {speed:.2f} m/s")
                        wheel_speeds = steer_angle_to_wheel_speeds(steering_angle, speed)
                        if debug_mode:
                            print(f"\tWheel speeds (rad/s): FL={wheel_speeds[0]:.2f}, FR={wheel_speeds[1]:.2f}, RL={wheel_speeds[2]:.2f}, RR={wheel_speeds[3]:.2f}")
                        set_wheel_speeds(d, *wheel_speeds)
                    case MotionState.ACCELERATE_BACKWARD:
                        # speed = min(0.0, speed)  # Ensure starting from non-positive speed
                        speed -= delta_v
                        delta_v = v_to_delta_v(speed)
                        print(f"\tAccelerate backward: new speed = {speed:.2f} m/s")
                        wheel_speeds = steer_angle_to_wheel_speeds(steering_angle, speed)
                        if debug_mode:
                            print(f"\tWheel speeds (rad/s): FL={wheel_speeds[0]:.2f}, FR={wheel_speeds[1]:.2f}, RL={wheel_speeds[2]:.2f}, RR={wheel_speeds[3]:.2f}")    
                        set_wheel_speeds(d, *wheel_speeds)
                    case MotionState.STEER_LEFT:
                        # steering_angle = max(0.0, steering_angle)  # Ensure starting from non-negative angle
                        steering_angle = clip_steering_angle(steering_angle + delta_angle)
                        print(f"\tTurning left: new steering angle = {steering_angle:.2f} degrees")
                        wheel_speeds = steer_angle_to_wheel_speeds(steering_angle, speed)
                        if debug_mode:
                            print(f"\tWheel speeds (rad/s): FL={wheel_speeds[0]:.2f}, FR={wheel_speeds[1]:.2f}, RL={wheel_speeds[2]:.2f}, RR={wheel_speeds[3]:.2f}")
                        set_wheel_speeds(d, *wheel_speeds)
                    case MotionState.STEER_RIGHT:
                        # steering_angle = min(0.0, steering_angle)  # Ensure starting from non-positive angle
                        steering_angle = clip_steering_angle(steering_angle - delta_angle)
                        print(f"\tTurning right: new steering angle = {steering_angle:.2f} degrees")
                        wheel_speeds = steer_angle_to_wheel_speeds(steering_angle, speed)
                        if debug_mode:
                            print(f"\tWheel speeds (rad/s): FL={wheel_speeds[0]:.2f}, FR={wheel_speeds[1]:.2f}, RL={wheel_speeds[2]:.2f}, RR={wheel_speeds[3]:.2f}")
                        set_wheel_speeds(d, *wheel_speeds)
                    case MotionState.STEER_CENTER:
                        print("\tCentering steering: new steering angle = 0.00 degrees")
                        steering_angle = 0.0
                        wheel_speeds = steer_angle_to_wheel_speeds(steering_angle, speed)
                        if debug_mode:
                            print(f"\tWheel speeds (rad/s): FL={wheel_speeds[0]:.2f}, FR={wheel_speeds[1]:.2f}, RL={wheel_speeds[2]:.2f}, RR={wheel_speeds[3]:.2f}")
                        set_wheel_speeds(d, *wheel_speeds)
                    case MotionState.STOP:
                        print("\tStopping robot: setting speed to 0.0 m/s")
                        set_wheel_speeds(d, 0, 0, 0, 0)
                        speed = 0.0
                motion_request = MotionState.NONE
                motion_state = motion_request

    #end simulation loop

    if not v.is_running():
        exit()
    
print("Simulation completed successfully!")
