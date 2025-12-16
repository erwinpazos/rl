import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
import xml.etree.ElementTree as ET

print("="*60)
print("MJX SIMULATION TEST - SIMPLIFIED MODEL")
print("="*60)
print("\nLoading simplified MuJoCo model (MJX compatible)...")

def extract_robot_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    components = {
        'compiler': None, 'option': None, 'default': None,
        'asset': None, 'robot_body': None, 'actuators': None, 'visual': None
    }
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
            for body in child:
                if body.get('name') == 'robot':
                    components['robot_body'] = body
        elif child.tag == 'actuator':
            components['actuators'] = child
        elif child.tag == 'visual':
            components['visual'] = child
    return components

def extract_corridor_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    components = {
        'compiler': None, 'option': None, 'default': None,
        'asset': None, 'corridor_geom': None, 'actuators': None
    }
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

def build_combined_model(robot_components, corridor_components, robot_height=0.45):
    root = ET.Element('mujoco')
    root.set('model', 'robot_with_corridor_mjx')
    
    if robot_components['compiler'] is not None:
        root.append(robot_components['compiler'])
    
    option = ET.Element('option')
    option.set('timestep', '0.01')
    option.set('gravity', '0 0 -9.81')
    option.set('solver', 'Newton')
    option.set('integrator', 'RK4')
    option.set('iterations', '50')
    root.append(option)
    
    size = ET.Element('size')
    size.set('njmax', '1000')
    size.set('nconmax', '500')
    root.append(size)
    
    if robot_components['default'] is not None:
        root.append(robot_components['default'])
    if robot_components['visual'] is not None:
        root.append(robot_components['visual'])
    
    asset = ET.Element('asset')
    added_material_names = set()
    
    if robot_components['asset'] is not None:
        for original_material in robot_components['asset']:
            material_name = original_material.get('name', '')
            if material_name not in added_material_names:
                asset.append(original_material)
                added_material_names.add(material_name)
    if corridor_components['asset'] is not None:
        for original_material in corridor_components['asset']:
            material_name = original_material.get('name', '')
            if material_name not in added_material_names:
                asset.append(original_material)
                added_material_names.add(material_name)
    
    root.append(asset)
    
    worldbody = ET.Element('worldbody')
    
    if robot_components['robot_body'] is not None:
        robot_z_position = robot_height - 0.1
        current_pos = robot_components['robot_body'].get('pos', '0 0 0.2')
        pos_parts = current_pos.split()
        if len(pos_parts) == 3:
            new_pos = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
            robot_components['robot_body'].set('pos', new_pos)
        worldbody.append(robot_components['robot_body'])
    
    if corridor_components['corridor_geom'] is not None:
        for geom in corridor_components['corridor_geom']:
            worldbody.append(geom)
    
    root.append(worldbody)
    
    if robot_components['actuators'] is not None:
        root.append(robot_components['actuators'])
    
    xml_string = ET.tostring(root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml_string)
    return model

# Load simplified model (MJX compatible)
print("Loading robot_simple_mjx.xml (capsule wheels, plane floor)...")
m = mujoco.MjModel.from_xml_path("robot_simple_mjx.xml")
print(f"✓ Model loaded: {m.ngeom} geometries, {m.nu} actuators")

print("Converting to MJX (GPU)...")
try:
    model_mjx = mjx.put_model(m)
    print("✓ Model converted to MJX successfully!")
except Exception as e:
    print(f"✗ MJX conversion failed: {e}")
    print("\nThis is expected - the corridor has complex geometries (boxes, cylinders)")
    print("that MJX doesn't support yet. MJX only supports:")
    print("  - Planes")
    print("  - Spheres")
    print("  - Capsules")
    print("\nYour corridor has:")
    print(f"  - {m.ngeom} geometries (mostly boxes)")
    print("  - Robot with cylinder wheels")
    print("\nConclusion: MJX cannot handle this environment.")
    exit(1)

print("Creating MJX data...")
data_mjx = mjx.make_data(model_mjx)

@jit
def step_mjx(model, data, ctrl):
    data = data.replace(ctrl=ctrl)
    data = mjx.step(model, data)
    return data

print("\nRunning MJX simulation (GPU)...")
print("Moving robot forward for 1000 steps...")

ctrl = jnp.array([5.0, 5.0, 5.0, 5.0], dtype=jnp.float32)

start_time = time.time()
for i in range(1000):
    data_mjx = step_mjx(model_mjx, data_mjx, ctrl)
    if i % 100 == 0:
        pos = np.array(data_mjx.qpos[:3])
        print(f"Step {i}: position = {pos}")

elapsed = time.time() - start_time
print(f"\n✓ Simulation complete!")
print(f"Time: {elapsed:.3f}s")
print(f"Steps per second: {1000/elapsed:.0f}")
print(f"Final position: {np.array(data_mjx.qpos[:3])}")
