

# ======================================================================
# NEW CELL
# ======================================================================import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET

# Define paths
corridor_path = Path("./corridor_3x100_no_full_obstacles.xml")
robot_path = Path("./four_wheels_robot.xml")

# Function to merge robot into corridor environment
def merge_robot_into_corridor(corridor_xml_path, robot_xml_path):
    """Merge the robot XML into the corridor environment."""
    
    # Parse corridor environment
    corridor_tree = ET.parse(corridor_xml_path)
    corridor_root = corridor_tree.getroot()
    
    # Parse robot definition
    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()
    
    # Extract robot components
    robot_worldbody = robot_root.find('worldbody')
    robot_actuators = robot_root.find('actuator')
    robot_assets = robot_root.find('asset')
    
    # Find corridor worldbody
    corridor_worldbody = corridor_root.find('worldbody')
    
    # Add robot body to corridor (position at start of corridor)
    if robot_worldbody is not None:
        for body in robot_worldbody:
            if body.get('name') == 'robot':
                # Clone and position robot at start
                robot_body = ET.fromstring(ET.tostring(body))
                robot_body.set('pos', '2.0 0 0.2')  # Start position in corridor
                corridor_worldbody.append(robot_body)
    
    # Add robot actuators
    corridor_actuators = corridor_root.find('actuator')
    if corridor_actuators is None:
        corridor_actuators = ET.SubElement(corridor_root, 'actuator')
    if robot_actuators is not None:
        for actuator in robot_actuators:
            corridor_actuators.append(ET.fromstring(ET.tostring(actuator)))
    
    # Add robot materials to assets
    corridor_assets = corridor_root.find('asset')
    if robot_assets is not None:
        for material in robot_assets.findall('material'):
            # Check if material already exists
            mat_name = material.get('name')
            existing = corridor_assets.find(f".//material[@name='{mat_name}']")
            if existing is None:
                corridor_assets.append(ET.fromstring(ET.tostring(material)))
    
    # Add a camera for nice overhead view
    # Check if worldbody has a camera, if not add one
    existing_camera = corridor_worldbody.find(".//camera[@name='tracking']")
    if existing_camera is None:
        camera = ET.Element('camera')
        camera.set('name', 'tracking')
        camera.set('mode', 'trackcom')
        camera.set('pos', '-5 0 5')
        camera.set('xyaxes', '0 -1 0 0.3 0 1')
        corridor_worldbody.append(camera)
    
    # Convert to string
    return ET.tostring(corridor_root, encoding='unicode')

# Merge the XMLs
merged_xml = merge_robot_into_corridor(corridor_path, robot_path)

# Load model from merged XML string
model = mujoco.MjModel.from_xml_string(merged_xml)
data = mujoco.MjData(model)

# Reset to initial state
mujoco.mj_resetData(model, data)

# Create renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Step simulation a few times to let physics settle
for _ in range(10):
    mujoco.mj_step(model, data)

# Update scene and render with tracking camera
renderer.update_scene(data, camera="tracking")
pixels = renderer.render()

# Display the image
plt.figure(figsize=(12, 8))
plt.imshow(pixels)
plt.axis('off')
plt.title('Four-Wheel Robot in Corridor Environment (Starting Position)')
plt.tight_layout()
plt.show()

print(f"Model loaded successfully!")
print(f"Number of actuators: {model.nu}")
print(f"Number of generalized coordinates (nq): {model.nq}")
print(f"Number of degrees of freedom (nv): {model.nv}")
print(f"Robot starting position: {data.qpos[:3]}")

# Question 1 - Explore actuators

print(f"Number of actuators (model.nu): {model.nu}")
print("\nActuator names:")
for i in range(model.nu):
    actuator_name = model.actuator(i).name
    print(f"  Actuator {i}: {actuator_name}")

print("\nThese actuators control the 4 wheels of the robot.")
print("Each actuator applies torque to rotate one wheel.")


# Question 2 - Check control ranges

print("Control ranges for each actuator:")
print(f"model.actuator_ctrlrange shape: {model.actuator_ctrlrange.shape}")
print(f"\nControl ranges (min, max) for each actuator:")
for i in range(model.nu):
    ctrl_min, ctrl_max = model.actuator_ctrlrange[i]
    actuator_name = model.actuator(i).name
    print(f"  {actuator_name}: [{ctrl_min:.2f}, {ctrl_max:.2f}]")

print("\nThese ranges tell us:")
print("- The robot can apply torque in both directions (forward/backward)")
print("- Each wheel can be controlled independently")
print("- The robot can move forward, backward, turn, and rotate in place")


# Question 3 - Send commands and visualize

# 1) print command shape
print(f"data.ctrl shape: {data.ctrl.shape}")
print(f"data.ctrl represents the control inputs for {model.nu} actuators")

# 2) print command initial values
print(f"\nInitial control values: {data.ctrl}")

# 3) Reset the simulation
mujoco.mj_resetData(model, data)
print("\nSimulation reset")

# 4) Set all wheels to spin forward at 1.0 rad/s
data.ctrl[:] = [1.0, 1.0, 1.0, 1.0]
print(f"Set all wheels to 1.0: {data.ctrl}")

# 5) Do 800 steps of the simulation and render the new state
for _ in range(800):
    mujoco.mj_step(model, data)

# Render the result
renderer.update_scene(data, camera="tracking")
pixels = renderer.render()

plt.figure(figsize=(12, 8))
plt.imshow(pixels)
plt.axis('off')
plt.title('Robot after 800 steps with all wheels at 1.0')
plt.tight_layout()
plt.show()

print(f"\nRobot position after 800 steps: {data.qpos[:3]}")
print("The robot moved forward along the corridor!")


# Question 4 - Explore robot state

# Positions (generalized coordinates) describes dimensions such as x, y, z, and orientations
print(f"data.qpos shape: {data.qpos.shape}")
print(f"data.qpos (positions): {data.qpos}")
print("\nqpos typically contains:")
print("  - Position: x, y, z (first 3 elements)")
print("  - Orientation: quaternion (next 4 elements for 3D rotation)")
print("  - Joint angles: wheel rotations (remaining elements)")

# Velocities (generalized velocities) describes dimensions such as linear and angular velocities
print(f"\ndata.qvel shape: {data.qvel.shape}")
print(f"data.qvel (velocities): {data.qvel}")
print("\nqvel typically contains:")
print("  - Linear velocity: vx, vy, vz (first 3 elements)")
print("  - Angular velocity: wx, wy, wz (next 3 elements)")
print("  - Joint velocities: wheel angular velocities (remaining elements)")

print(f"\nTotal state variables: {model.nq} positions + {model.nv} velocities = {model.nq + model.nv}")


# TODO: Question 5 - Explore robot position in corridor

# Extract robot position
robot_x, robot_y, robot_z = 0.0, 0.0, 0.0
print(f"Robot current x-position: {robot_x:.2f}m")

# Corridor parameters
corridor_start = 0.0
corridor_length = 100.0
corridor_end = corridor_start + corridor_length

print(f"Corridor: start at x={corridor_start:.1f}m, end at x={corridor_end:.1f}m")

# TODO: compute distance traveled and remaining distance
travel_distance = 0.0
print(f"Distance traveled from start: {travel_distance:.2f}m")
remaining_distance = 0.0
print(f"Remaining distance to goal: {remaining_distance:.2f}m")
progress_percent = 0.0
print(f"Progress: {progress_percent:.2f}%")

# TODO: Question 6 - Explore sensory information about floor cells

# Calculate current row from qpos

# Calculate current column from qpos

# Corridor structure
corridor_length = 100.0  # meters
corridor_width = 3.0  # meters
cell_width = 0.5  # meters (0.5m × 0.5m cells)
n_cells_per_row = 6  # 3m width / 0.5m = 6 cells per row

print(f"Corridor structure:")
print(f"  - Length: {corridor_length}m")
print(f"  - Width: {corridor_width}m")
print(f"  - Cell size: {cell_width}m × {cell_width}m")
print(f"  - Cells per row: {n_cells_per_row}")
print(f"  - Row length: {cell_width}m")
print()

# Define observation window: 3 rows (behind, under, ahead)
rows_to_observe = [-1, 0, 1]  # relative to current row

# Example: create a mock observation (in real implementation, this would come from the environment)
# Cell types: 0=flat, 1=bump, 2=hole
# What is its shape?

# Question 7 - Extract cell states from MuJoCo


print("=== Extracting Floor Cell Information from MuJoCo ===\n")

# 1) Explore the model's geometries
print(f"Total number of geometries: {model.ngeom}")
print("\nFirst 20 geometry names:")
for i in range(min(20, model.ngeom)):
    geom_name = model.geom(i).name
    print(f"  {i}: {geom_name if geom_name else '(unnamed)'}")

# 2) Find floor-related geometries and understand naming convention
print("APPROACH 1: SEMANTIC METHOD (using geometry names)\n")
print("Floor geometries (filtering for 'cell' in name):")
floor_geoms = []
for i in range(model.ngeom):
    geom_name = model.geom(i).name
    if geom_name and 'cell' in geom_name:
        floor_geoms.append(geom_name)
print(f"Found {len(floor_geoms)} floor cell geometries")
if len(floor_geoms) > 0:
    print(f"Examples: {floor_geoms[:5]}")

# 3a) Semantic approach: Design a way to map geometries to the cell grid structure using names
print("Building cell type lookup table from names...")
print("Assuming naming convention: names contain 'bump', 'hole', or default to 'flat'\n")

def get_cell_type_from_name(name):
    """Determine cell type from geometry name."""
    name_lower = name.lower()
    if 'bump' in name_lower or 'ramp' in name_lower:
        return 1  # bump
    elif 'hole' in name_lower:
        return 2  # hole
    else:
        return 0  # flat (default)

# Build a spatial index: map (row, col) -> cell_type
cell_map_semantic = {}
corridor_width = 3.0
cell_width = 0.5  # Updated: 0.5m × 0.5m cells
corridor_length = 100
n_x = 200
n_y = 6
half_width = corridor_width/2.0

def p_to_idx(x, y):
    # snap (x,y) to nearest cell indices
    ix = max(0, min(n_x-1, int(x / cell_width)))
    iy = max(0, min(n_y-1, int((y + half_width) / cell_width)))
    return ix, iy

def idx_to_p(ix, iy):
    # snap tile indices (ix,iy) to position (x,y), center of cell
    x = (ix+0.5) * cell_width
    y = (iy+0.5) * cell_width - half_width
    return x, y

def contains_2D(p,center,sizes):
    x,y=p
    cx,cy,_ = center
    half_size_x,half_size_y,_ = sizes
    min_x = cx-half_size_x
    max_x = cx+half_size_x
    min_y = cy-half_size_y
    max_y = cy+half_size_y
    if min_x <= x and x<= max_x and min_y <= y and y <= max_y:
        return True
    return False

# Initialize all cells as flat (0)
for r in range(0, n_x):
    for c in range(0, n_y):
        cell_map_semantic[(r, c)] = 0  # default: flat

# Now iterate through all geometries and update the map
for geom_id in range(model.ngeom):
    geom_name = model.geom(geom_id).name
    if geom_name and ('cell' in geom_name or 'ramp' in geom_name.lower()):
        # Get geometry position
        geom_pos = model.geom_pos[geom_id]
        geom_size = model.geom_size[geom_id]
        
        # Get cell type from name
        cell_type = get_cell_type_from_name(geom_name)
        
        # Find which cells this geometry covers
        for r in range(0, n_x):
            for c in range(0, n_y):
                cell_center_x, cell_center_y = idx_to_p(r, c)
                if contains_2D((cell_center_x, cell_center_y), geom_pos, geom_size):
                    cell_map_semantic[(r, c)] = cell_type

print(f"\nBuilt semantic lookup table with {len(cell_map_semantic)} cells")

test_positions = [
    (2., 0.0),   # Robot's starting position (flat below)
    (6.25, -0.75),  # Should be a hole
    (2.25, -1.25),   # Should be a bump
]

for test in test_positions:
    p = p_to_idx(test[0], test[1])
    cell_type = cell_map_semantic.get(p, None)
    type_name = ['flat', 'bump', 'hole'][cell_type] if cell_type is not None else "NOT FOUND"
    print(f"Testing semantic detection at position ({test[0]:.1f}, {test[1]:.1f}): {type_name}")

print("\n" + "="*70 + "\n")



# 3b) Geometric approach: Use collision detection to probe each cell
#     - Ray casting or collision queries at different heights
#     - Detect bumps (collision slightly above ground)
#     - Detect holes (no collision at ground level)


print("APPROACH 2: GEOMETRIC METHOD (using collision detection)\n")
print("Using MuJoCo ray casting to detect floor type at specific positions...")

def detect_cell_type_geometric(x, y, z_ground=0.05, z_bump_test=0.1):
    """
    Detect cell type using geometric probing with ray casting.
    
    Args:
        x, y: World coordinates of the cell center
        z_ground: Ground level height
        z_bump_test: Height to test for bumps
    
    Returns:
        0=flat, 1=bump, 2=hole
    """
    # Cast ray downward from above to detect ground
    ray_start = np.array([x, y, 1.0], dtype=np.float64)  # Start from above
    ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # Point downward
    
    # Prepare output array for geom_id (must be writable, shape [1,1], dtype int32)
    geom_id_array = np.array([[-1]], dtype=np.int32)
    
    # Use mj_ray to find first collision
    # Returns distance to collision (or -1 if no collision)

    geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
    geomgroup[5] = 0  # ignore group 5 (ghosts)
    
    distance = mujoco.mj_ray(model, data, ray_start, ray_direction, 
                              geomgroup, 1, -1, geom_id_array)
    print(f"Ray cast at ({x:.2f}, {y:.2f}): distance={distance:.3f}")
    
    geom_id = geom_id_array[0, 0]
    
    
    if geom_id == -1 or distance < 0:
        # No collision detected = hole (no floor at this position)
        return 2  # hole
    
    ground_distance = 1.0 - z_ground
    epsilon = 1.e-3
    if abs(distance-ground_distance) < epsilon:
        return 0  # flat
    elif distance < ground_distance:
        return 1  # bump
    elif distance > ground_distance:
        return 2  # hole
    else:
        raise ValueError("Unexpected ray distance value")


# Test geometric detection at a few positions
print("Testing geometric detection at sample positions:")

cell_map_geometric = {}
for test_x, test_y in test_positions:
    cell_type = detect_cell_type_geometric(test_x, test_y)
    type_name = ['flat', 'bump', 'hole'][cell_type]
    print(f"  Position ({test_x:.1f}, {test_y:.1f}): {type_name}")
    
    row_x = int(round(test_x / cell_width))
    col_y = int(round((test_y + corridor_width/2) / cell_width))
    cell_map_geometric[(row_x, col_y)] = cell_type

print(f"\nBuilt geometric lookup with {len(cell_map_geometric)} test cells")
print("\n" + "="*70 + "\n")

# 4) Implement a function to get observation around the robot
print("Implementing observation function (using semantic method)...\n")

def get_floor_observation(robot_x, robot_y, n_rows_ahead=1, n_rows_behind=1, n_cols=6):
    """
    Get floor cell observations around the robot.
    
    Args:
        robot_x: Robot's x position
        robot_y: Robot's y position
        n_rows_ahead: Number of rows to observe ahead
        n_rows_behind: Number of rows to observe behind
        n_cols: Number of columns (cells per row) - updated to 6
    
    Returns:
        2D numpy array of shape (n_rows_behind + 1 + n_rows_ahead, n_cols)
        with cell types: 0=flat, 1=bump, 2=hole
    """
    # Calculate robot's cell position
    robot_row = int(robot_x / cell_width)
    robot_col = int((robot_y + corridor_width/2) / cell_width)
    
    # Total number of rows in observation
    total_rows = n_rows_behind + 1 + n_rows_ahead
    
    # Create observation array
    observation = np.zeros((total_rows, n_cols), dtype=np.int32)
    
    # Fill observation array
    for i in range(total_rows):
        # Calculate actual row index (behind -> under -> ahead)
        row_offset = i - n_rows_behind
        actual_row = robot_row + row_offset
        
        for j in range(n_cols):
            # Get cell type from semantic map
            cell_type = cell_map_semantic.get((actual_row, j), 0)  # default to flat
            observation[i, j] = cell_type if cell_type is not None else 0
    
    return observation

# Test the observation function
robot_x = data.qpos[0]
robot_y = data.qpos[1]

observation = get_floor_observation(robot_x, robot_y, n_rows_ahead=1, n_rows_behind=1)

def mirror_row(row):
    """
    As the y=-1.5 is on the right side, we mirror the row for easier interpretation.
    0=leftmost cell, 5=rightmost cell
    0 1 2 3 4 5  -->  5 4 3 2 1 0
    """
    return row[::-1]

print("Floor observation around robot:")
print(f"Robot at x={robot_x:.2f}, y={robot_y:.2f}")
print(f"Observation shape: {observation.shape} (rows × cols)")
print(f"\nObservation array:")
print(observation)
print("\nRow interpretation:")
print(f"  Row 2 (ahead,  x≈{int(robot_x / cell_width) + 1} × 0.5m = {(int(robot_x / cell_width) + 1) * 0.5:.1f}m):\t{mirror_row(observation[2])}")
print(f"  Row 1 (under,  x≈{int(robot_x / cell_width)} × 0.5m = {int(robot_x / cell_width) * 0.5:.1f}m):\t{mirror_row(observation[1])}")
print(f"  Row 0 (behind, x≈{int(robot_x / cell_width) - 1} × 0.5m = {(int(robot_x / cell_width) - 1) * 0.5:.1f}m):\t{mirror_row(observation[0])}")
print("\n(0=flat, 1=bump, 2=hole)")


print("Now show observations for test positions:")
for test in test_positions:
    robot_x, robot_y = test
    observation = get_floor_observation(robot_x, robot_y, n_rows_ahead=2, n_rows_behind=2)
    print(f"\nRobot at x={robot_x:.2f}, y={robot_y:.2f}")
    print(f"Observation shape: {observation.shape} (rows × cols)")
    print(f"\nObservation array:")
    print(observation)
    print("\nRow interpretation:")
    print(f"  Row 4 (ahead,  x≈{int(robot_x / cell_width) + 2} × 0.5m = {(int(robot_x / cell_width) + 2) * 0.5:.1f}m):\t{mirror_row(observation[4])}")
    print(f"  Row 3 (ahead,  x≈{int(robot_x / cell_width) + 1} × 0.5m = {(int(robot_x / cell_width) + 1) * 0.5:.1f}m):\t{mirror_row(observation[3])}")
    print(f"  Row 2 (under,  x≈{int(robot_x / cell_width)} × 0.5m = {int(robot_x / cell_width) * 0.5:.1f}m):\t{mirror_row(observation[2])}")
    print(f"  Row 1 (behind, x≈{int(robot_x / cell_width) - 1} × 0.5m = {(int(robot_x / cell_width) - 1) * 0.5:.1f}m):\t{mirror_row(observation[1])}")
    print(f"  Row 0 (behind, x≈{int(robot_x / cell_width) - 2} × 0.5m = {(int(robot_x / cell_width) - 2) * 0.5:.1f}m):\t{mirror_row(observation[0])}")
    print("\n(0=flat, 1=bump, 2=hole)")
    print("" + "="*50)

# Question 8 - Calculate state space complexity

print("=== State Space Complexity ===\n")

# Calculate combinatorial explosion
# Corridor: 100m long × 3m wide, cells are 0.5m × 0.5m
# Each row is 0.5m long and contains 6 cells across the width
corridor_length = 100.0  # meters
corridor_width = 3.0  # meters
cell_size = 0.5  # meters (0.5m × 0.5m)
cells_per_row = 6  # 3m / 0.5m = 6 cells
total_rows = int(corridor_length / cell_size)  # 100m / 0.5m = 200 rows

observation_configs = [
    ("Minimal (1 row)", 1, cells_per_row * 1),
    ("Local (3 rows)", 3, cells_per_row * 3),
    ("Extended (7 rows)", 7, cells_per_row * 7),
    ("Full corridor (200 rows)", total_rows, cells_per_row * total_rows)
]

print(f"Corridor: {corridor_length}m long × {corridor_width}m wide")
print(f"Cell size: {cell_size}m × {cell_size}m")
print(f"Each row is {cell_size}m long and has {cells_per_row} cells across the width.")
print(f"Total rows in corridor: {total_rows} rows")
print("\nIf each cell has 3 possible states (flat/bump/hole):\n")

for name, n_rows, n_cells in observation_configs:
    #TODO compute n_combinations
    n_combinations = 0
    print(f"{name:35s}: {n_rows} rows × {cells_per_row} cells = {n_cells} cells")
    print(f"{'':35s}   3^{n_cells} = {n_combinations:,.0f} combinations")
    
    if n_cells <= 42:
        print(f"{'':35s}   ({n_combinations:,.0f} discrete states)")
    else:
        print(f"{'':35s}   ({n_combinations:.2e} - astronomical!)")
    print()

print("\n" + "="*70 + "\n")
print("State space characteristics:")
print()
print("FLOOR OBSERVATION ONLY (discrete):")
print(f"  - For local window (18 cells): {3**18:,} possible floor configurations")
print(f"  - This is manageable for lookup tables")
print()
print("ROBOT STATE (continuous):")
print(f"  - Position: (x, y) - 2 continuous dimensions")
print(f"  - Orientation: (θ) - 1 continuous dimension")
print(f"  - Velocities: (vx, vy, ω) - 3 continuous dimensions")
print(f"  - Total robot state: ~6+ continuous dimensions")
print()
print("COMBINED STATE SPACE (hybrid):")
print(f"  - Discrete floor observation × Continuous robot state")
print(f"  - This is a HYBRID state space (discrete + continuous)")
print(f"  - Total: INFINITE (due to continuous components)")
print()
print("💡 KEY INSIGHT: Trade-offs")
print()
print("LARGER OBSERVATION WINDOW:")
print("  ✅ More information → better planning")
print("  ✅ Can see obstacles further ahead")
print("  ❌ Exponentially larger state space")
print("  ❌ Harder to learn (more data needed)")
print()
print("SMALLER OBSERVATION WINDOW:")
print("  ✅ Smaller state space → faster learning")
print("  ✅ Less memory/computation required")
print("  ❌ Limited \"vision\" → reactive only")
print("  ❌ Can't plan ahead effectively")
print()
print("🎯 PRACTICAL APPROACH:")
print("  - Use function approximation (neural networks) instead of lookup tables")
print("  - This handles continuous states AND large discrete spaces")
print("  - Local observation window (3-7 rows) is usually sufficient")

# Question 9 - Detect terminal conditions

print("=== Terminal Condition Detection ===\n")

# Define corridor parameters
CORRIDOR_LENGTH = 100.0
CORRIDOR_WIDTH = 3.0  # 3 meters wide

# Implement a check function
def check_terminal_conditions(robot_state):
    """
    Check if robot has reached a terminal state.
    
    Args:
        robot_state: Dictionary or tuple with robot position information
        
    Returns:
        terminated: Boolean - True if episode ended naturally (success or failure)
        truncated: Boolean - True if episode was cut short (not used here, for completeness)
        info: Dictionary with diagnostic information
    """
    # Extract positions
    if isinstance(robot_state, dict):
        robot_x = robot_state['x']
        robot_y = robot_state['y']
        robot_z = robot_state['z']
    else:
        robot_x = robot_state[0]
        robot_y = robot_state[1]
        robot_z = robot_state[2]
    
    # Initialize
    terminated = False
    truncated = False  # Not used for natural termination
    info = {
        'position': (robot_x, robot_y, robot_z),
        'distance_to_goal': max(0, corridor_length - robot_x),
    }
    
    # Check terminal conditions
    if robot_x >= CORRIDOR_LENGTH:
        # Success: reached goal (must be beyond the end of the corridor)
        terminated = True
        info['success'] = True
        info['termination_reason'] = 'reached_goal'
    elif robot_z < 0.1 and robot_x > 0:
        # Failure: fell in hole (z position too low but still within corridor)
        terminated = True
        info['success'] = False
        info['termination_reason'] = 'fell_in_hole'
    elif robot_x < -1.0:
        # Failure: went too far backward
        terminated = True
        info['success'] = False
        info['termination_reason'] = 'went_backward'
    
    return terminated, truncated, info

# TODO Implement a reward function
def compute_reward(robot_state):
    """
    Compute reward based on robot state and terminal conditions.
    
    Args:
        robot_state: Dictionary or tuple with keys/indices for x, y, z positions
        
    Returns:
        reward: Float reward value
    """
    # Extract positions (handle both dict and array-like inputs)
    if isinstance(robot_state, dict):
        robot_x = robot_state['x']
        robot_y = robot_state['y']
        robot_z = robot_state['z']
    else:
        robot_x = robot_state[0]
        robot_y = robot_state[1]
        robot_z = robot_state[2]
    
    # Check terminal conditions and assign rewards
    # Use check_terminal_conditions function
    terminated, _, _ = check_terminal_conditions(robot_state)
    
    # Success: reached goal (must be beyond the end of the corridor)
    if robot_x >= CORRIDOR_LENGTH:
        return +100.0, True, "SUCCESS: Reached goal!"
    
    # Failure: fell in hole
    elif robot_z < 0.1 and robot_x > 0:
        return -100.0, True, "FAILURE: Fell in hole!"
    
    # Failure: went too far backward
    elif robot_x < -1.0:
        return -50.0, True, "FAILURE: Went backward!"
    
    # Episode continues - small reward for moving forward
    else:
        # Reward proportional to progress (encourage moving forward)
        r = 0.01 * robot_x  # Small positive reward for being further along
        return r, False, None





# Test the functions with current robot state
print("Testing reward and check functions:\n")

# Get current robot state
robot_state = {
    'x': data.qpos[0],
    'y': data.qpos[1],
    'z': data.qpos[2]
}

print(f"Current robot position: x={robot_state['x']:.2f}m, y={robot_state['y']:.2f}m, z={robot_state['z']:.2f}m\n")

# Test reward function
reward, terminated, reason = compute_reward(robot_state)
print("Reward Function Output:")
print(f"  Reward: {reward:+.2f}")
print(f"  Terminated: {terminated}")
if reason:
    print(f"  Reason: {reason}")
print()

# Test check function
terminated, truncated, info = check_terminal_conditions(robot_state)
print("Check Function Output:")
print(f"  Terminated: {terminated}")
print(f"  Truncated: {truncated}")
print(f"  Info: {info}")
print()

# Test with various scenarios
print("=" * 70)
print("Testing various scenarios:\n")

test_scenarios = [
    ({'x': 100.0, 'y': 0.0, 'z': 0.15}, "At goal"),
    ({'x': 50.0, 'y': 0.0, 'z': -1.0}, "Fell in hole"),
    ({'x': 30.0, 'y': 0.0, 'z': 0.15}, "Normal state (continuing)"),
    ({'x': -2.0, 'y': 0.0, 'z': 0.15}, "Went too far backward"),
]

for state, description in test_scenarios:
    reward, term, reason = compute_reward(state)
    print(f"{description:30s} → reward={reward:+7.2f}, terminated={term}, reason: {reason}")

# Question 10 - Implementing a step function

def step(action, n_steps=10):
    """
    Execute one step in the environment.
    
    Args:
        action: Control inputs for the 4 wheels (array of 4 values)
        n_steps: Number of MuJoCo simulation steps per environment step
    
    Returns:
        next_state: Robot state after action (x, y, z positions)
        reward: Reward obtained
        terminated: Whether episode ended
        truncated: Whether episode was truncated (not used here)
        info: Additional information
    """
    # Apply action to controls
    data.ctrl[:] = action
    
    # Step simulation multiple times for smoother physics
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    
    # Get next state
    next_state = {
        'x': data.qpos[0],
        'y': data.qpos[1],
        'z': data.qpos[2]
    }
    
    # Compute reward and check termination
    reward, terminated, reason = compute_reward(next_state)
    
    # Check terminal conditions for info
    _, truncated, info = check_terminal_conditions(next_state)
    info['termination_reason'] = reason
    
    return next_state, reward, terminated, truncated, info

# Test the step function
print("Testing step function:\n")
mujoco.mj_resetData(model, data)

# Try moving forward
action = np.array([1.0, 1.0, 1.0, 1.0])  # All wheels forward
next_state, reward, terminated, truncated, info = step(action)

print(f"Action: {action}")
print(f"Next state: x={next_state['x']:.2f}, y={next_state['y']:.2f}, z={next_state['z']:.2f}")
print(f"Reward: {reward:.2f}")
print(f"Terminated: {terminated}")
print(f"Info: {info}")


# Question 11 - Progress-based rewards and value functions

# Part A: Write progress-based reward functions

# Corridor parameters (needed for reward functions)
CORRIDOR_LENGTH = 100.0

# Global variables to track previous position for delta calculation (separate for each function)
previous_x_p1 = 0.0
previous_x_p2 = 0.0

def reward_p1(robot_state):
    """
    Progress-based reward that does NOT penalize backward movement.
    Rewards based on change in position (delta).
    """
    global previous_x_p1
    
    if isinstance(robot_state, dict):
        robot_x = robot_state['x']
        robot_z = robot_state['z']
    else:
        robot_x = robot_state[0]
        robot_z = robot_state[2]
    
    # Terminal rewards
    if robot_x >= CORRIDOR_LENGTH:
        return +100.0, True, "SUCCESS: Reached goal!"
    elif robot_z < 0.1 and robot_x > 0:
        return -100.0, True, "FAILURE: Fell in hole!"
    
    # Progress reward: reward forward movement
    delta_x = robot_x - previous_x_p1
    previous_x_p1 = robot_x
    
    # Only reward forward movement (ignore backward)
    reward = max(0, delta_x) * 10.0  # Scale up the reward
    
    return reward, False, None

def reward_p2(robot_state):
    """
    Progress-based reward that DOES penalize backward movement.
    """
    global previous_x_p2
    
    if isinstance(robot_state, dict):
        robot_x = robot_state['x']
        robot_z = robot_state['z']
    else:
        robot_x = robot_state[0]
        robot_z = robot_state[2]
    
    # Terminal rewards
    if robot_x >= CORRIDOR_LENGTH:
        return +100.0, True, "SUCCESS: Reached goal!"
    elif robot_z < 0.1 and robot_x > 0:
        return -100.0, True, "FAILURE: Fell in hole!"
    elif robot_x < -1.0:
        return -50.0, True, "FAILURE: Went backward!"
    
    # Progress reward: reward forward, penalize backward
    delta_x = robot_x - previous_x_p2
    previous_x_p2 = robot_x
    
    # Reward forward movement, penalize backward
    reward = delta_x * 10.0  # Can be positive or negative
    
    return reward, False, None

# Part B: Write a value function that accumulates rewards

def compute_value(state, action, gamma=0.95, reward_fn=reward_p2):
    """
    Compute value function: V(s) = r(s,a) + gamma * V(s')
    
    Args:
        state: Current state
        action: Action to take
        gamma: Discount factor (0.9-0.99)
        reward_fn: Reward function to use
    
    Returns:
        value: Estimated value of the state
    """
    # Get immediate reward
    reward, terminated, _ = reward_fn(state)
    
    if terminated:
        # Terminal state: value is just the reward
        return reward
    
    # For simplicity, estimate future value as discounted current reward
    # In real RL, this would use a learned value function
    estimated_future_value = reward * gamma
    
    value = reward + gamma * estimated_future_value
    
    return value

# Test the reward functions
print("Testing reward functions:\n")
mujoco.mj_resetData(model, data)

# Move forward
data.ctrl[:] = [1.0, 1.0, 1.0, 1.0]
for _ in range(50):
    mujoco.mj_step(model, data)

state = {'x': data.qpos[0], 'y': data.qpos[1], 'z': data.qpos[2]}

# Reset previous_x before testing
previous_x_p1 = 0.0
previous_x_p2 = 0.0

r1, _, _ = reward_p1(state)
r2, _, _ = reward_p2(state)

print(f"Position: x={state['x']:.2f}m")
print(f"reward_p1 (no backward penalty): {r1:.2f}")
print(f"reward_p2 (with backward penalty): {r2:.2f}")

# Reset before computing value
previous_x_p2 = 0.0
print(f"Value estimate (gamma=0.95): {compute_value(state, [1,1,1,1]):.2f}")


# Question 12 - Trajectory analysis with value accumulation

# Step 1: Create a simple control sequence (move forward)
control_sequence = []
for i in range(100):
    # Simple forward movement
    control_sequence.append([1.0, 1.0, 1.0, 1.0])

# Step 2: Simulate trajectory and compute rewards at each step
mujoco.mj_resetData(model, data)
# Reset global previous_x for reward functions
previous_x = data.qpos[0]

trajectory = []
rewards_p1 = []
rewards_p2 = []
cumulative_p1 = []
cumulative_p2 = []

cum_p1 = 0.0
cum_p2 = 0.0

for action in control_sequence:
    # Apply action
    data.ctrl[:] = action
    for _ in range(10):
        mujoco.mj_step(model, data)
    
    # Record state
    state = {'x': data.qpos[0], 'y': data.qpos[1], 'z': data.qpos[2]}
    trajectory.append(state.copy())
    
    # Compute rewards
    r1, term1, _ = reward_p1(state)
    r2, term2, _ = reward_p2(state)
    
    rewards_p1.append(r1)
    rewards_p2.append(r2)
    
    cum_p1 += r1
    cum_p2 += r2
    
    cumulative_p1.append(cum_p1)
    cumulative_p2.append(cum_p2)
    
    if term1 or term2:
        break

# Step 3 & 4: Plot comparison of accumulated values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_p1, label='reward_p1 (no backward penalty)', alpha=0.7)
plt.plot(rewards_p2, label='reward_p2 (with backward penalty)', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Instantaneous Rewards')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cumulative_p1, label='Cumulative reward_p1', linewidth=2)
plt.plot(cumulative_p2, label='Cumulative reward_p2', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.title('Accumulated Value Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nFinal cumulative rewards:")
print(f"  reward_p1: {cum_p1:.2f}")
print(f"  reward_p2: {cum_p2:.2f}")
print(f"  Final position: x={trajectory[-1]['x']:.2f}m")


# Question 13 - Discount factor visualization

# Part A: Real-time plotting integration
# Note: For real-time plotting, you would modify t01_4_wheels_robot_pilot.py
# Here we'll demonstrate the concept in the notebook

# Part B: Compare different gamma values
gamma_values = [1.0, 0.95, 0.9, 0.5]

# Compute discounted cumulative rewards for each gamma
discounted_values = {}

for gamma in gamma_values:
    discounted_cum = []
    cum_value = 0.0
    
    for t, reward in enumerate(rewards_p1):
        cum_value += (gamma ** t) * reward
        discounted_cum.append(cum_value)
    
    discounted_values[gamma] = discounted_cum

# Plot comparison
plt.figure(figsize=(12, 6))

for gamma in gamma_values:
    plt.plot(discounted_values[gamma], label=f'γ = {gamma}', linewidth=2)

plt.xlabel('Step', fontsize=12)
plt.ylabel('Discounted Cumulative Value', fontsize=12)
plt.title('Impact of Discount Factor γ on Value Accumulation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nFinal discounted values:")
for gamma in gamma_values:
    print(f"  γ = {gamma}: {discounted_values[gamma][-1]:.2f}")

print("\nObservations:")
print("- γ = 1.0: No discounting, all future rewards count equally")
print("- γ = 0.95: Standard RL value, slight preference for immediate rewards")
print("- γ = 0.9: More myopic, stronger preference for near-term rewards")
print("- γ = 0.5: Very short-sighted, distant rewards barely matter")


# Question 14 - Implement canonical RL API

class CorridorEnv:
    """Canonical RL environment wrapper for MuJoCo corridor."""
    
    def __init__(self, model_obj, data_obj, max_steps=1000):
        """Initialize the environment.
        
        Args:
            model_obj: Existing MuJoCo model object
            data_obj: Existing MuJoCo data object
            max_steps: Maximum steps before truncation
        """
        # Use existing MuJoCo model and data
        self.model = model_obj
        self.data = data_obj
        
        # Store max_steps
        self.max_steps = max_steps
        
        # Initialize step counter
        self.current_step = 0
        
        # Track previous position for reward calculation
        self.previous_x = 0.0
    
    def reset(self, seed=None):
        """Reset environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            initial_state: Initial observation (numpy array)
            info: Dictionary with initial information
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset previous position (after reset, robot is at starting position)
        self.previous_x = self.data.qpos[0] if len(self.data.qpos) > 0 else 0.0
        
        # Get initial state (position + velocity)
        initial_state = np.concatenate([self.data.qpos[:3], self.data.qvel[:3]])
        
        # Create info dict
        info = {
            'position': (self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]),
            'step': 0
        }
        
        return initial_state, info
    
    def step(self, action):
        """Take one step in the environment.
        
        Args:
            action: Numpy array of wheel controls [4]
            
        Returns:
            next_state: Next observation (numpy array)
            reward: Immediate reward (float)
            terminated: Whether episode ended naturally (bool)
            truncated: Whether episode was cut short (bool)
            info: Dictionary with diagnostic information
        """
        # Apply action to MuJoCo controls
        self.data.ctrl[:] = action
        
        # Step MuJoCo simulation (10 substeps for smoother physics)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Increment step counter
        self.current_step += 1
        
        # Get next state
        next_state = np.concatenate([self.data.qpos[:3], self.data.qvel[:3]])
        
        # Compute reward using reward_p2
        robot_x = self.data.qpos[0]
        robot_z = self.data.qpos[2]
        
        # Terminal rewards
        if robot_x >= CORRIDOR_LENGTH:
            reward = +100.0
            terminated = True
            reason = "SUCCESS: Reached goal!"
        elif robot_z < 0.1 and robot_x > 0:
            reward = -100.0
            terminated = True
            reason = "FAILURE: Fell in hole!"
        elif robot_x < -1.0:
            reward = -50.0
            terminated = True
            reason = "FAILURE: Went backward!"
        else:
            # Progress reward
            delta_x = robot_x - self.previous_x
            self.previous_x = robot_x
            reward = delta_x * 10.0
            terminated = False
            reason = None
        
        # Check if truncated (max_steps reached)
        truncated = self.current_step >= self.max_steps
        
        # Build info dict
        info = {
            'position': (self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]),
            'velocity': (self.data.qvel[0], self.data.qvel[1], self.data.qvel[2]),
            'step': self.current_step,
            'termination_reason': reason
        }
        
        return next_state, reward, terminated, truncated, info

# Part C: Test with canonical training loop
print("\n=== Testing Canonical RL API ===\n")

# Create environment instance using existing model and data
env = CorridorEnv(model, data, max_steps=500)

# Random agent
def random_agent():
    """Generate random action in valid range [-1, 1] for each wheel."""
    return np.random.uniform(-1.0, 1.0, size=4)

# Run 5 episodes
num_episodes = 5

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done:
        # Agent selects action
        action = random_agent()
        
        # Environment steps
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Check if episode is over
        done = terminated or truncated
        
        # Update state and accumulate reward
        state = next_state
        episode_reward += reward
        steps += 1
    
    # Print episode statistics
    final_x = info['position'][0]
    reason = info['termination_reason'] if terminated else 'truncated'
    print(f"Episode {episode + 1}:")
    print(f"  Total Reward: {episode_reward:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Final Position: x={final_x:.2f}m")
    print(f"  Termination: {reason}")
    print()

print("\n✓ Canonical RL API implementation complete!")


