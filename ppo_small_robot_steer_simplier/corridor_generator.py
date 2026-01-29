"""
Générateur de corridors aléatoires pour l'entraînement RL.
Génère le XML MuJoCo dynamiquement (comme generate_corridors.py).

Règles:
- Cellules de 0.5m × 0.5m
- Corridor de 100m × 3m
- Passage minimum de 2 cellules de large (1m)
- Trous minimum de 2×2 cellules

Zones:
- 0-5m: flat uniquement (spawn)
- 5-20m: flat + trous
- 20-40m: flat + bumps larges
- 40-60m: flat + bumps moyens
- 60-100m: flat + trous + bumps petits
"""
import random

# Paramètres du corridor
CELL_SIZE = 0.5
CORRIDOR_LENGTH_M = 100.0
CORRIDOR_WIDTH_M = 3.0
NUM_CELLS_X = int(CORRIDOR_LENGTH_M / CELL_SIZE)  # 200
NUM_CELLS_Y = int(CORRIDOR_WIDTH_M / CELL_SIZE)   # 6

# Bump settings (identiques à generate_corridors.py)
BUMP_SETTINGS = {
    "small": {"half_z": 0.025, "mat_name": "mat_bump_small"},
    "medium": {"half_z": 0.1, "mat_name": "mat_bump_medium"},
    "large": {"half_z": 0.25, "mat_name": "mat_bump_large"}
}

BASE_FLAT_HALF_Z = 0.025


def generate_corridor_grid(seed=None):
    """Génère une grille de corridor aléatoire."""
    if seed is not None:
        random.seed(seed)
    
    # Initialiser tout en flat
    grid = [['flat' for _ in range(NUM_CELLS_Y)] for _ in range(NUM_CELLS_X)]
    
    # Zone 0: 0-5m (0-10 cellules) - FLAT UNIQUEMENT (spawn)
    # Déjà flat par défaut
    
    # Zone 1: 5-19m (10-38 cellules) - flat + trous
    passage_y = _generate_zone_with_obstacles(grid, 10, 38, obstacle_type='hole')
    # Transition 19-21m (38-42 cellules) - 1m FLAT
    
    # Zone 2: 21-39m (42-78 cellules) - flat + bumps larges
    passage_y = _generate_zone_with_obstacles(grid, 42, 78, obstacle_type='bump', bump_type="large", initial_passage_y=passage_y)
    # Transition 39-41m (78-82 cellules) - 1m FLAT
    
    # Zone 3: 41-59m (82-118 cellules) - flat + bumps moyens
    passage_y = _generate_zone_with_obstacles(grid, 82, 118, obstacle_type='bump', bump_type="medium", initial_passage_y=passage_y)
    # Transition 59-61m (118-122 cellules) - 1m FLAT
    
    # Zone 4: 61-100m (122-200 cellules) - flat + trous + bumps petits
    _generate_zone_mixed(grid, 122, 200, initial_passage_y=passage_y)
    
    return grid


def _generate_zone_with_obstacles(grid, start_x, end_x, obstacle_type='hole', bump_type=None, initial_passage_y=None):
    """Génère des obstacles dans une zone avec passage garanti."""
    x = start_x
    
    if initial_passage_y is None:
        passage_y = random.randint(0, NUM_CELLS_Y - 2)
    else:
        passage_y = initial_passage_y
    
    while x < end_x:
        if random.random() < 0.75:
            obs_length = random.randint(2, 4)
            obs_width = random.randint(2, min(4, NUM_CELLS_Y - 2))
            
            possible_y_starts = []
            for y in range(NUM_CELLS_Y - obs_width + 1):
                obs_end = y + obs_width
                passage_end = passage_y + 2
                if obs_end <= passage_y or y >= passage_end:
                    possible_y_starts.append(y)
            
            if possible_y_starts:
                y_start = random.choice(possible_y_starts)
                
                for dx in range(obs_length):
                    for dy in range(obs_width):
                        xi = x + dx
                        yi = y_start + dy
                        if start_x <= xi < end_x and 0 <= yi < NUM_CELLS_Y:
                            if obstacle_type == 'hole':
                                grid[xi][yi] = 'hole'
                            else:
                                grid[xi][yi] = ('bump', bump_type)
                
                x += obs_length
            
            gap = random.randint(2, 4)
            x += gap
            
            if random.random() < 0.6:
                passage_y = random.randint(0, NUM_CELLS_Y - 2)
        else:
            x += 1
    
    return passage_y


def _generate_zone_mixed(grid, start_x, end_x, initial_passage_y=None):
    """Génère une zone mixte avec trous et petits bumps."""
    x = start_x
    
    if initial_passage_y is None:
        passage_y = random.randint(0, NUM_CELLS_Y - 2)
    else:
        passage_y = initial_passage_y
    
    while x < end_x:
        if random.random() < 0.7:
            if random.random() < 0.5:
                obstacle_type = 'hole'
                bump_type = None
            else:
                obstacle_type = 'bump'
                bump_type = 'small'
            
            obs_length = random.randint(2, 4)
            obs_width = random.randint(2, min(4, NUM_CELLS_Y - 2))
            
            possible_y_starts = []
            for y in range(NUM_CELLS_Y - obs_width + 1):
                obs_end = y + obs_width
                passage_end = passage_y + 2
                if obs_end <= passage_y or y >= passage_end:
                    possible_y_starts.append(y)
            
            if possible_y_starts:
                y_start = random.choice(possible_y_starts)
                
                for dx in range(obs_length):
                    for dy in range(obs_width):
                        xi = x + dx
                        yi = y_start + dy
                        if start_x <= xi < end_x and 0 <= yi < NUM_CELLS_Y:
                            if obstacle_type == 'hole':
                                grid[xi][yi] = 'hole'
                            else:
                                grid[xi][yi] = ('bump', bump_type)
                
                x += obs_length
            
            gap = random.randint(2, 4)
            x += gap
            
            if random.random() < 0.6:
                passage_y = random.randint(0, NUM_CELLS_Y - 2)
        else:
            x += 1
    
    return passage_y


def grid_to_xml_string(grid):
    """Convertit la grille en string XML MuJoCo (sans écrire de fichier)."""
    
    bump_materials = ""
    for key, settings in BUMP_SETTINGS.items():
        mat_name = settings["mat_name"]
        if key == "small":
            rgb = "1 1 0.2"
        elif key == "medium":
            rgb = "1 0.5 0"
        else:
            rgb = "0.8 0 0"
        bump_materials += f'    <material name="{mat_name}" texture="tex_wall" rgba="{rgb} 1" specular="0.5" shininess="0.5" />\n'
    
    xml_header = f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="corridor_{int(CORRIDOR_WIDTH_M)}x{int(CORRIDOR_LENGTH_M)}">
  <compiler angle="degree" autolimits="true" />
  <option timestep="0.005" gravity="0 0 -9.81" />
  <size njmax="4000" nconmax="1000" />
  <asset>
    <texture name="tex_grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.15 0.15 0.15" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8" />
    <texture name="tex_wall" type="2d" builtin="checker" rgb1="0.5 0.5 0.5" rgb2="0.55 0.55 0.55" width="300" height="300" />
    <material name="mat_floor" texture="tex_grid" texrepeat="2 2" specular="0.1" shininess="0.1" />
{bump_materials.rstrip()}
  </asset>
  <worldbody>
    <light name="key_light" pos="4 4 4" dir="-1 -1 -1" directional="true" diffuse="1.1 1.1 1.1" specular="0.4 0.4 0.4" castshadow="true" />
    <light name="fill_light" pos="-6 4 2" dir="1 -0.5 -1" directional="true" ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" castshadow="false" />

"""
    
    xml_geoms = []
    flat_counter = 0
    bump_counter = 0
    
    y_offset = -CORRIDOR_WIDTH_M / 2
    half_cell = CELL_SIZE / 2.0
    z_floor_top = BASE_FLAT_HALF_Z * 2.0
    
    for x_idx in range(NUM_CELLS_X):
        for y_idx in range(NUM_CELLS_Y):
            cell = grid[x_idx][y_idx]
            
            center_x = (x_idx * CELL_SIZE) + half_cell
            center_y = (y_idx * CELL_SIZE) + half_cell + y_offset
            
            if cell == 'flat':
                geom = f'    <geom type="box" material="mat_floor" size="{half_cell:.3f} {half_cell:.3f} {BASE_FLAT_HALF_Z:.3f}" pos="{center_x:.3f} {center_y:.3f} {BASE_FLAT_HALF_Z:.3f}" name="flat_{flat_counter}" />'
                xml_geoms.append(geom)
                flat_counter += 1
            
            elif isinstance(cell, tuple) and cell[0] == 'bump':
                bump_type = cell[1]
                settings = BUMP_SETTINGS[bump_type]
                half_z = settings["half_z"]
                mat_name = settings["mat_name"]
                center_z = z_floor_top + half_z
                
                geom = f'    <geom type="box" material="{mat_name}" size="{half_cell:.3f} {half_cell:.3f} {half_z:.3f}" pos="{center_x:.3f} {center_y:.3f} {center_z:.3f}" name="bump_{bump_counter}" />'
                xml_geoms.append(geom)
                bump_counter += 1
            
            # 'hole' = pas de géométrie (le robot tombe)
    
    xml_footer = """
  </worldbody>
</mujoco>"""
    
    return xml_header + "\n".join(xml_geoms) + xml_footer


def grid_to_cell_map(grid, target_cell_size):
    """
    Convertit la grille de génération (0.5m) vers une cell_map à la résolution cible.
    
    Args:
        grid: grille générée (cellules 0.5m)
        target_cell_size: taille des cellules cibles (ex: 0.1m)
    
    Returns:
        cell_map: dict {(row, col): cell_type} où cell_type est 0=flat, 1=bump, 2=hole
    """
    cell_map = {}
    
    scale = CELL_SIZE / target_cell_size  # ex: 0.5/0.1 = 5
    
    n_rows = int(CORRIDOR_LENGTH_M / target_cell_size) + 100
    n_cols = int(CORRIDOR_WIDTH_M / target_cell_size)
    
    # Initialiser tout en trou par défaut
    for r in range(n_rows):
        for c in range(n_cols):
            cell_map[(r, c)] = 2
    
    # Convertir la grille
    for gen_x in range(NUM_CELLS_X):
        for gen_y in range(NUM_CELLS_Y):
            cell_value = grid[gen_x][gen_y]
            
            if cell_value == 'flat':
                cell_type = 0
            elif cell_value == 'hole':
                cell_type = 2
            elif isinstance(cell_value, tuple) and cell_value[0] == 'bump':
                cell_type = 1
            else:
                cell_type = 2
            
            start_row = int(gen_x * scale)
            end_row = int((gen_x + 1) * scale)
            start_col = int(gen_y * scale)
            end_col = int((gen_y + 1) * scale)
            
            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        cell_map[(r, c)] = cell_type
    
    return cell_map
