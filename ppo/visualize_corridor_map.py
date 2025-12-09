"""
Visualize the entire corridor cell map as the robot would see it.
"""
from robot_corridor_env_new import RobotCorridorEnv
import numpy as np

def visualize_full_corridor():
    """Print the entire corridor cell map."""
    # Create environment
    env = RobotCorridorEnv()
    
    # Get the cell map
    cell_map = env.cell_map_semantic
    
    # Find dimensions
    max_row = max(r for r, c in cell_map.keys())
    max_col = max(c for r, c in cell_map.keys())
    
    print("="*80)
    print("CORRIDOR CELL MAP")
    print("="*80)
    print(f"Dimensions: {max_row+1} rows × {max_col+1} cols")
    print(f"Cell size: {env.cell_width}m × {env.cell_width}m")
    print(f"Total length: {(max_row+1) * env.cell_width}m")
    print(f"Total width: {(max_col+1) * env.cell_width}m")
    print()
    print("Legend: . = flat (0)  # = bump (1)  X = hole (2)")
    print("="*80)
    print()
    
    # Print the map
    for row in range(max_row + 1):
        line = f"Row {row:3d} ({row*env.cell_width:5.1f}m): "
        for col in range(max_col + 1):
            cell_type = cell_map.get((row, col), 0)
            if cell_type == 0:
                line += "."
            elif cell_type == 1:
                line += "#"
            elif cell_type == 2:
                line += "X"
            else:
                line += "?"
        print(line)
    
    print()
    print("="*80)
    
    # Statistics
    total_cells = (max_row + 1) * (max_col + 1)
    flat_count = sum(1 for v in cell_map.values() if v == 0)
    bump_count = sum(1 for v in cell_map.values() if v == 1)
    hole_count = sum(1 for v in cell_map.values() if v == 2)
    
    print("STATISTICS:")
    print(f"  Total cells: {total_cells}")
    print(f"  Flat cells:  {flat_count} ({100*flat_count/total_cells:.1f}%)")
    print(f"  Bump cells:  {bump_count} ({100*bump_count/total_cells:.1f}%)")
    print(f"  Hole cells:  {hole_count} ({100*hole_count/total_cells:.1f}%)")
    print("="*80)
    
    env.close()

if __name__ == "__main__":
    visualize_full_corridor()
