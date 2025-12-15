#!/usr/bin/env python3
"""
Debug script to see what the robot actually observes.
"""
import numpy as np
from robot_corridor_env_new import RobotCorridorEnv

def debug_observation():
    """Debug what the robot sees."""
    env = RobotCorridorEnv(corridor_xml="corridor_3x100.xml")
    
    print("Environment created")
    print(f"Corridor length: {env.corridor_length}")
    print(f"Corridor width: {env.corridor_width}")
    print(f"Cell width: {env.cell_width}")
    print(f"Observation rows: {env.n_observation_rows}")
    print(f"Observation cols: {env.n_cols}")
    print(f"Expected obs size: {6 + env.n_observation_rows * env.n_cols}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nActual obs size: {len(obs)}")
    print(f"Robot position: x={info['x_position']:.2f}, y={info['y_position']:.2f}, z={info['z_position']:.2f}")
    
    # Extract observations
    robot_state = obs[:6]  # x, y, z, vx, vy, vz
    wheel_contact = obs[6:10]  # cell type under each wheel
    grid_obs = obs[10:].reshape(env.n_observation_rows, env.n_cols)
    
    print(f"\nRobot state: {robot_state}")
    print(f"Wheel contact: {wheel_contact} (0=flat, 1=ramp, 2=hole, 3=bump)")
    print(f"  FL={wheel_contact[0]}, FR={wheel_contact[1]}, RL={wheel_contact[2]}, RR={wheel_contact[3]}")
    print(f"Grid shape: {grid_obs.shape}")
    
    # Calculate robot and wheel positions FIRST
    robot_x, robot_y = info['x_position'], info['y_position']
    robot_row = int(robot_x / env.cell_width)
    robot_col = int((robot_y + env.corridor_width/2) / env.cell_width)
    
    # Calculate wheel positions
    quat = env.data.qpos[3:7]  # w, x, y, z
    robot_heading = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                             1 - 2*(quat[2]**2 + quat[3]**2))
    
    wheelbase = 0.70
    track_width = 0.60
    cos_h = np.cos(robot_heading)
    sin_h = np.sin(robot_heading)
    
    wheel_offsets = [
        (wheelbase/2, -track_width/2),   # Front-left
        (wheelbase/2, track_width/2),    # Front-right  
        (-wheelbase/2, -track_width/2),  # Rear-left
        (-wheelbase/2, track_width/2),   # Rear-right
    ]
    
    wheel_positions = []
    for offset_x, offset_y in wheel_offsets:
        world_x = robot_x + offset_x * cos_h - offset_y * sin_h
        world_y = robot_y + offset_x * sin_h + offset_y * cos_h
        wheel_row = int(world_x / env.cell_width)
        wheel_col = int((world_y + env.corridor_width/2) / env.cell_width)
        wheel_positions.append((wheel_row, wheel_col))
    
    # Print grid (0=flat, 1=ramp, 2=hole, 3=bump) with robot positions
    print("\nGrid observation (0=flat, 1=ramp, 2=hole, 3=bump):")
    print("Rows: behind <- robot <- ahead")
    
    for i, row in enumerate(grid_obs):
        row_type = "BEHIND" if i < env.n_rows_behind else ("ROBOT" if i == env.n_rows_behind else "AHEAD")
        
        # Create display row with robot positions marked
        display_row = []
        for j, cell_val in enumerate(row):
            # Calculate actual world row for this observation row
            actual_row = robot_row + (i - env.n_rows_behind)
            
            # Check if any wheel or center is at this position
            marker = str(int(cell_val))
            if (actual_row, j) == (robot_row, robot_col):
                marker = 'C'  # Center
            elif (actual_row, j) in wheel_positions:
                wheel_idx = wheel_positions.index((actual_row, j))
                if wheel_idx < 2:
                    marker = 'F'  # Front wheel
                else:
                    marker = 'R'  # Rear wheel
            
            display_row.append(marker)
        
        print(f"Row {i:2d} ({row_type:6s}): {' '.join(display_row)}")
    
    # Print detailed wheel information
    wheel_names = ['FL', 'FR', 'RL', 'RR']
    print(f"\nRobot grid position: row={robot_row}, col={robot_col}")
    print(f"Robot center: x={robot_x:.3f}, y={robot_y:.3f}, heading={np.degrees(robot_heading):.1f}°")
    print(f"\nWheel positions:")
    
    for i, (wheel_row, wheel_col) in enumerate(wheel_positions):
        cell_type = env.cell_map_semantic.get((wheel_row, wheel_col), 2)
        print(f"  {wheel_names[i]}: grid=({wheel_row}, {wheel_col}) cell={int(cell_type)}")
    
    # Check cell map around robot with wheel positions marked
    print("\nCell map around robot (C=center, F=front wheels, R=rear wheels):")
    for r in range(robot_row-2, robot_row+8):
        row_cells = []
        for c in range(env.n_cols):
            cell_type = env.cell_map_semantic.get((r, c), 2)
            
            # Mark special positions
            marker = str(int(cell_type))
            if (r, c) == (robot_row, robot_col):
                marker = 'C'  # Center
            elif (r, c) in wheel_positions:
                wheel_idx = wheel_positions.index((r, c))
                if wheel_idx < 2:
                    marker = 'F'  # Front wheel
                else:
                    marker = 'R'  # Rear wheel
            
            row_cells.append(marker)
        print(f"Row {r:3d}: {' '.join(row_cells)}")
    
    env.close()

if __name__ == "__main__":
    debug_observation()