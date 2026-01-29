#!/usr/bin/env python3
"""
Analyse du pattern des bumps dans le corridor pour comprendre le biais gauche/droite.
"""
import re

def analyze_corridor_pattern():
    # Read the corridor XML file
    with open('ppo_small_robot_steer/corridor_3x100_no_full_obstacles.xml', 'r') as f:
        content = f.read()

    # Extract all bump positions
    bump_pattern = r'floor_bump_\d+.*?pos="([0-9.]+) ([0-9.-]+) [0-9.]+"'
    bumps = re.findall(bump_pattern, content)

    print('ANALYSE DU PATTERN DES BUMPS:')
    print('=' * 50)
    print('X (distance) | Y (position latérale)')
    print('-' * 50)

    left_bumps = 0  # Y < 0 (côté gauche)
    right_bumps = 0  # Y > 0 (côté droit)

    for i, (x, y) in enumerate(bumps):
        x_val = float(x)
        y_val = float(y)
        side = 'GAUCHE' if y_val < 0 else 'DROITE'
        print(f'{x_val:8.2f}m   | {y_val:6.2f}m ({side})')
        
        if y_val < 0:
            left_bumps += 1
        else:
            right_bumps += 1

    print('-' * 50)
    print(f'TOTAL: {len(bumps)} bumps')
    print(f'Côté GAUCHE (Y < 0): {left_bumps} bumps ({100*left_bumps/len(bumps):.1f}%)')
    print(f'Côté DROITE (Y > 0): {right_bumps} bumps ({100*right_bumps/len(bumps):.1f}%)')
    print()

    # Analyser le pattern de navigation optimal
    print('PATTERN DE NAVIGATION OPTIMAL:')
    print('=' * 50)
    for i, (x, y) in enumerate(bumps):
        x_val = float(x)
        y_val = float(y)
        
        if y_val < 0:  # Bump à gauche
            optimal_move = 'TOURNER À DROITE'
        else:  # Bump à droite
            optimal_move = 'TOURNER À GAUCHE'
        
        print(f'Bump {i+1:2d} à {x_val:6.2f}m, Y={y_val:6.2f}m → {optimal_move}')

    # Compter les mouvements optimaux
    left_turns_needed = sum(1 for _, y in bumps if float(y) > 0)  # Bump à droite = tourner à gauche
    right_turns_needed = sum(1 for _, y in bumps if float(y) < 0)  # Bump à gauche = tourner à droite

    print('-' * 50)
    print(f'MOUVEMENTS OPTIMAUX REQUIS:')
    print(f'Tourner à GAUCHE: {left_turns_needed} fois ({100*left_turns_needed/len(bumps):.1f}%)')
    print(f'Tourner à DROITE: {right_turns_needed} fois ({100*right_turns_needed/len(bumps):.1f}%)')

    if left_turns_needed > right_turns_needed:
        bias = 'GAUCHE'
        diff = left_turns_needed - right_turns_needed
    else:
        bias = 'DROITE'
        diff = right_turns_needed - left_turns_needed

    print(f'BIAIS: Le corridor favorise les virages à {bias} (+{diff} virages)')
    
    return left_turns_needed, right_turns_needed, bias, diff

if __name__ == "__main__":
    analyze_corridor_pattern()