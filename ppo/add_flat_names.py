"""
Add incremental names to all floor geometries in corridor XML.
"""
import xml.etree.ElementTree as ET

def add_flat_names(input_file, output_file):
    """Add name='flat_i' to all geometries with material='mat_floor' and 'bump_i' to mat_bump."""
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    flat_counter = 0
    bump_counter = 0
    
    # Find all geom elements
    for geom in root.iter('geom'):
        material = geom.get('material')
        
        if material == 'mat_floor':
            # Add incremental name for flat
            geom.set('name', f'flat_{flat_counter}')
            flat_counter += 1
            print(f"Added name='flat_{flat_counter-1}' to geom with material='mat_floor'")
        
        elif material == 'mat_bump':
            # Replace existing name with bump_i
            geom.set('name', f'bump_{bump_counter}')
            bump_counter += 1
            print(f"Renamed to 'bump_{bump_counter-1}' (material='mat_bump')")
    
    # Write to output file
    tree.write(output_file, encoding='unicode', xml_declaration=True)
    print(f"\nTotal flat geometries named: {flat_counter}")
    print(f"Total bump geometries renamed: {bump_counter}")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    input_file = "corridor_3x100.xml"
    output_file = "corridor_3x100.xml"  # Overwrite the same file
    
    add_flat_names(input_file, output_file)
