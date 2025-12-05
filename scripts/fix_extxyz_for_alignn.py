#!/usr/bin/env python3
"""Fix extxyz format for ALIGNN compatibility."""

from pathlib import Path
import re

def fix_extxyz_file(input_file, output_file):
    """Fix extxyz format to be ALIGNN compatible."""
    with open(input_file, 'r') as f_in:
        content = f_in.read()
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.isdigit():  # Number of atoms
            n_atoms = int(line)
            new_lines.append(str(n_atoms))
            i += 1
            
            if i < len(lines) and lines[i].strip():
                # Parse properties line
                props_line = lines[i].strip()
                
                # Extract energy value
                energy_match = re.search(r'energy=([\d\.\-\+eE]+)', props_line)
                if energy_match:
                    energy = energy_match.group(1)
                    # ALIGNN expects: properties=species:S:1:pos:R:3 energy=___ 
                    new_lines.append(f'energy={energy} structure_id={i//(n_atoms+2)}')
                else:
                    new_lines.append(props_line)
                
                i += 1
            
            # Copy atom lines
            for _ in range(n_atoms):
                if i < len(lines) and lines[i].strip():
                    new_lines.append(lines[i].strip())
                    i += 1
        
        i += 1
    
    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(new_lines))

# Fix all files
for name in ['train', 'val', 'test']:
    input_file = Path(f'/home/arash/dft/data/alignn_unified/{name}.extxyz')
    output_file = Path(f'/home/arash/dft/data/alignn_unified/{name}_fixed.extxyz')
    
    print(f"Fixing {name}...")
    fix_extxyz_file(input_file, output_file)
    print(f"Done: {output_file}")



