#!/usr/bin/env python3
"""Preprocess MPtrj JSON file into individual entry files for efficient lazy loading."""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any
import argparse

def preprocess_mptrj_json(input_file: str, output_dir: str, max_entries: int = None):
    """Preprocess MPtrj JSON file into individual entry files.
    
    Args:
        input_file: Path to the large MPtrj JSON file
        output_dir: Directory to store individual entry files
        max_entries: Maximum number of entries to process (for testing)
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preprocessing {input_file} into {output_dir}")
    print(f"Input file size: {input_path.stat().st_size / (1024**3):.2f} GB")
    
    # Process the JSON file
    entry_count = 0
    processed_count = 0
    
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Loaded JSON data with {len(raw_data)} parent structures")
    
    # Process each parent structure
    for parent_id, structures in raw_data.items():
        if not isinstance(structures, dict):
            print(f"Warning: Skipping malformed parent entry: {parent_id}")
            continue
        
        for structure_id, structure_data in structures.items():
            if max_entries and processed_count >= max_entries:
                break
                
            # Create entry file
            entry_file = output_path / f"{parent_id}_{structure_id}.json"
            
            # Prepare entry data
            entry_data = {
                'parent_id': parent_id,
                'structure_id': structure_id,
                'structure_data': structure_data
            }
            
            # Write individual entry file
            with open(entry_file, 'w') as f:
                json.dump(entry_data, f, indent=2)
            
            processed_count += 1
            
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} entries...")
        
        entry_count += len(structures)
        
        if max_entries and processed_count >= max_entries:
            break
    
    print(f"Preprocessing complete!")
    print(f"Total entries processed: {processed_count}")
    print(f"Output directory: {output_dir}")
    print(f"Average file size: {sum(f.stat().st_size for f in output_path.glob('*.json')) / processed_count / 1024:.2f} KB")

def main():
    parser = argparse.ArgumentParser(description="Preprocess MPtrj JSON file into individual entry files")
    parser.add_argument("input_file", help="Path to MPtrj JSON file")
    parser.add_argument("output_dir", help="Output directory for entry files")
    parser.add_argument("--max-entries", type=int, help="Maximum number of entries to process (for testing)")
    
    args = parser.parse_args()
    
    try:
        preprocess_mptrj_json(args.input_file, args.output_dir, args.max_entries)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

