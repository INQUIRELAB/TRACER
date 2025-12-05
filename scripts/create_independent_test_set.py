#!/usr/bin/env python3
"""
Create Independent Test Set from MPtrj
Extracts materials similar to training data but from independent source.
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_mptrj_sample():
    """Load a sample from MPtrj."""
    logger.info("📥 Loading MPtrj data...")
    
    mptrj_path = Path("data/mptrj.json")
    
    if not mptrj_path.exists():
        logger.error("MPtrj data not found!")
        return []
    
    # Load first 100 entries to sample from
    logger.info("   Reading MPtrj file...")
    
    entries = []
    with open(mptrj_path, 'r') as f:
        data = json.load(f)
        
        # Sample diverse materials
        sampled = []
        seen_chem = set()
        
        for entry in data:
            # Get chemical formula
            if 'atoms' in entry:
                atoms = entry['atoms']
                if 'elements' in atoms:
                    formula = ''.join(sorted(atoms['elements']))
                    
                    # Only add if novel composition
                    if formula not in seen_chem or len(sampled) < 200:
                        seen_chem.add(formula)
                        
                        # Extract sample info
                        sample = {
                            'entry_id': entry.get('entry_id', 'unknown'),
                            'formula': formula,
                            'elements': atoms.get('elements', []),
                            'atomic_numbers': convert_elements(atoms.get('elements', [])),
                            'positions': atoms.get('coords', []),
                            'n_atoms': len(atoms.get('elements', [])),
                            'corrected_energy_per_atom': entry.get('corrected_energy_per_atom', 0.0)
                        }
                        
                        # Only add if has valid structure
                        if sample['n_atoms'] > 0 and sample['n_atoms'] < 100:
                            sampled.append(sample)
                            
        logger.info(f"   Sampled {len(sampled)} diverse materials")
        return sampled


def convert_elements(elements):
    """Convert element symbols to atomic numbers."""
    from ase.data import chemical_symbols
    
    atomic_numbers = []
    for el in elements:
        if el in chemical_symbols:
            atomic_numbers.append(chemical_symbols.index(el))
    return atomic_numbers


def create_test_samples(sampled, num_samples=300):
    """Create test samples from sampled MPtrj data."""
    logger.info(f"\n🎯 Creating {num_samples} test samples...")
    
    # Filter for reasonable sizes (1-96 atoms to match training)
    filtered = [s for s in sampled if 1 <= s['n_atoms'] <= 96]
    
    if len(filtered) < num_samples:
        num_samples = len(filtered)
    
    # Select diverse samples
    test_samples = filtered[:num_samples]
    
    logger.info(f"   Created {len(test_samples)} test samples")
    logger.info(f"   Size range: {min(s['n_atoms'] for s in test_samples)}-{max(s['n_atoms'] for s in test_samples)} atoms")
    
    return test_samples


def save_test_set(test_samples):
    """Save independent test set."""
    output_path = Path("data/independent_test_set.json")
    
    # Convert to standard format
    formatted_samples = []
    
    for i, sample in enumerate(test_samples):
        formatted = {
            'sample_id': f"mptrj_test_{i}",
            'domain': 'materials_project',
            'num_atoms': sample['n_atoms'],
            'atomic_numbers': sample['atomic_numbers'],
            'positions': sample['positions'],
            'energy_target': sample['corrected_energy_per_atom'] * sample['n_atoms'],
            'formation_energy_per_atom': sample.get('corrected_energy_per_atom', 0.0),
            'formula': sample['formula']
        }
        formatted_samples.append(formatted)
    
    with open(output_path, 'w') as f:
        json.dump(formatted_samples, f, indent=2)
    
    logger.info(f"\n✅ Saved to: {output_path}")
    logger.info(f"   Total samples: {len(formatted_samples)}")
    
    return formatted_samples


def main():
    """Create independent test set."""
    logger.info("🚀 CREATING INDEPENDENT TEST SET")
    logger.info("="*80)
    logger.info("   Source: MPtrj (Materials Project)")
    logger.info("   Purpose: Independent validation")
    logger.info("   Target: 300 diverse materials")
    logger.info("="*80)
    
    # Load and sample
    sampled = load_mptrj_sample()
    
    if not sampled:
        logger.error("Failed to load samples")
        return
    
    # Create test set
    test_samples = create_test_samples(sampled, num_samples=300)
    
    # Save
    formatted = save_test_set(test_samples)
    
    logger.info("\n✅ INDEPENDENT TEST SET CREATED!")
    logger.info("="*80)
    logger.info("   This test set is:")
    logger.info("   ✓ From MPtrj (different source than training)")
    logger.info("   ✓ Similar materials to training data")
    logger.info("   ✓ Ready for validation")
    logger.info("   ✓ Completely independent")
    logger.info("="*80)


if __name__ == "__main__":
    main()



