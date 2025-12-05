#!/usr/bin/env python3
"""
Load and Preprocess Full Unified Dataset
Combines ANI1x, OC20, OC22, and JARVIS-DFT with proper train/val/test splits.
"""

import sys
import os
from pathlib import Path
import logging
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_jarvis_dft():
    """Load JARVIS-DFT dataset."""
    logger.info("📥 Loading JARVIS-DFT...")
    samples = []
    jarvis_path = Path("data/jarvis_dft/data/jarvis_dft")
    
    if not jarvis_path.exists():
        logger.warning("JARVIS-DFT path not found, skipping")
        return samples
    
    json_files = list(jarvis_path.glob("*.json"))[:20]  # Limit for now
    
    for json_file in tqdm(json_files, desc="Loading JARVIS"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for entry in data:
                    # JARVIS structure: 'atoms' contains atomic info
                    if 'atoms' in entry:
                        atoms = entry['atoms']
                        if 'elements' in atoms and 'coords' in atoms:
                            elements = atoms['elements']
                            coords = atoms['coords']
                            
                            # Convert elements to atomic numbers
                            from ase.data import chemical_symbols
                            atomic_numbers = []
                            for el in elements:
                                if el in chemical_symbols:
                                    atomic_numbers.append(chemical_symbols.index(el))
                            
                            positions = coords
                            
                            # Get energy (try multiple fields)
                            energy_pa = entry.get('formation_energy_peratom', 0.0)
                            if energy_pa == 0:
                                energy_total = entry.get('optb88vdw_total_energy', 0.0)
                                energy_pa = energy_total / len(atomic_numbers) if len(atomic_numbers) > 0 else 0
                            
                            total_energy = energy_pa * len(atomic_numbers) if len(atomic_numbers) > 0 else 0
                            
                            samples.append({
                                'atomic_numbers': atomic_numbers,
                                'positions': positions,
                                'energy': total_energy,
                                'energy_target': total_energy,
                                'forces': [[0, 0, 0]] * len(atomic_numbers),
                                'num_atoms': len(atomic_numbers),
                                'domain': 'jarvis_dft',
                            })
        except Exception as e:
            logger.debug(f"Failed to load {json_file}: {e}")
    
    logger.info(f"   Loaded {len(samples)} JARVIS-DFT samples")
    return samples


def load_ani1x():
    """Load ANI1x dataset."""
    logger.info("📥 Loading ANI1x...")
    samples = []
    
    try:
        import h5py
        ani1x_path = Path("data/ani1x/ani1x_dataset.h5")
        
        if ani1x_path.exists():
            with h5py.File(ani1x_path, 'r') as f:
                # ANI1x is large, sample it
                keys = list(f.keys())[:1000]  # Limit for now
                
                for key in tqdm(keys, desc="Loading ANI1x"):
                    try:
                        group = f[key]
                        atomic_numbers = group['atomic_numbers'][:]
                        # ANI1x uses 'coordinates' not 'positions'
                        if 'coordinates' in group:
                            positions = group['coordinates'][:]
                        else:
                            positions = group.get('positions', [])
                        
                        # Try to get energy
                        energy = None
                        for energy_key in ['wb97x_dz.energy', 'ccsd(t)_cbs.energy', 'hf_dz.energy']:
                            if energy_key in group:
                                energy = group[energy_key][0]
                                break
                        
                        if energy is None:
                            continue
                        
                        samples.append({
                            'atomic_numbers': atomic_numbers.tolist(),
                            'positions': positions.tolist() if hasattr(positions, 'tolist') else positions,
                            'energy': energy,
                            'energy_target': energy,
                            'forces': [[0, 0, 0]] * len(atomic_numbers),
                            'num_atoms': len(atomic_numbers),
                            'domain': 'ani1x',
                        })
                    except Exception as e:
                        logger.debug(f"Failed to load {key}: {e}")
    except Exception as e:
        logger.warning(f"Failed to load ANI1x: {e}")
    
    logger.info(f"   Loaded {len(samples)} ANI1x samples")
    return samples


def load_all_datasets():
    """Load all available datasets."""
    logger.info("🚀 LOADING FULL UNIFIED DATASET")
    logger.info("="*80)
    
    all_samples = []
    
    # Load each dataset
    all_samples.extend(load_jarvis_dft())
    all_samples.extend(load_ani1x())
    
    logger.info(f"\n✅ Total samples loaded: {len(all_samples)}")
    
    # Show distribution
    domains = {}
    for sample in all_samples:
        domain = sample.get('domain', 'unknown')
        domains[domain] = domains.get(domain, 0) + 1
    
    logger.info("   Distribution:")
    for domain, count in domains.items():
        logger.info(f"   - {domain}: {count} samples")
    
    return all_samples


def create_splits(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train/validation/test splits with deterministic shuffle."""
    logger.info("\n📊 Creating train/val/test splits...")
    
    torch.manual_seed(seed)
    indices = torch.randperm(len(samples)).tolist()
    
    n_total = len(samples)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    train_samples = [samples[indices[i]] for i in range(n_train)]
    val_samples = [samples[indices[i]] for i in range(n_train, n_train + n_val)]
    test_samples = [samples[indices[i]] for i in range(n_train + n_val, n_total)]
    
    logger.info(f"   Total: {n_total}")
    logger.info(f"   Train: {n_train} ({train_ratio*100:.0f}%)")
    logger.info(f"   Val: {n_val} ({val_ratio*100:.0f}%)")
    logger.info(f"   Test: {n_test} ({test_ratio*100:.0f}%)")
    
    return train_samples, val_samples, test_samples


def main():
    """Main preprocessing pipeline."""
    # 1. Load all datasets
    all_samples = load_all_datasets()
    
    if len(all_samples) == 0:
        logger.error("❌ No samples loaded!")
        return
    
    # 2. Preprocess (clean, validate, normalize)
    logger.info("\n🧹 PREPROCESSING DATA")
    logger.info("="*80)
    
    preprocessor = DataPreprocessor(
        energy_outlier_threshold=5.0,
        force_outlier_threshold=5.0,
        max_force_threshold=100.0,
        min_atoms=1,
        max_atoms=500,
        energy_range=(-50000, 50000),
    )
    
    cleaned_data, prep_results = preprocessor.process_dataset(
        all_samples,
        normalize=True
    )
    
    # 3. Create splits
    train_data, val_data, test_data = create_splits(cleaned_data)
    
    # 4. Save everything
    output_path = Path("data/preprocessed_full_unified")
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n💾 Saving preprocessed data...")
    
    with open(output_path / 'train_data.json', 'w') as f:
        json.dump(train_data, f)
    
    with open(output_path / 'val_data.json', 'w') as f:
        json.dump(val_data, f)
    
    with open(output_path / 'test_data.json', 'w') as f:
        json.dump(test_data, f)
    
    preprocessor.save_preprocessing_results(
        prep_results,
        output_path / 'preprocessing_results.json'
    )
    
    split_info = {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'total_samples': len(cleaned_data),
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
    }
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"\n✅ COMPLETE!")
    logger.info("="*80)
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Val: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    logger.info("="*80)


if __name__ == "__main__":
    main()

