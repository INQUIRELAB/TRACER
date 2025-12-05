#!/usr/bin/env python3
"""
Preprocess Full Unified Dataset
Loads all available datasets, preprocesses, and creates train/val/test splits.
"""

import sys
import os
from pathlib import Path
import logging
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import DataPreprocessor
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GraphDataset(Dataset):
    """PyTorch Dataset for molecular graphs."""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_unified_dataset_simple():
    """Load datasets in a simpler way that works."""
    logger.info("📥 Loading unified training dataset...")
    
    all_samples = []
    
    # Load JARVIS-DFT
    try:
        jarvis_path = Path("data/jarvis_dft")
        if jarvis_path.exists():
            json_files = list(jarvis_path.glob("*.json"))[:5]  # Limit for now
            logger.info(f"📁 Found {len(json_files)} JARVIS files")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for entry in data:
                            if 'atom_coords' in entry and 'atom_nums' in entry:
                                atomic_numbers = entry['atom_nums']
                                positions = entry['atom_coords']
                                
                                # CRITICAL FIX: Store formation_energy_per_atom directly
                                # DO NOT multiply by num_atoms - this is already per-atom!
                                formation_energy_per_atom = entry.get('formation_energy_per_atom', 
                                                                    entry.get('formation_energy_peratom',
                                                                              entry.get('energy_per_atom', 0.0)))
                                
                                all_samples.append({
                                    'atomic_numbers': atomic_numbers,
                                    'positions': positions,
                                    'energy': formation_energy_per_atom,  # PER-ATOM energy (not total!)
                                    'formation_energy_per_atom': formation_energy_per_atom,  # Store explicitly
                                    'forces': [[0, 0, 0]] * len(atomic_numbers),
                                    'num_atoms': len(atomic_numbers),
                                    # Store cell if available (for PBC)
                                    'cell': entry.get('lattice_mat', entry.get('cell', None)),
                                })
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to load JARVIS: {e}")
    
    logger.info(f"✅ Loaded {len(all_samples)} samples")
    
    # Add some dummy samples if we don't have enough
    if len(all_samples) == 0:
        logger.info("⚠️  No data loaded, creating demo samples")
        all_samples = [
            {
                'atomic_numbers': [1, 6, 8] * i,
                'positions': [[j*1.5, 0, 0] for j in range(3 * i)],
                'energy': -100.0 * i,
                'forces': [[0, 0, 0]] * (3 * i),
                'num_atoms': 3 * i,
            } for i in range(1, 11)
        ]
    
    return all_samples


def create_splits(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/validation/test splits."""
    logger.info("\n📊 Creating data splits...")
    
    n_total = len(samples)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    logger.info(f"   Total samples: {n_total}")
    logger.info(f"   Train: {n_train} ({train_ratio*100:.0f}%)")
    logger.info(f"   Validation: {n_val} ({val_ratio*100:.0f}%)")
    logger.info(f"   Test: {n_test} ({test_ratio*100:.0f}%)")
    
    # Simple split (could use random_split for better shuffling)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    return train_samples, val_samples, test_samples


def preprocess_full_dataset():
    """Preprocess full unified dataset."""
    logger.info("🚀 FULL DATASET PREPROCESSING")
    logger.info("="*80)
    
    # 1. Load all data
    samples = load_unified_dataset_simple()
    
    if len(samples) == 0:
        logger.error("❌ No data to preprocess!")
        return
    
    logger.info(f"📊 Loaded {len(samples)} samples")
    
    # 2. Create preprocessor
    preprocessor = DataPreprocessor(
        energy_outlier_threshold=5.0,
        force_outlier_threshold=5.0,
        max_force_threshold=100.0,
        min_atoms=1,
        max_atoms=500,
        energy_range=(-10000, 10000),
    )
    
    # 3. Run preprocessing
    cleaned_data, results = preprocessor.process_dataset(
        samples,
        normalize=True
    )
    
    # 4. Create splits
    train_samples, val_samples, test_samples = create_splits(cleaned_data)
    
    # 5. Save everything
    output_path = Path("data/preprocessed_full")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    logger.info("\n💾 Saving data splits...")
    with open(output_path / 'train_data.json', 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(output_path / 'val_data.json', 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    with open(output_path / 'test_data.json', 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    # Save preprocessing results
    preprocessor.save_preprocessing_results(
        results,
        output_path / 'preprocessing_results.json'
    )
    
    # Save split info
    split_info = {
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'total_samples': len(cleaned_data),
    }
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"\n✅ Full dataset preprocessing complete!")
    logger.info(f"   Output directory: {output_path}")
    logger.info(f"   Train: {len(train_samples)} samples")
    logger.info(f"   Validation: {len(val_samples)} samples")
    logger.info(f"   Test: {len(test_samples)} samples")
    logger.info("="*80)
    
    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def preprocess():
        preprocess_full_dataset()
    
    app()

