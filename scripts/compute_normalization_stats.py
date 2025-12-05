#!/usr/bin/env python3
"""
Compute and save normalization statistics for training data.
This will help diagnose energy scale issues.
"""

import sys
from pathlib import Path
import torch
import json
import logging
from omegaconf import DictConfig
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_training_data_stats():
    """Compute statistics on training data to understand energy scale."""
    
    logger.info("📊 COMPUTING TRAINING DATA STATISTICS")
    logger.info("="*80)
    
    # Try to load unified dataset
    try:
        from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, UnifiedDatasetConfig
        
        # Load dataset config
        config = UnifiedDatasetConfig()
        registry = UnifiedDatasetRegistry(config)
        
        # Get training split
        train_dataset = registry.get_training_split()
        
        logger.info(f"✅ Loaded training dataset: {len(train_dataset)} samples")
        
        # Collect all energies
        energies = []
        per_atom_energies = []
        
        logger.info("🔬 Computing statistics...")
        for i, data in enumerate(tqdm(train_dataset, desc="Processing")):
            if hasattr(data, 'energy'):
                if isinstance(data.energy, torch.Tensor):
                    energy = data.energy.item()
                else:
                    energy = data.energy
                energies.append(energy)
                
                # Compute per-atom energy if we have num_atoms
                if hasattr(data, 'num_atoms'):
                    per_atom = energy / data.num_atoms
                    per_atom_energies.append(per_atom)
                    
        if len(energies) > 0:
            energies = np.array(energies)
            per_atom = np.array(per_atom_energies) if len(per_atom_energies) > 0 else None
            
            stats = {
                'total_energies': {
                    'count': len(energies),
                    'mean': float(np.mean(energies)),
                    'std': float(np.std(energies)),
                    'min': float(np.min(energies)),
                    'max': float(np.max(energies)),
                    'median': float(np.median(energies)),
                }
            }
            
            if per_atom is not None:
                stats['per_atom_energies'] = {
                    'count': len(per_atom),
                    'mean': float(np.mean(per_atom)),
                    'std': float(np.std(per_atom)),
                    'min': float(np.min(per_atom)),
                    'max': float(np.max(per_atom)),
                    'median': float(np.median(per_atom)),
                }
            
            logger.info("\n📊 ENERGY STATISTICS:")
            logger.info("="*80)
            logger.info(f"Total Energies (eV):")
            logger.info(f"   Count: {stats['total_energies']['count']}")
            logger.info(f"   Mean: {stats['total_energies']['mean']:.4f}")
            logger.info(f"   Std: {stats['total_energies']['std']:.4f}")
            logger.info(f"   Min: {stats['total_energies']['min']:.4f}")
            logger.info(f"   Max: {stats['total_energies']['max']:.4f}")
            logger.info(f"   Median: {stats['total_energies']['median']:.4f}")
            
            if per_atom is not None:
                logger.info(f"\nPer-Atom Energies (eV/atom):")
                logger.info(f"   Count: {stats['per_atom_energies']['count']}")
                logger.info(f"   Mean: {stats['per_atom_energies']['mean']:.4f}")
                logger.info(f"   Std: {stats['per_atom_energies']['std']:.4f}")
                logger.info(f"   Min: {stats['per_atom_energies']['min']:.4f}")
                logger.info(f"   Max: {stats['per_atom_energies']['max']:.4f}")
                logger.info(f"   Median: {stats['per_atom_energies']['median']:.4f}")
            
            # Save stats
            output_path = Path("artifacts/normalization_stats.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"\n✅ Saved statistics to {output_path}")
            
        else:
            logger.error("❌ No energies found in training data")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute statistics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compute_training_data_stats()



