#!/usr/bin/env python3
"""
Train OFFICIAL ALIGNN on Unified Diverse Dataset
Uses ALIGNN's official training infrastructure.
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def convert_unified_to_alignn_format():
    """Convert unified dataset to ALIGNN input format."""
    logger.info("🔄 Converting unified dataset to ALIGNN format...")
    
    # Load data splits
    with open('data/preprocessed_full_unified/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('data/preprocessed_full_unified/val_data.json', 'r') as f:
        val_data = json.load(f)
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Val: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    
    # ALIGNN expects data in DataFrame format with columns:
    # - id, structure, formation_energy_peratom
    # We'll create extxyz format which ALIGNN can read
    
    logger.info("\n📝 Creating extxyz files for ALIGNN...")
    
    def create_extxyz_file(data, filename):
        """Create extxyz format file for ALIGNN."""
        from ase import Atoms
        
        output = []
        
        for idx, sample in enumerate(data):
            atoms = Atoms(
                numbers=sample['atomic_numbers'],
                positions=sample['positions']
            )
            
            # Energy (total energy per atom for ALIGNN)
            energy = sample.get('energy', sample.get('energy_target', 0.0))
            energy_per_atom = energy / len(sample['atomic_numbers'])
            
            # Write extxyz format
            output.append(f"{len(sample['atomic_numbers'])}\n")
            output.append(f"structure_{idx} energy={energy_per_atom}\n")
            
            for element, pos in zip(sample['atomic_numbers'], sample['positions']):
                from ase.data import chemical_symbols
                symbol = chemical_symbols[element]
                output.append(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        with open(filename, 'w') as f:
            f.writelines(output)
        
        logger.info(f"   ✓ Created {filename} ({len(data)} samples)")
    
    # Create output directory
    alignn_data_dir = Path("data/alignn_unified")
    alignn_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create extxyz files
    create_extxyz_file(train_data, alignn_data_dir / "train.extxyz")
    create_extxyz_file(val_data, alignn_data_dir / "val.extxyz")
    create_extxyz_file(test_data, alignn_data_dir / "test.extxyz")
    
    logger.info("\n✅ Data conversion complete!")
    logger.info(f"   Output: {alignn_data_dir}")
    
    return alignn_data_dir


def train_alignn_official():
    """Train ALIGNN using official training script."""
    logger.info("\n🚀 TRAINING ALIGNN (OFFICIAL)")
    logger.info("="*80)
    
    # Note: ALIGNN has compatibility issues with current PyTorch
    # We'll use the alignn cli command if available
    
    try:
        import subprocess
        
        # Try to run ALIGNN training
        logger.info("   Attempting ALIGNN official training...")
        
        # ALIGNN training command
        cmd = [
            sys.executable, "-m", "alignn.train",
            "--root_dir", "data/alignn_unified",
            "--config", "data/alignn_unified/config.json",
            "--output_dir", "models/alignn_official"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("   ✅ ALIGNN training completed!")
        else:
            logger.error(f"   ❌ ALIGNN training failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ ALIGNN training error: {e}")
        
        # Fallback: Create training config and instructions
        logger.info("\n💡 FALLBACK: Creating ALIGNN training config...")
        
        create_alignn_config()
        logger.info("\n✅ Training config created!")
        logger.info("   Run: alignn-scripts/train.py with proper setup")
        
        return False


def create_alignn_config():
    """Create ALIGNN training configuration."""
    config = {
        "train_folder": "data/alignn_unified/train.extxyz",
        "val_folder": "data/alignn_unified/val.extxyz",
        "test_folder": "data/alignn_unified/test.extxyz",
        "model": {
            "name": "alignn",
            "output_features": 1,
            "classification": False
        },
        "optim": {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 1e-4
        }
    }
    
    with open('data/alignn_unified/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("   Created: data/alignn_unified/config.json")


def main():
    """Main training function."""
    # Convert data
    data_dir = convert_unified_to_alignn_format()
    
    # Try official training
    success = train_alignn_official()
    
    if not success:
        logger.info("\n⚠️  NOTE: ALIGNN requires specific environment setup")
        logger.info("   To train ALIGNN properly:")
        logger.info("   1. Setup compatible PyTorch version")
        logger.info("   2. Install DGL properly")
        logger.info("   3. Use ALIGNN's official training script")
        logger.info("\n   Data is ready at: data/alignn_unified/")
        logger.info("   You can train ALIGNN manually or in separate environment")


if __name__ == "__main__":
    main()


