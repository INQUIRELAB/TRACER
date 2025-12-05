#!/usr/bin/env python3
"""
Setup ALIGNN for unified dataset comparison.
Downloads, adapts, and prepares ALIGNN for training on unified diverse dataset.
"""

import sys
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_alignn_installation():
    """Check if ALIGNN is installed."""
    try:
        import alignn
        logger.info("✅ ALIGNN is already installed")
        return True
    except ImportError:
        logger.warning("❌ ALIGNN not installed")
        logger.info("   Installing ALIGNN...")
        return False


def install_alignn():
    """Install ALIGNN via pip."""
    import subprocess
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "alignn",
            "--no-cache-dir"
        ])
        logger.info("✅ ALIGNN installed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("❌ Failed to install ALIGNN")
        return False


def prepare_unified_dataset_for_alignn():
    """Convert unified dataset to ALIGNN format."""
    logger.info("\n🔄 Preparing unified dataset for ALIGNN...")
    
    # Load preprocessed data
    with open('data/preprocessed_full_unified/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('data/preprocessed_full_unified/val_data.json', 'r') as f:
        val_data = json.load(f)
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Val: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    
    # ALIGNN expects ASE Atoms objects
    from ase import Atoms
    
    # Create ALIGNN-compatible dataset
    alignn_dataset = []
    
    for sample in train_data:
        atoms = Atoms(
            numbers=sample['atomic_numbers'],
            positions=sample['positions'],
        )
        
        alignn_sample = {
            'atoms': atoms,
            'target': sample.get('energy_target', sample.get('energy', 0.0)),
            'properties': {
                'formation_energy_peratom': sample.get('energy_target', 0.0) / len(sample['atomic_numbers'])
            }
        }
        alignn_dataset.append(alignn_sample)
    
    logger.info(f"   Converted {len(alignn_dataset)} samples to ALIGNN format")
    
    return train_data, val_data, test_data


def create_alignn_config():
    """Create ALIGNN training configuration."""
    config = {
        "model": {
            "name": "alignn",
            "alignn_layers": 4,
            "gcn_layers": 2,
            "atom_input_features": 92,
            "classification": False,
            "regression": True,
            "output_features": 1
        },
        "optim": {
            "batch_size": 16,
            "epochs": 50,
            "learning_rate": 1e-4,
            "lr_scheduler": "ReduceLROnPlateau",
            "weight_decay": 1e-5
        },
        "dataset": {
            "name": "unified_diverse",
            "train": "data/preprocessed_full_unified/train_data.json",
            "val": "data/preprocessed_full_unified/val_data.json",
            "test": "data/preprocessed_full_unified/test_data.json"
        }
    }
    
    config_path = Path("config/alignn_comparison_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Created ALIGNN config: {config_path}")
    return config


def main():
    """Setup ALIGNN comparison."""
    logger.info("🚀 SETTING UP ALIGNN COMPARISON")
    logger.info("="*80)
    
    # 1. Check/install ALIGNN
    if not check_alignn_installation():
        if not install_alignn():
            logger.error("Cannot proceed without ALIGNN")
            return
    
    # 2. Prepare dataset
    train_data, val_data, test_data = prepare_unified_dataset_for_alignn()
    
    # 3. Create config
    config = create_alignn_config()
    
    logger.info("\n✅ ALIGNN SETUP COMPLETE!")
    logger.info("="*80)
    logger.info("\n📝 Next steps:")
    logger.info("   1. Review config: config/alignn_comparison_config.json")
    logger.info("   2. Run training: scripts/train_alignn_comparison.py")
    logger.info("   3. Compare results: scripts/compare_with_alignn.py")
    logger.info("="*80)


if __name__ == "__main__":
    main()



