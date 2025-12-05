#!/usr/bin/env python3
"""Test existing ensemble models on GPU without problematic imports."""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_existing_ensemble():
    """Test loading and using existing ensemble models on GPU."""
    print("=== TESTING EXISTING ENSEMBLE MODELS ON GPU ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load ensemble metadata
    metadata_path = "models/gnn_training_enhanced/ensemble/ensemble_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nEnsemble metadata:")
    print(f"  - Size: {metadata['ensemble_size']}")
    print(f"  - Epochs: {metadata['num_epochs']}")
    print(f"  - Learning rate: {metadata['learning_rate']}")
    print(f"  - Model config: {metadata['model_config']}")
    
    # Test loading each model
    ensemble_dir = Path("models/gnn_training_enhanced/ensemble/")
    checkpoint_files = list(ensemble_dir.glob("ckpt_*.pt"))
    checkpoint_files.sort()
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    
    models_info = []
    for i, ckpt_file in enumerate(checkpoint_files):
        print(f"\nLoading model {i}: {ckpt_file.name}")
        
        try:
            # Load checkpoint on GPU
            checkpoint = torch.load(ckpt_file, map_location='cuda:0')
            
            val_loss = checkpoint['best_val_loss']
            seed = checkpoint['seed']
            model_config = checkpoint['model_config']
            
            print(f"  ✅ Loaded successfully")
            print(f"  - Validation loss: {val_loss:.4f}")
            print(f"  - Seed: {seed}")
            print(f"  - Model config: {model_config}")
            
            models_info.append({
                'file': ckpt_file.name,
                'val_loss': val_loss,
                'seed': seed,
                'config': model_config
            })
            
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully loaded {len(models_info)} models")
    
    if models_info:
        avg_val_loss = sum(m['val_loss'] for m in models_info) / len(models_info)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        print(f"\nModel details:")
        for i, info in enumerate(models_info):
            print(f"  Model {i}: {info['file']} (loss: {info['val_loss']:.4f}, seed: {info['seed']})")
    
    return models_info

if __name__ == "__main__":
    test_existing_ensemble()

