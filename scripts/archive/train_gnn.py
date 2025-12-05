#!/usr/bin/env python3
"""GNN training script for DFT→GNN→QNN pipeline."""

from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
import sys
import os
import warnings

# Suppress CUDA compatibility warnings for RTX 5090
warnings.filterwarnings("ignore", message=".*CUDA capability sm_120.*")
warnings.filterwarnings("ignore", message=".*not compatible with the current PyTorch installation.*")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Only use RTX 4090 (GPU 1)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure safe multiprocessing before any data loading
from dft_hybrid.data.io import set_safe_mp
set_safe_mp()

from gnn.model import SchNetWrapper
from gnn.train import GNNTrainer, SupervisedLoss
from graphs.periodic_graph import PeriodicGraph
# from dft_hybrid.data.factory import create_dataset_factory  # Not needed for LMDB
from dft_hybrid.data.jarvis_dft import create_jarvis_dataloader


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(config: DictConfig) -> None:
    """Train GNN surrogate model with optional ensemble support.
    
    Args:
        config: Configuration from Hydra containing all training parameters
    """
    # Get training parameters from config
    output_dir = config.get('output_dir', 'models/gnn_training')
    ensemble = config.get('ensemble', None)
    seed = config.get('seed', config.pipeline.seed)
    epochs = config.get('epochs', config.gnn.num_epochs)
    learning_rate = config.get('learning_rate', config.gnn.learning_rate)
    w_e = config.get('w_e', 1.0)
    w_f = config.get('w_f', 100.0)
    w_s = config.get('w_s', 10.0)
    
    # Dataset info for JARVIS-DFT
    dataset_info = {
        'name': 'jarvis_dft',
        'root': 'data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json',
        'split': 'train',
        'batch_size': config.dataset.batch_size,
        'cutoff': config.dataset.cutoff
    }
    
    print(f"Training GNN model on dataset: {dataset_info['name']}")
    print(f"Dataset root: {dataset_info['root']}")
    print(f"Split: {dataset_info['split']}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Cutoff radius: {dataset_info['cutoff']}")
    print(f"Output directory: {output_dir}")
    print(f"Ensemble size: {ensemble if ensemble else 1}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Loss weights - E: {w_e}, F: {w_f}, S: {w_s}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device and ensure GPU usage - explicitly use RTX 4090 (GPU 2)
    if torch.cuda.is_available():
        # Check available GPUs and select RTX 4090
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        # Find RTX 4090 GPU
        target_gpu = None
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
            if "RTX 4090" in gpu_name:
                target_gpu = i
                break
        
        if target_gpu is not None:
            device = torch.device(f"cuda:{target_gpu}")
            print(f"✅ Using RTX 4090: GPU {target_gpu}")
        else:
            # Fallback to GPU 0 if RTX 4090 not found (when CUDA_VISIBLE_DEVICES is set)
            device = torch.device("cuda:0")
            print(f"⚠️ RTX 4090 not found, using GPU 0")
        
        print(f"Using GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {torch.cuda.get_device_properties(device.index).major}.{torch.cuda.get_device_properties(device.index).minor}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Load training data using JARVIS-DFT
    print("Loading training data from JARVIS-DFT...")
    try:
        # Use JARVIS-DFT dataset with real DFT targets
        jarvis_data_path = "data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json"
        batch_size = config.dataset.batch_size
        cutoff_radius = config.dataset.cutoff
        
        print(f"Using JARVIS-DFT dataset from: {jarvis_data_path}")
        print(f"Batch size: {batch_size}")
        print(f"Cutoff radius: {cutoff_radius}")
        
        # Create JARVIS-DFT dataloaders
        train_loader = create_jarvis_dataloader(
            data_path=jarvis_data_path,
            batch_size=batch_size,
            cutoff_radius=cutoff_radius,
            max_samples=50000,  # Limit for faster training
            shuffle=True,
            num_workers=0  # Disable multiprocessing for memory efficiency
        )
        
        val_loader = create_jarvis_dataloader(
            data_path=jarvis_data_path,
            batch_size=batch_size,
            cutoff_radius=cutoff_radius,
            max_samples=5000,  # Smaller validation set
            shuffle=False,
            num_workers=0
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"Error loading JARVIS-DFT dataset: {e}")
        print("Falling back to dummy data for demonstration...")
        train_loader = val_loader = None
    
    if ensemble and ensemble > 1:
        # Ensemble training
        print(f"Training {ensemble} ensemble members...")
        
        for i in range(ensemble):
            print(f"\nTraining ensemble member {i+1}/{ensemble}")
            
            # Set different seed for each ensemble member
            member_seed = seed + i * 1000
            set_seed(member_seed)
            
            # Create model
            model = SchNetWrapper(
                hidden_channels=128,
                num_filters=128,
                num_interactions=6,
                num_gaussians=50,
                cutoff=dataset_info['cutoff'],
                max_num_neighbors=32
            ).to(device)
            
            # Create trainer
            trainer = GNNTrainer(
                model=model,
                device=device,
                w_e=w_e,
                w_f=w_f,
                w_s=w_s,
            )
            
            # Train model
            checkpoint_dir = output_dir / f"ensemble_{i+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            if train_loader is not None and val_loader is not None:
                trainer.train(train_loader, val_loader, num_epochs=epochs, 
                              save_dir=checkpoint_dir)
            else:
                print(f"Skipping training for ensemble member {i+1} (no data available)")
            
            # Save final model
            model_path = checkpoint_dir / "final_model.pt"
            torch.save(model.state_dict(), model_path)
            
            print(f"Ensemble member {i+1} saved to {model_path}")
    
    else:
        # Single model training
        print("Training single model...")
        
        # Set seed
        set_seed(seed)
        
        # Create model
        model = SchNetWrapper(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=dataset_info['cutoff'],
            max_num_neighbors=32
        ).to(device)
        
        # Create trainer
        trainer = GNNTrainer(
            model=model,
            device=device,
            w_e=w_e,
            w_f=w_f,
            w_s=w_s,
        )
        
        # Train model
        checkpoint_dir = output_dir / "single_model"
        checkpoint_dir.mkdir(exist_ok=True)
        
        if train_loader is not None and val_loader is not None:
            trainer.train(train_loader, val_loader, num_epochs=epochs,
                          save_dir=checkpoint_dir)
        else:
            print("Skipping training (no data available)")
        
        # Save final model
        model_path = checkpoint_dir / "final_model.pt"
        torch.save(model.state_dict(), model_path)
        
        print(f"Model saved to {model_path}")
    
    print("\nGNN training completed!")


if __name__ == "__main__":
    main()
