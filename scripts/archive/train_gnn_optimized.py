#!/usr/bin/env python3
"""Optimized GNN training script with advanced techniques to minimize validation loss."""

from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
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
from dft_hybrid.data.jarvis_dft import create_jarvis_dataloader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_optimized_model(cutoff: float, device: torch.device) -> SchNetWrapper:
    """Create an optimized SchNet model with better architecture."""
    return SchNetWrapper(
        hidden_channels=256,      # Increased from 128
        num_filters=256,         # Increased from 128
        num_interactions=8,      # Increased from 6
        num_gaussians=64,        # Increased from 50
        cutoff=cutoff,
        max_num_neighbors=64     # Increased from 32
    ).to(device)


def create_optimized_optimizer(model: nn.Module, learning_rate: float = 1e-4) -> AdamW:
    """Create an optimized AdamW optimizer with better parameters."""
    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,       # L2 regularization
        betas=(0.9, 0.999),     # Standard Adam betas
        eps=1e-8
    )


def create_optimized_scheduler(optimizer: torch.optim.Optimizer, 
                              scheduler_type: str = "plateau") -> object:
    """Create an optimized learning rate scheduler."""
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8,          # Increased patience
            min_lr=1e-7          # Minimum learning rate
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=100,           # Maximum epochs
            eta_min=1e-7         # Minimum learning rate
        )
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=20,        # Reduce LR every 20 epochs
            gamma=0.5            # Reduce by half
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(config: DictConfig) -> None:
    """Train optimized GNN surrogate model with advanced techniques."""
    # Get training parameters from config
    output_dir = config.get('output_dir', 'models/gnn_training_optimized')
    seed = config.get('seed', config.pipeline.seed)
    epochs = config.get('epochs', config.gnn.num_epochs)
    learning_rate = config.get('learning_rate', config.gnn.learning_rate)
    w_e = config.get('w_e', 1.0)
    w_f = config.get('w_f', 100.0)
    w_s = config.get('w_s', 10.0)
    
    # Advanced training parameters
    scheduler_type = config.get('scheduler_type', 'plateau')
    use_early_stopping = config.get('use_early_stopping', True)
    early_stopping_patience = config.get('early_stopping_patience', 15)
    use_gradient_clipping = config.get('use_gradient_clipping', True)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # Dataset info for JARVIS-DFT
    dataset_info = {
        'name': 'jarvis_dft',
        'root': 'data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json',
        'split': 'train',
        'batch_size': config.dataset.batch_size,
        'cutoff': config.dataset.cutoff
    }
    
    print("🚀 Starting OPTIMIZED GNN Training")
    print("=" * 50)
    print(f"Dataset: {dataset_info['name']}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Cutoff radius: {dataset_info['cutoff']}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Early stopping: {use_early_stopping} (patience: {early_stopping_patience})")
    print(f"Gradient clipping: {use_gradient_clipping} (max_norm: {max_grad_norm})")
    print(f"Loss weights - E: {w_e}, F: {w_f}, S: {w_s}")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device and ensure GPU usage
    if torch.cuda.is_available():
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
            device = torch.device("cuda:0")
            print(f"⚠️ RTX 4090 not found, using GPU 0")
        
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Load training data using JARVIS-DFT
    print("\n📊 Loading JARVIS-DFT dataset...")
    try:
        jarvis_data_path = "data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json"
        batch_size = config.dataset.batch_size
        cutoff_radius = config.dataset.cutoff
        
        # Create optimized dataloaders with larger datasets
        train_loader = create_jarvis_dataloader(
            data_path=jarvis_data_path,
            batch_size=batch_size,
            cutoff_radius=cutoff_radius,
            max_samples=100000,  # Increased from 50000
            shuffle=True,
            num_workers=0
        )
        
        val_loader = create_jarvis_dataloader(
            data_path=jarvis_data_path,
            batch_size=batch_size,
            cutoff_radius=cutoff_radius,
            max_samples=10000,   # Increased from 5000
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"❌ Error loading JARVIS-DFT dataset: {e}")
        return
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create optimized model
    print("\n🧠 Creating optimized model...")
    model = create_optimized_model(cutoff_radius, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create optimized optimizer
    print("\n⚙️ Creating optimized optimizer...")
    optimizer = create_optimized_optimizer(model, learning_rate)
    
    # Create optimized scheduler
    print(f"\n📉 Creating {scheduler_type} scheduler...")
    scheduler = create_optimized_scheduler(optimizer, scheduler_type)
    
    # Create trainer with advanced features
    print("\n🏋️ Creating optimized trainer...")
    trainer = GNNTrainer(
        model=model,
        device=device,
        w_e=w_e,
        w_f=w_f,
        w_s=w_s,
        use_gradient_clipping=use_gradient_clipping,
        max_grad_norm=max_grad_norm,
        use_early_stopping=use_early_stopping,
        patience=early_stopping_patience
    )
    
    # Train model
    print("\n🎯 Starting optimized training...")
    checkpoint_dir = output_dir / "optimized_model"
    checkpoint_dir.mkdir(exist_ok=True)
    
    metrics_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs,
        save_dir=checkpoint_dir,
        save_every=5  # Save more frequently
    )
    
    # Save final model
    model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Print final results
    print("\n🎉 OPTIMIZED TRAINING COMPLETED!")
    print("=" * 50)
    if metrics_history:
        final_metrics = metrics_history[-1]
        print(f"Final Training Loss: {final_metrics['train_loss']:.6f}")
        print(f"Final Validation Loss: {final_metrics['val_loss']:.6f}")
        print(f"Best Validation Loss: {trainer.best_val_loss:.6f}")
        print(f"Improvement: {((13.001648 - trainer.best_val_loss) / 13.001648 * 100):.2f}%")
    print(f"Model saved to: {model_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
