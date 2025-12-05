#!/usr/bin/env python3
"""Enhanced GNN training script with ensemble support and uncertainty quantification."""

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
import argparse

# Suppress CUDA compatibility warnings for RTX 5090
warnings.filterwarnings("ignore", message=".*CUDA capability sm_120.*")
warnings.filterwarnings("ignore", message=".*not compatible with the current PyTorch installation.*")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only use RTX 4090 (GPU 0)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure safe multiprocessing before any data loading
from dft_hybrid.data.io import set_safe_mp
set_safe_mp()

from gnn.model import SchNetWrapper
from gnn.train import GNNTrainer, SupervisedLoss
from gnn.uncertainty import EnsembleTrainer, EnsembleUncertainty
from graphs.periodic_graph import PeriodicGraph
from dft_hybrid.data.jarvis_dft import create_jarvis_dataloader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_optimized_optimizer(model: torch.nn.Module, lr: float = 1e-4) -> AdamW:
    """Create an optimized AdamW optimizer."""
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,      # L2 regularization
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
            step_size=10,        # Decay every 10 epochs
            gamma=0.5            # Halve LR
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_single_model(config: DictConfig, train_loader, val_loader, 
                      device: torch.device, output_dir: Path) -> str:
    """Train a single GNN model."""
    print("\n🧠 Creating optimized model...")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create model
    model = SchNetWrapper(
        hidden_channels=256,
        num_filters=256,
        num_interactions=8,
        num_gaussians=64,
        cutoff=config.dataset.cutoff,
        max_num_neighbors=64
    ).to(device)
    
    print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
    
    # Create optimizer and scheduler
    optimizer = create_optimized_optimizer(model, config.gnn.learning_rate)
    scheduler = create_optimized_scheduler(optimizer, config.get('scheduler_type', 'plateau'))
    
    # Create trainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        w_e=config.get('w_e', 1.0),
        w_f=config.get('w_f', 100.0),
        w_s=config.get('w_s', 10.0),
        use_gradient_clipping=True,
        max_grad_norm=config.get('max_grad_norm', 1.0),
        use_early_stopping=True,
        patience=config.get('early_stopping_patience', 15)
    )
    
    # Train model
    checkpoint_dir = output_dir / "single_model"
    checkpoint_dir.mkdir(exist_ok=True)
    
    trainer.train(train_loader, val_loader, 
                 optimizer=optimizer, scheduler=scheduler,
                 num_epochs=config.gnn.num_epochs, save_dir=checkpoint_dir)
    
    # Save final model
    model_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"\nModel saved to: {model_path}")
    return str(model_path)


def train_ensemble(config: DictConfig, train_loader, val_loader, 
                   device: torch.device, output_dir: Path) -> str:
    """Train ensemble of GNN models."""
    print(f"\n🎯 Training ensemble of {config.ensemble} models...")
    
    # Create ensemble trainer
    ensemble_trainer = EnsembleTrainer(
        model_class=SchNetWrapper,
        model_config={
            'hidden_channels': 256,
            'num_filters': 256,
            'num_interactions': 8,
            'num_gaussians': 64,
            'cutoff': config.dataset.cutoff,
            'max_num_neighbors': 64
        },
        device=device
    )
    
    # Train ensemble
    ensemble_dir = ensemble_trainer.train_ensemble(
        train_loader=train_loader,
        val_loader=val_loader,
        ensemble_size=config.ensemble,
        num_epochs=config.gnn.num_epochs,
        learning_rate=config.gnn.learning_rate,
        save_dir=str(output_dir / "ensemble")
    )
    
    return ensemble_dir


@hydra.main(version_base=None, config_path="../src/config", config_name="config")
def main(config: DictConfig) -> None:
    """Train GNN surrogate model with optional ensemble support.
    
    Args:
        config: Configuration from Hydra containing all training parameters
    """
    print("🚀 Starting Enhanced GNN Training")
    print("==================================================")
    
    # Get training parameters from config
    output_dir = config.get('output_dir', 'models/gnn_training_enhanced')
    ensemble = config.get('ensemble', None)
    seed = config.get('seed', config.pipeline.seed)
    epochs = config.get('epochs', config.gnn.num_epochs)
    learning_rate = config.get('learning_rate', config.gnn.learning_rate)
    w_e = config.get('w_e', 1.0)
    w_f = config.get('w_f', 100.0)
    w_s = config.get('w_s', 10.0)
    
    # Optimization specific parameters
    scheduler_type = config.get('scheduler_type', 'plateau')
    early_stopping_patience = config.get('early_stopping_patience', 15)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # Dataset info for JARVIS-DFT
    dataset_info = {
        'name': 'jarvis_dft',
        'root': 'data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json',
        'split': 'train',
        'batch_size': config.dataset.batch_size,
        'cutoff': config.dataset.cutoff
    }
    
    print(f"Dataset: {dataset_info['name']}")
    print(f"Batch size: {dataset_info['batch_size']}")
    print(f"Cutoff radius: {dataset_info['cutoff']}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Early stopping: True (patience: {early_stopping_patience})")
    print(f"Gradient clipping: True (max_norm: {max_grad_norm})")
    print(f"Loss weights - E: {w_e}, F: {w_f}, S: {w_s}")
    if ensemble:
        print(f"Ensemble size: {ensemble}")
    print("==================================================")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device and ensure GPU usage - explicitly use RTX 4090 (GPU 0)
    if torch.cuda.is_available():
        # Check available GPUs and select RTX 4090
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
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
        print(f"Using device: {device}")
        # Clear GPU cache
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
        
        print(f"Loading JARVIS-DFT data from {jarvis_data_path}")
        
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
        print(f"Error loading JARVIS-DFT dataset: {e}")
        print("Falling back to dummy data for demonstration...")
        train_loader = val_loader = None
    
    if train_loader is None or val_loader is None:
        print("❌ No data available for training")
        return
    
    if ensemble and ensemble > 1:
        # Ensemble training
        ensemble_dir = train_ensemble(config, train_loader, val_loader, device, output_dir)
        print(f"\n🎉 Ensemble training completed!")
        print(f"📁 Ensemble saved to: {ensemble_dir}")
        
        # Quick ensemble evaluation
        print(f"\n🔍 Quick ensemble evaluation...")
        ensemble_uncertainty = EnsembleUncertainty(device=device)
        ensemble_uncertainty.load_ensemble(ensemble_dir)
        
        # Create small test set for quick evaluation
        test_loader = create_jarvis_dataloader(
            data_path=jarvis_data_path,
            batch_size=batch_size,
            cutoff_radius=cutoff_radius,
            max_samples=1000,
            shuffle=False,
            num_workers=0
        )
        
        metrics = ensemble_uncertainty.evaluate_uncertainty(test_loader)
        
        print("\n📈 Quick Ensemble Results")
        print("=" * 50)
        print(f"MAE(E_total):           {metrics['mae_mean']:.6f} eV")
        print(f"MAE(E_per_atom):        {metrics['mae_per_atom']:.6f} eV/atom")
        print(f"Mean Uncertainty:       {metrics['mean_uncertainty']:.6f} eV")
        print(f"Mean Uncertainty/atom:  {metrics['mean_uncertainty_per_atom']:.6f} eV/atom")
        print(f"Uncertainty Correlation: {metrics['uncertainty_correlation']:.4f}")
        print(f"Ensemble Size:          {metrics['ensemble_size']}")
        print("=" * 50)
        
    else:
        # Single model training
        model_path = train_single_model(config, train_loader, val_loader, device, output_dir)
        print(f"\n🎉 Single model training completed!")
        print(f"📁 Model saved to: {model_path}")
    
    print("\n🎉 ENHANCED TRAINING COMPLETED!")
    print("==================================================")


if __name__ == "__main__":
    main()


