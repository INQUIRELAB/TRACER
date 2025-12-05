#!/usr/bin/env python3
"""Training script using unified dataset registry with temperature-based sampling."""

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
import logging

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
from dft_hybrid.data.unified_registry import (
    UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig, DatasetDomain,
    create_unified_dataloader, UnitConverter
)
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


def create_unified_config_from_hydra(config: DictConfig) -> UnifiedDatasetConfig:
    """Create unified dataset configuration from Hydra config."""
    datasets = []
    
    # Convert Hydra config to unified config
    for dataset_name, dataset_config in config.datasets.items():
        if not dataset_config.get('enabled', True):
            continue
            
        domain_id = DatasetDomain(dataset_config.domain_id)
        
        unified_dataset = DatasetConfig(
            domain_id=domain_id,
            name=dataset_config.name,
            path=dataset_config.path,
            energy_unit=dataset_config.get('energy_unit', 'eV'),
            force_unit=dataset_config.get('force_unit', 'eV/Å'),
            stress_unit=dataset_config.get('stress_unit', 'eV/Å³'),
            temperature_range=tuple(dataset_config.get('temperature_range', [0, 1000])),
            atomic_species=dataset_config.get('atomic_species', []),
            max_samples=dataset_config.get('max_samples', None),
            weight=dataset_config.get('weight', 1.0),
            enabled=dataset_config.get('enabled', True)
        )
        
        datasets.append(unified_dataset)
    
    return UnifiedDatasetConfig(
        datasets=datasets,
        mix_strategy=config.dataset.mix_strategy,
        temperature_tau=config.dataset.temperature_tau,
        unit_conversion=config.dataset.unit_conversion,
        validation_split=config.dataset.validation_split,
        test_split=config.dataset.test_split,
        random_seed=config.dataset.random_seed,
        normalize_energies=True  # Enable energy normalization by default
    )


@hydra.main(version_base=None, config_path="../src/config", config_name="unified_dataset")
def main(config: DictConfig) -> None:
    """Train GNN surrogate model using unified dataset registry.
    
    Args:
        config: Configuration from Hydra containing all training parameters
    """
    print("🚀 Starting Unified Dataset Registry Training")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("UnifiedTraining")
    
    # Get training parameters from config
    output_dir = config.get('output_dir', 'models/unified_training')
    ensemble = config.get('ensemble', None)
    seed = config.get('seed', config.dataset.random_seed)
    epochs = config.get('epochs', config.training.epochs)
    learning_rate = config.get('learning_rate', config.training.learning_rate)
    w_e = config.get('w_e', config.training.w_e)
    w_f = config.get('w_f', config.training.w_f)
    w_s = config.get('w_s', config.training.w_s)
    
    # Optimization specific parameters
    scheduler_type = config.get('scheduler_type', config.training.scheduler_type)
    early_stopping_patience = config.get('early_stopping_patience', config.training.early_stopping_patience)
    max_grad_norm = config.get('max_grad_norm', config.training.max_grad_norm)
    
    print(f"Mix strategy: {config.dataset.mix_strategy}")
    print(f"Temperature tau: {config.dataset.temperature_tau}")
    print(f"Unit conversion: {config.dataset.unit_conversion}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Early stopping: True (patience: {early_stopping_patience})")
    print(f"Gradient clipping: True (max_norm: {max_grad_norm})")
    print(f"Loss weights - E: {w_e}, F: {w_f}, S: {w_s}")
    if ensemble:
        print(f"Ensemble size: {ensemble}")
    print("=" * 60)
    
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
    
    # Create unified dataset configuration
    print("\n📊 Creating unified dataset configuration...")
    unified_config = create_unified_config_from_hydra(config)
    
    # Print dataset information
    print(f"Enabled datasets: {len([d for d in unified_config.datasets if d.enabled])}")
    for dataset in unified_config.datasets:
        if dataset.enabled:
            print(f"  - {dataset.name} ({dataset.domain_id.value}): {dataset.max_samples or 'all'} samples, weight={dataset.weight}")
    
    # Create unified dataloaders
    print(f"\n🔄 Creating unified dataloaders...")
    try:
        train_loader = create_unified_dataloader(
            config=unified_config,
            batch_size=config.training.batch_size,
            max_samples=None,  # Use all available samples
            shuffle=True,
            num_workers=0
        )
        
        val_loader = create_unified_dataloader(
            config=unified_config,
            batch_size=config.training.batch_size,
            max_samples=None,   # Use all available samples
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Training batches: {len(train_loader)}")
        print(f"✅ Validation batches: {len(val_loader)}")
        
    except Exception as e:
        logger.error(f"Error creating unified dataloaders: {e}")
        print("❌ Failed to create dataloaders")
        return
    
    if ensemble and ensemble > 1:
        # Ensemble training
        print(f"\n🎯 Training ensemble of {ensemble} models...")
        
        ensemble_trainer = EnsembleTrainer(
            model_class=SchNetWrapper,
            model_config={
                'hidden_channels': config.model.hidden_channels,
                'num_filters': config.model.num_filters,
                'num_interactions': config.model.num_interactions,
                'num_gaussians': config.model.num_gaussians,
                'cutoff': config.model.cutoff,
                'max_num_neighbors': config.model.max_num_neighbors
            },
            device=device
        )
        
        # Train ensemble
        ensemble_dir = ensemble_trainer.train_ensemble(
            train_loader=train_loader,
            val_loader=val_loader,
            ensemble_size=ensemble,
            num_epochs=epochs,
            learning_rate=learning_rate,
            save_dir=str(output_dir / "ensemble")
        )
        
        print(f"\n🎉 Ensemble training completed!")
        print(f"📁 Ensemble saved to: {ensemble_dir}")
        
    else:
        # Single model training
        print("\n🧠 Creating optimized model...")
        
        # Set seed
        set_seed(seed)
        
        # Create model
        model = SchNetWrapper(
            hidden_channels=config.model.hidden_channels,
            num_filters=config.model.num_filters,
            num_interactions=config.model.num_interactions,
            num_gaussians=config.model.num_gaussians,
            cutoff=config.model.cutoff,
            max_num_neighbors=config.model.max_num_neighbors
        ).to(device)
        
        print(f"📈 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
        
        # Create optimizer and scheduler
        optimizer = create_optimized_optimizer(model, learning_rate)
        scheduler = create_optimized_scheduler(optimizer, scheduler_type)
        
        # Create trainer
        trainer = GNNTrainer(
            model=model,
            device=device,
            w_e=w_e,
            w_f=w_f,
            w_s=w_s,
            use_gradient_clipping=True,
            max_grad_norm=max_grad_norm,
            use_early_stopping=True,
            patience=early_stopping_patience
        )
        
        # Train model
        checkpoint_dir = output_dir / "unified_model"
        checkpoint_dir.mkdir(exist_ok=True)
        
        trainer.train(train_loader, val_loader, 
                     optimizer=optimizer, scheduler=scheduler,
                     num_epochs=epochs, save_dir=checkpoint_dir)
        
        # Save final model
        model_path = checkpoint_dir / "final_model.pt"
        torch.save(model.state_dict(), model_path)
        
        print(f"\nModel saved to: {model_path}")
    
    print("\n🎉 UNIFIED TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
