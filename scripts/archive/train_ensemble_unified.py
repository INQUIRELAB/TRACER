#!/usr/bin/env python3
"""
Train N=5 ensemble models on the complete unified dataset.
Ensures all 5 domains (JARVIS-DFT, JARVIS-Elastic, OC20-S2EF, OC22-S2EF, ANI1x) are loaded.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from gnn.domain_aware_model import DomainAwareSchNet, DomainAdapterConfig
from gnn.multi_stage_trainer import MultiStageTrainer, DomainSpecificLoss
from gnn.multi_task_trainer import MultiTaskConfig
from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, create_unified_config_from_hydra


def setup_logging(log_dir: Path, ensemble_id: int):
    """Setup logging for ensemble training."""
    log_file = log_dir / f"ensemble_{ensemble_id}.log"
    
    # Create logger
    logger = logging.getLogger(f"ensemble_{ensemble_id}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_unified_dataloader(config, split: str, batch_size: int, 
                             max_samples: int = None, shuffle: bool = True, 
                             num_workers: int = 4, logger=None):
    """Create unified dataloader for training/validation."""
    if logger:
        logger.info("Creating unified dataloader...")
    
    # Create unified config
    unified_config = create_unified_config_from_hydra(config)
    
    if logger:
        logger.info(f"Unified config created with {len(unified_config.datasets)} datasets:")
        for dataset in unified_config.datasets:
            logger.info(f"  - {dataset.name}: {dataset.domain_id.value}")
    
    # Create registry
    registry = UnifiedDatasetRegistry(unified_config)
    
    # Load datasets
    datasets = registry.load_all_datasets()
    
    if split == 'train':
        dataset = datasets['train']
    elif split == 'val':
        dataset = datasets['val']
    else:
        raise ValueError(f"Unknown split: {split}")
    
    if logger:
        logger.info(f"Loaded {len(dataset)} samples for {split} split")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=registry.collate_fn,
        pin_memory=True
    )
    
    return dataloader


def create_domain_adapter_config_from_hydra(config) -> DomainAdapterConfig:
    """Create DomainAdapterConfig from Hydra config."""
    adapter_config = config.get('domain_adapter', {})
    
    return DomainAdapterConfig(
        # Domain embedding
        domain_embedding_dim=adapter_config.get('domain_embedding_dim', 64),
        num_domains=adapter_config.get('num_domains', 5),
        
        # FiLM parameters
        film_dim=adapter_config.get('film_dim', 128),
        film_use_bias=adapter_config.get('film_use_bias', True),
        
        # LoRA parameters
        lora_rank=adapter_config.get('lora_rank', 8),
        lora_alpha=adapter_config.get('lora_alpha', 16.0),
        lora_dropout=adapter_config.get('lora_dropout', 0.1),
        
        # Fine-tuning parameters
        fine_tune_layers=adapter_config.get('fine_tune_layers', 2),
        fine_tune_lr=adapter_config.get('fine_tune_lr', 1e-5),
        freeze_backbone=adapter_config.get('freeze_backbone', True)
    )


def create_multi_task_config_from_hydra(config) -> MultiTaskConfig:
    """Create MultiTaskConfig from Hydra config."""
    return MultiTaskConfig(
        # Loss weights
        w_e=config.training.get('w_e', 1.0),
        w_f=config.training.get('w_f', 1.0),
        w_s=config.training.get('w_s', 0.0),
        
        # Missing label strategy
        missing_label_strategy=config.training.get('missing_label_strategy', 'ignore'),
        
        # Per-domain logging
        enable_per_domain_logging=config.training.get('enable_per_domain_logging', True)
    )


def train_single_ensemble_model(ensemble_id: int, config, args, save_dir: Path, log_dir: Path):
    """Train a single ensemble model."""
    logger = setup_logging(log_dir, ensemble_id)
    logger.info(f"Starting training for ensemble model {ensemble_id}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + ensemble_id)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set CUDA device
    if device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device.index)
        torch.cuda.set_device(device)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_unified_dataloader(
        config, 'train', args.batch_size, 
        max_samples=args.max_samples, shuffle=True, logger=logger
    )
    val_loader = create_unified_dataloader(
        config, 'val', args.batch_size,
        max_samples=args.max_samples, shuffle=False, logger=logger
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create domain adapter config
    domain_adapter_config = create_domain_adapter_config_from_hydra(config)
    logger.info(f"Domain adapter config: {domain_adapter_config}")
    
    # Compute energy statistics from data
    logger.info("Computing energy statistics from data...")
    unified_config = create_unified_config_from_hydra(config)
    registry = UnifiedDatasetRegistry(unified_config)
    datasets = registry.load_all_datasets()
    
    # Collect energy values from train set
    train_energies = []
    for graph, targets in datasets['train']:
        if 'energy' in targets:
            train_energies.append(targets['energy'].item())
    
    train_energies = torch.tensor(train_energies)
    energy_mean = train_energies.mean().item()
    energy_std = train_energies.std().item()
    
    logger.info(f"Energy statistics: mean={energy_mean:.6f}, std={energy_std:.6f}")
    
    # Create model
    logger.info("Creating model...")
    model = DomainAwareSchNet(
        hidden_channels=config.model.hidden_channels,
        num_filters=config.model.num_filters,
        num_interactions=config.model.num_interactions,
        num_gaussians=config.model.num_gaussians,
        cutoff=config.model.cutoff,
        max_num_neighbors=config.model.max_num_neighbors,
        readout=config.model.readout,
        dipole=config.model.dipole,
        mean=energy_mean,
        std=energy_std,
        atomref=config.model.atomref,
        domain_adapter_config=domain_adapter_config
    ).to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create multi-task config
    multi_task_config = create_multi_task_config_from_hydra(config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MultiStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        multi_task_config=multi_task_config,
        domain_adapter_config=domain_adapter_config,
        config=config,
        logger=logger
    )
    
    # Train model
    logger.info("Starting multi-stage training...")
    best_model_path = trainer.train_multi_stage(
        save_dir=save_dir / f"ensemble_{ensemble_id}",
        log_dir=log_dir
    )
    
    logger.info(f"Training completed. Best model saved to: {best_model_path}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train N=5 ensemble models on unified dataset')
    parser.add_argument('--config', type=str, default='src/config/multi_stage_dataset.yaml',
                       help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='models/ensemble_unified',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs/ensemble_unified',
                       help='Directory to save logs')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--ensemble_size', type=int, default=5,
                       help='Number of ensemble models to train')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start ensemble ID (for resuming)')
    
    args = parser.parse_args()
    
    # Create directories
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.training.batch_size = args.batch_size
    
    print(f"Training {args.ensemble_size} ensemble models on unified dataset")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {args.device}")
    
    # Train ensemble models
    ensemble_paths = []
    start_time = datetime.now()
    
    for i in range(args.start_from, args.ensemble_size):
        print(f"\n{'='*60}")
        print(f"Training ensemble model {i+1}/{args.ensemble_size}")
        print(f"{'='*60}")
        
        try:
            model_path = train_single_ensemble_model(
                ensemble_id=i,
                config=config,
                args=args,
                save_dir=save_dir,
                log_dir=log_dir
            )
            ensemble_paths.append(model_path)
            print(f"✅ Ensemble model {i+1} completed successfully")
            
        except Exception as e:
            print(f"❌ Ensemble model {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save ensemble metadata
    metadata = {
        "ensemble_size": len(ensemble_paths),
        "total_requested": args.ensemble_size,
        "start_from": args.start_from,
        "training_duration": str(duration),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_file": args.config,
        "batch_size": args.batch_size,
        "device": args.device,
        "max_samples": args.max_samples,
        "model_config": {
            "hidden_channels": config.model.hidden_channels,
            "num_filters": config.model.num_filters,
            "num_interactions": config.model.num_interactions,
            "num_gaussians": config.model.num_gaussians,
            "cutoff": config.model.cutoff,
            "max_num_neighbors": config.model.max_num_neighbors
        },
        "checkpoint_files": [Path(p).name for p in ensemble_paths],
        "full_paths": ensemble_paths
    }
    
    metadata_file = save_dir / "ensemble_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Successfully trained: {len(ensemble_paths)}/{args.ensemble_size} models")
    print(f"Training duration: {duration}")
    print(f"Models saved to: {save_dir}")
    print(f"Metadata saved to: {metadata_file}")
    
    if len(ensemble_paths) == args.ensemble_size:
        print("🎉 All ensemble models trained successfully!")
    else:
        print(f"⚠️  {args.ensemble_size - len(ensemble_paths)} models failed to train")


if __name__ == "__main__":
    main()


