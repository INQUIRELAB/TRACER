"""
Multi-task training script using the enhanced trainer with missing label support.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from gnn.multi_task_trainer import MultiTaskTrainer, MultiTaskConfig
from gnn.model import SchNetWrapper
from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, create_unified_config_from_hydra


def create_multi_task_config_from_hydra(config) -> MultiTaskConfig:
    """Create MultiTaskConfig from Hydra config."""
    return MultiTaskConfig(
        # Loss weights
        w_e=config.training.get('w_e', 1.0),
        w_f=config.training.get('w_f', 10.0),
        w_s=config.training.get('w_s', 0.0),
        
        # Training parameters
        learning_rate=config.training.get('learning_rate', 1e-4),
        weight_decay=config.training.get('weight_decay', 1e-6),
        max_grad_norm=config.training.get('max_grad_norm', 1.0),
        
        # Scheduler parameters
        scheduler_type=config.training.get('scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=config.training.get('scheduler_patience', 10),
        scheduler_factor=config.training.get('scheduler_factor', 0.5),
        scheduler_min_lr=config.training.get('scheduler_min_lr', 1e-7),
        
        # Early stopping
        early_stopping_patience=config.training.get('early_stopping_patience', 20),
        early_stopping_min_delta=config.training.get('early_stopping_min_delta', 1e-6),
        
        # Logging
        log_every_n_steps=config.training.get('log_every_n_steps', 100),
        save_every_n_epochs=config.training.get('save_every_n_epochs', 10),
        
        # Multi-task specific
        enable_per_domain_logging=config.training.get('enable_per_domain_logging', True),
        missing_label_strategy=config.training.get('missing_label_strategy', 'ignore')
    )


def create_unified_dataloader(config, split: str, batch_size: int, 
                             max_samples: int = None, shuffle: bool = True, 
                             num_workers: int = 4):
    """Create unified dataloader for training/validation."""
    # Create unified config
    unified_config = create_unified_config_from_hydra(config)
    
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


def main():
    parser = argparse.ArgumentParser(description='Multi-task GNN training')
    parser.add_argument('--config', type=str, default='src/config/unified_dataset.yaml',
                       help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--w_e', type=float, default=1.0,
                       help='Energy loss weight')
    parser.add_argument('--w_f', type=float, default=10.0,
                       help='Force loss weight')
    parser.add_argument('--w_s', type=float, default=0.0,
                       help='Stress loss weight')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='models/multi_task',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs/multi_task',
                       help='Directory to save logs')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per dataset')
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.w_e = args.w_e
    config.training.w_f = args.w_f
    config.training.w_s = args.w_s
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set CUDA device
    if device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device.index)
        torch.cuda.set_device(device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_unified_dataloader(
        config, 'train', args.batch_size, 
        max_samples=args.max_samples, shuffle=True
    )
    val_loader = create_unified_dataloader(
        config, 'val', args.batch_size,
        max_samples=args.max_samples, shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = SchNetWrapper(
        hidden_channels=config.model.get('hidden_channels', 128),
        num_filters=config.model.get('num_filters', 128),
        num_interactions=config.model.get('num_interactions', 6),
        num_gaussians=config.model.get('num_gaussians', 50),
        cutoff=config.model.get('cutoff', 10.0),
        max_num_neighbors=config.model.get('max_num_neighbors', 32),
        readout=config.model.get('readout', 'add'),
        dipole=config.model.get('dipole', False),
        mean=config.model.get('mean', 0.0),
        std=config.model.get('std', 1.0),
        atomref=config.model.get('atomref', None)
    )
    
    # Create multi-task config
    multi_task_config = create_multi_task_config_from_hydra(config)
    
    # Create trainer
    print("Creating multi-task trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        config=multi_task_config,
        device=device,
        log_dir=args.log_dir
    )
    
    # Train
    print("Starting training...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Final epoch: {results['final_epoch']}")
    
    if results['domain_summary']:
        print("\nPer-domain performance summary:")
        for domain, metrics in results['domain_summary'].items():
            print(f"  {domain}:")
            for metric, stats in metrics.items():
                print(f"    {metric}: {stats['latest']:.6f}")


if __name__ == '__main__':
    main()


