"""
Multi-stage training script for domain-aware GNN with FiLM and LoRA.
Implements progressive training strategy for optimal domain adaptation.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from gnn.domain_aware_model import DomainAwareSchNet, DomainAdapterConfig
from gnn.multi_stage_trainer import MultiStageTrainer, DomainSpecificLoss
from gnn.multi_task_trainer import MultiTaskConfig
from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, create_unified_config_from_hydra


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
    parser = argparse.ArgumentParser(description='Multi-stage domain-aware GNN training')
    parser.add_argument('--config', type=str, default='src/config/domain_aware_dataset.yaml',
                       help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='models/multi_stage',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs/multi_stage',
                       help='Directory to save logs')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--stage', type=int, default=None,
                       help='Train specific stage only (0-3)')
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.training.batch_size = args.batch_size
    
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
    
    # Create domain adapter config
    domain_adapter_config = create_domain_adapter_config_from_hydra(config)
    
    # Compute energy statistics from data
    print("Computing energy statistics from data...")
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
    
    print(f"Energy statistics - Mean: {energy_mean:.6f}, Std: {energy_std:.6f}")
    
    # Create model
    print("Creating domain-aware model...")
    model = DomainAwareSchNet(
        hidden_channels=config.model.get('hidden_channels', 128),
        num_filters=config.model.get('num_filters', 128),
        num_interactions=config.model.get('num_interactions', 6),
        num_gaussians=config.model.get('num_gaussians', 50),
        cutoff=config.model.get('cutoff', 10.0),
        max_num_neighbors=config.model.get('max_num_neighbors', 32),
        readout=config.model.get('readout', 'add'),
        dipole=config.model.get('dipole', False),
        mean=energy_mean,
        std=energy_std,
        atomref=config.model.get('atomref', None),
        adapter_config=domain_adapter_config
    )
    
    # FIX: Apply the aggressive fix to the output layer
    print("Applying output layer fix...")
    schnet_model = model.schnet_model
    for name, module in schnet_model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.out_features == 1:
            print(f"Fixing output layer: {name}")
            with torch.no_grad():
                # Scale down weights by 1000x
                module.weight.data *= 0.001
                if hasattr(module, 'bias') and module.bias is not None:
                    # Set bias to target mean
                    module.bias.data.fill_(energy_mean)
            print(f"Output layer fixed - weights scaled by 1000x, bias set to {energy_mean:.6f}")
            break
    
    # Move model to device
    model = model.to(device)
    
    # Create multi-task config
    multi_task_config = create_multi_task_config_from_hydra(config)
    
    # Create multi-stage trainer
    print("Creating multi-stage trainer...")
    trainer = MultiStageTrainer(
        model=model,
        config=domain_adapter_config,
        device=device,
        base_lr=config.training.get('learning_rate', 1e-4)
    )
    
    # Print parameter summary
    param_summary = trainer.get_parameter_summary()
    print("\n📊 Parameter Summary:")
    for stage_name, summary in param_summary.items():
        print(f"  {stage_name}:")
        print(f"    Total: {summary['total_parameters']:,}")
        print(f"    Trainable: {summary['trainable_parameters']:,} ({summary['trainable_ratio']:.1%})")
        print(f"    Frozen: {summary['frozen_parameters']:,}")
    
    # Create domain-specific loss
    loss_fn = DomainSpecificLoss(multi_task_config)
    
    # Training
    if args.stage is not None:
        print(f"🎯 Training Stage {args.stage} only...")
        stage_result = trainer.train_stage(args.stage, train_loader, val_loader, loss_fn)
        print(f"Stage {args.stage} completed. Best val loss: {stage_result['best_val_loss']:.6f}")
    else:
        print("🎯 Starting Multi-Stage Training Pipeline...")
        stage_results = trainer.train_all_stages(train_loader, val_loader, multi_task_config)
        
        print("\n🎉 Multi-stage training completed!")
        print("Stage Results:")
        for stage_name, result in stage_results.items():
            print(f"  {stage_name}: Best val loss = {result['best_val_loss']:.6f}")
    
    # Save final model
    os.makedirs(args.save_dir, exist_ok=True)
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'adapter_config': domain_adapter_config,
        'multi_task_config': multi_task_config
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()
