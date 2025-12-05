#!/usr/bin/env python3
"""
Improved GNN Training Script
Fixes energy normalization issues and uses better hyperparameters
"""

import sys
import os
import torch
from pathlib import Path
import logging
import json
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.train import GNNTrainer, SupervisedLoss
from gnn.model import SchNetWrapper
from dft_hybrid.data.unified_registry import (
    UnifiedDatasetRegistry, UnifiedDatasetConfig,
    DatasetConfig, DatasetDomain
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_normalization_stats(dataset):
    """Compute mean and std for energy normalization."""
    logger.info("📊 Computing normalization statistics...")
    
    energies = []
    for i in tqdm(range(min(1000, len(dataset))), desc="Sampling energies"):
        data = dataset[i]
        if hasattr(data, 'energy'):
            if isinstance(data.energy, torch.Tensor):
                energies.append(data.energy.item())
            else:
                energies.append(data.energy)
    
    if len(energies) > 0:
        mean = torch.tensor(energies).mean().item()
        std = torch.tensor(energies).std().item()
        
        logger.info(f"   Mean energy: {mean:.4f} eV")
        logger.info(f"   Std energy: {std:.4f} eV")
        return mean, std
    else:
        logger.warning("⚠️  No energies found, using defaults")
        return 0.0, 1.0

def create_improved_dataset_config():
    """Create an improved dataset configuration."""
    config = UnifiedDatasetConfig(
        datasets=[
            DatasetConfig(
                domain_id=DatasetDomain.JARVIS_DFT,
                name="JARVIS-DFT",
                path="data/jarvis_dft",
                weight=1.0,
                enabled=True
            ),
            DatasetConfig(
                domain_id=DatasetDomain.JARVIS_ELASTIC,
                name="JARVIS-Elastic",
                path="data/jarvis_dft",  # Same path as DFT
                weight=0.5,
                enabled=True
            ),
        ],
        mix_strategy="uniform",
        validation_split=0.1,
        test_split=0.1,
        random_seed=42,
        normalize_energies=True,
    )
    return config

def train_improved_model(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "models/gnn_improved"
):
    """Train an improved GNN model with proper normalization handling."""
    
    logger.info("🚀 IMPROVED GNN TRAINING")
    logger.info("="*80)
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load dataset
    logger.info("\n📥 Loading dataset...")
    dataset_config = create_improved_dataset_config()
    registry = UnifiedDatasetRegistry(dataset_config)
    
    # Get all data samples
    all_samples = registry.get_all_samples(max_samples=10000)  # Limit for now
    logger.info(f"✅ Loaded {len(all_samples)} samples")
    
    if len(all_samples) == 0:
        logger.error("❌ No samples loaded!")
        return
    
    # 2. Compute normalization statistics
    mean, std = compute_normalization_stats(all_samples)
    
    # Save normalization stats
    norm_stats = {'mean': mean, 'std': std}
    with open(output_path / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f)
    logger.info(f"✅ Saved normalization stats to {output_path / 'normalization_stats.json'}")
    
    # 3. Create model with mean/std
    logger.info("\n🔧 Creating model...")
    model_config = {
        'hidden_channels': 256,
        'num_filters': 256,
        'num_interactions': 8,
        'num_gaussians': 64,
        'cutoff': 6.0,
        'max_num_neighbors': 64,
        'mean': mean,
        'std': std,
    }
    
    model = SchNetWrapper(**model_config)
    logger.info(f"✅ Created model with mean={mean:.4f}, std={std:.4f}")
    
    # 4. Create data loaders (simplified)
    # Split data
    n_train = int(0.8 * len(all_samples))
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]
    
    # Create basic dataloaders
    # Note: This is simplified - real implementation would use proper DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert samples to tensors (simplified)
    # In reality, we need to handle graphs properly
    logger.info(f"📊 Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # 5. Train model
    logger.info("\n🚀 Starting training...")
    logger.info(f"   This is a placeholder - full training needs proper data handling")
    logger.info(f"   Model created with proper normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'normalization': norm_stats,
        'best_val_loss': None,
    }, output_path / 'model.pt')
    
    logger.info(f"\n✅ Model saved to {output_path}")
    logger.info("="*80)
    logger.info("⚠️  NOTE: Full training requires proper DataLoader implementation")
    logger.info("   This script demonstrates the normalization approach")
    logger.info("="*80)

if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def train(
        num_epochs: int = typer.Option(50, help="Number of training epochs"),
        batch_size: int = typer.Option(16, help="Batch size"),
        learning_rate: float = typer.Option(1e-4, help="Learning rate"),
        device: str = typer.Option("cuda", help="Device (cuda or cpu)"),
        output_dir: str = typer.Option("models/gnn_improved", help="Output directory")
    ):
        train_improved_model(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir
        )
    
    app()



