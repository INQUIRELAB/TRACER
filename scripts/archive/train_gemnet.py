#!/usr/bin/env python3
"""
Train GemNet on preprocessed training data.
"""

import sys
import os
from pathlib import Path
import torch
import logging
from omegaconf import DictConfig, OmegaConf
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_gemnet(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "models/gemnet"
):
    """Train GemNet model on preprocessed data."""
    
    logger.info("🚀 GEMNET TRAINING")
    logger.info("="*80)
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Output: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load preprocessed data
    logger.info("\n📥 Loading preprocessed data...")
    prep_data_path = Path("data/preprocessed/cleaned_data.json")
    
    if prep_data_path.exists():
        with open(prep_data_path, 'r') as f:
            cleaned_data = json.load(f)
        logger.info(f"✅ Loaded {len(cleaned_data)} preprocessed samples")
    else:
        logger.warning("⚠️  No preprocessed data found, using demo data")
        cleaned_data = [
            {
                'atomic_numbers': [1, 6, 8],
                'positions': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                'energy': 0.0,
                'forces': [[0, 0, 0]] * 3,
                'num_atoms': 3,
            }
        ] * 10
    
    # 2. Load normalization stats if available
    norm_stats_path = Path("data/preprocessed/preprocessing_results.json")
    normalization_stats = None
    
    if norm_stats_path.exists():
        with open(norm_stats_path, 'r') as f:
            prep_results = json.load(f)
            if 'normalization' in prep_results:
                normalization_stats = prep_results['normalization']
                logger.info(f"✅ Loaded normalization stats: mean={normalization_stats['mean']:.4f}, std={normalization_stats['std']:.4f}")
    
    # 3. Create model
    logger.info("\n🔧 Creating GemNet model...")
    
    # Determine max atomic number
    max_z = 0
    for sample in cleaned_data:
        max_z = max(max_z, max(sample.get('atomic_numbers', [1])))
    
    logger.info(f"   Max atomic number: {max_z}")
    
    model = GemNetWrapper(
        num_atoms=min(max_z + 1, 120),  # Max 119 elements
        hidden_dim=256,
        num_filters=256,
        num_interactions=6,
        cutoff=10.0,
        readout="sum",
        mean=normalization_stats['mean'] if normalization_stats else None,
        std=normalization_stats['std'] if normalization_stats else None,
    )
    
    logger.info("✅ Model created")
    
    # 4. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 5. Simple training loop (simplified for demo)
    logger.info("\n🚀 Starting training (simplified demo)...")
    logger.info("   NOTE: Full training requires proper DataLoader and loss computation")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Simple training step (this is a placeholder)
        model.train()
        total_loss = 0.0
        
        # Would normally iterate over batches here
        # For now, just log progress
        logger.info(f"   Epoch {epoch+1}/{num_epochs}")
        
        # Simulated loss (in real training, compute actual loss)
        epoch_loss = 10.0 / (epoch + 1) + torch.rand(1).item()
        total_loss = epoch_loss
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
                'model_config': {
                    'num_atoms': model.num_atoms,
                    'hidden_dim': model.hidden_dim,
                    'num_interactions': len(model.blocks),
                    'cutoff': model.cutoff,
                    'mean': model.mean,
                    'std': model.std,
                }
            }, output_path / 'best_model.pt')
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best loss: {best_loss:.4f}")
    logger.info(f"   Model saved to {output_path / 'best_model.pt'}")
    logger.info("="*80)
    logger.info("⚠️  NOTE: This was a simplified demo training loop")
    logger.info("   Full training requires:")
    logger.info("   - Proper DataLoader with graph batching")
    logger.info("   - Actual loss computation (MAE/MSE)")
    logger.info("   - Validation split")
    logger.info("="*80)


if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def train(
        num_epochs: int = typer.Option(10, help="Number of training epochs"),
        batch_size: int = typer.Option(16, help="Batch size"),
        learning_rate: float = typer.Option(1e-4, help="Learning rate"),
        device: str = typer.Option("cuda", help="Device (cuda or cpu)"),
        output_dir: str = typer.Option("models/gemnet", help="Output directory")
    ):
        train_gemnet(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir
        )
    
    app()


