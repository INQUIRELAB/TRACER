#!/usr/bin/env python3
"""
Train GemNet on Full Preprocessed Dataset
Includes proper train/validation/test splits and evaluation.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_preprocessed_data():
    """Load preprocessed train/val/test splits."""
    data_path = Path("data/preprocessed_full_unified")
    
    logger.info("📥 Loading preprocessed data splits...")
    
    with open(data_path / 'train_data.json', 'r') as f:
        train_data = json.load(f)
    with open(data_path / 'val_data.json', 'r') as f:
        val_data = json.load(f)
    with open(data_path / 'test_data.json', 'r') as f:
        test_data = json.load(f)
    
    with open(data_path / 'preprocessing_results.json', 'r') as f:
        prep_results = json.load(f)
        norm_stats = prep_results.get('normalization', {})
    
    # Re-normalize data properly (compute from actual data)
    import numpy as np
    all_energies = []
    for s in train_data:
        energy = s.get('energy', s.get('energy_target', 0))
        all_energies.append(energy)
    
    if len(all_energies) > 0:
        norm_mean = np.mean(all_energies)
        norm_std = np.std(all_energies)
        
        # Store correct normalization
        norm_stats = {'mean': norm_mean, 'std': norm_std}
        
        logger.info(f"   Computed normalization: mean={norm_mean:.4f}, std={norm_std:.4f}")
    
    logger.info(f"✅ Loaded:")
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Validation: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    logger.info(f"   Normalization: mean={norm_stats.get('mean', 0):.4f}, std={norm_stats.get('std', 1):.4f}")
    
    return train_data, val_data, test_data, norm_stats


def sample_to_pyg_data(sample):
    """Convert sample dict to PyTorch Geometric Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Create simple edge connectivity based on distance
    from torch_geometric.nn import radius_graph
    edge_index = radius_graph(positions, r=10.0, max_num_neighbors=32)
    
    # Create Data object
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        energy_target=torch.tensor([sample.get('energy', sample.get('energy_target', 0.0))], dtype=torch.float32)
    )
    
    return data


def train_gemnet_full(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "models/gemnet_full"
):
    """Train GemNet on full preprocessed dataset."""
    
    logger.info("🚀 TRAINING GEMNET ON FULL DATASET")
    logger.info("="*80)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"   Device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    train_data, val_data, test_data, norm_stats = load_preprocessed_data()
    
    # 2. Convert to PyG Data objects
    logger.info("\n🔄 Converting to PyG format...")
    train_pyg = [sample_to_pyg_data(s) for s in train_data]
    val_pyg = [sample_to_pyg_data(s) for s in val_data]
    test_pyg = [sample_to_pyg_data(s) for s in test_data]
    
    # 3. Create dataloaders
    train_loader = PyGDataLoader(train_pyg, batch_size=min(batch_size, len(train_pyg)), shuffle=True)
    val_loader = PyGDataLoader(val_pyg, batch_size=min(batch_size, len(val_pyg)), shuffle=False)
    test_loader = PyGDataLoader(test_pyg, batch_size=min(batch_size, len(test_pyg)), shuffle=False)
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # 4. Create model
    logger.info("\n🔧 Creating GemNet model...")
    
    max_z = max([max(s['atomic_numbers']) for s in train_data])
    model = GemNetWrapper(
        num_atoms=min(max_z + 1, 120),
        hidden_dim=256,
        num_filters=256,
        num_interactions=6,
        cutoff=10.0,
        readout="sum",
        mean=norm_stats.get('mean'),
        std=norm_stats.get('std'),
    ).to(device)
    
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Setup training
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 6. Training loop
    logger.info("\n🚀 Starting training...")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            energies, _, _ = model(batch, compute_forces=False)
            
            # Loss
            loss = criterion(energies, batch.energy_target)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            train_count += batch.num_graphs
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                energies, _, _ = model(batch, compute_forces=False)
                loss = criterion(energies, batch.energy_target)
                val_loss += loss.item() * batch.num_graphs
                val_count += batch.num_graphs
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'num_atoms': model.num_atoms,
                    'hidden_dim': model.hidden_dim,
                    'num_interactions': len(model.blocks),
                    'cutoff': model.cutoff,
                    'mean': model.mean,
                    'std': model.std,
                }
            }, output_path / 'best_model.pt')
            logger.info(f"   ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # 7. Test evaluation
    logger.info("\n📊 Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            energies, _, _ = model(batch, compute_forces=False)
            loss = criterion(energies, batch.energy_target)
            test_loss += loss.item() * batch.num_graphs
            test_count += batch.num_graphs
    
    avg_test_loss = test_loss / test_count if test_count > 0 else 0.0
    logger.info(f"   Test Loss: {avg_test_loss:.4f}")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Test loss: {avg_test_loss:.4f}")
    logger.info(f"   Model saved to: {output_path / 'best_model.pt'}")
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
        output_dir: str = typer.Option("models/gemnet_full", help="Output directory")
    ):
        train_gemnet_full(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir
        )
    
    app()

