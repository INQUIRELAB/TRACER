#!/usr/bin/env python3
"""
Train GemNet Baseline (No FiLM, No Domain Embedding) for Ablation Study
This trains a GemNet model without any domain adaptation components.
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
    
    # Compute normalization stats
    import numpy as np
    
    all_per_atom_energies = []
    for s in train_data:
        if 'formation_energy_per_atom' in s:
            per_atom_energy = s['formation_energy_per_atom']
        elif 'energy' in s:
            energy = s['energy']
            n_atoms = len(s.get('positions', []))
            if abs(energy) > 50 and n_atoms > 0:
                per_atom_energy = energy / n_atoms
            else:
                per_atom_energy = energy
        else:
            per_atom_energy = s.get('energy_target', 0.0)
        
        all_per_atom_energies.append(per_atom_energy)
    
    if len(all_per_atom_energies) > 0:
        norm_mean = np.mean(all_per_atom_energies)
        norm_std = np.std(all_per_atom_energies)
        norm_stats = {'mean': norm_mean, 'std': norm_std}
        logger.info(f"   Normalization: mean={norm_mean:.6f}, std={norm_std:.6f}")
    else:
        norm_stats = {'mean': 0.0, 'std': 1.0}
    
    logger.info(f"✅ Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data, norm_stats


def sample_to_pyg_data(sample, norm_mean=None, norm_std=None):
    """Convert sample dict to PyTorch Geometric Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Create edges using distance matrix (fallback for torch-cluster)
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # Get energy target (per-atom)
    if 'formation_energy_per_atom' in sample:
        per_atom_energy = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', 0.0)
        n_atoms = len(positions)
        if abs(energy) > 50 and n_atoms > 0:
            per_atom_energy = energy / n_atoms
        else:
            per_atom_energy = energy
    
    # Normalize
    if norm_mean is not None and norm_std is not None:
        per_atom_energy_normalized = (per_atom_energy - norm_mean) / norm_std
    else:
        per_atom_energy_normalized = per_atom_energy
    
    n_atoms = len(atomic_numbers)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=torch.tensor([per_atom_energy_normalized], dtype=torch.float32),
        energy_target_original=torch.tensor([per_atom_energy], dtype=torch.float32),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long)
    )
    
    # Add cell if available
    if 'cell' in sample and sample['cell'] is not None:
        data.cell = torch.tensor(sample['cell'], dtype=torch.float32)
    
    return data


def train_baseline(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "models/gemnet_baseline"
):
    """Train baseline GemNet model (no FiLM, no domain embedding."""
    
    logger.info("=" * 80)
    logger.info("  TRAINING GEMNET BASELINE (ABLATION)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"   No FiLM adaptation")
    logger.info(f"   No domain embeddings")
    logger.info(f"   Standard GemNet architecture")
    logger.info("")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"   Device: {device}")
    
    # Force RTX 4090
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_data, val_data, test_data, norm_stats = load_preprocessed_data()
    
    # Convert to PyG format
    logger.info("\n🔄 Converting to PyG format...")
    train_pyg = [sample_to_pyg_data(s, norm_stats.get('mean'), norm_stats.get('std')) for s in train_data]
    val_pyg = [sample_to_pyg_data(s, norm_stats.get('mean'), norm_stats.get('std')) for s in val_data]
    
    # Create dataloaders
    train_loader = PyGDataLoader(train_pyg, batch_size=min(batch_size, len(train_pyg)), shuffle=True)
    val_loader = PyGDataLoader(val_pyg, batch_size=min(batch_size, len(val_pyg)), shuffle=False)
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    
    # Create model (NO FiLM, NO domain embedding)
    logger.info("\n🔧 Creating GemNet Baseline model (no FiLM, no domain embedding)...")
    
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
        use_film=False,  # NO FiLM
        num_domains=0,    # NO domain embedding
        film_dim=0        # NO film
    ).to(device)
    
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   ✅ Baseline model (no adaptation)")
    
    # Setup training
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
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
            
            # Forward pass (NO domain_id needed)
            energies_total, _, _ = model(batch, compute_forces=False, domain_id=None)
            
            # Convert to per-atom
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_total.dtype, 
                                                  device=energies_total.device)
            
            energies_per_atom = energies_total / n_atoms_per_graph
            
            # Loss
            loss = criterion(energies_per_atom, batch.energy_target)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_count += 1
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                batch = batch.to(device)
                
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=None)
                
                # Convert to per-atom
                if hasattr(batch, 'batch'):
                    n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                    n_atoms_per_graph = n_atoms_per_graph.float()
                else:
                    n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                      dtype=energies_total.dtype, 
                                                      device=energies_total.device)
                
                energies_per_atom = energies_total / n_atoms_per_graph
                
                loss = criterion(energies_per_atom, batch.energy_target)
                val_loss += loss.item()
                val_count += 1
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
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
                'normalization': norm_stats,
                'target_type': 'per_atom',
                'model_config': {
                    'num_atoms': min(max_z + 1, 120),
                    'hidden_dim': 256,
                    'num_filters': 256,
                    'num_interactions': 6,
                    'cutoff': 10.0,
                    'use_film': False,  # Baseline: no FiLM
                    'num_domains': 0,   # Baseline: no domains
                }
            }, output_path / 'best_model.pt')
            logger.info(f"   ✅ Saved best model (val_loss={best_val_loss:.6f})")
    
    logger.info("="*80)
    logger.info("✅ Training completed!")
    logger.info(f"   Best validation loss: {best_val_loss:.6f} eV/atom")
    logger.info(f"   Model saved to: {output_path / 'best_model.pt'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GemNet Baseline for Ablation Study")
    parser.add_argument("train", type=str, help="Train command")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="models/gemnet_baseline", help="Output directory")
    
    args = parser.parse_args()
    
    train_baseline(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir
    )

