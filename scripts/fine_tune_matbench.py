#!/usr/bin/env python3
"""
Fine-tune GemNet model on Matbench task.

This script loads a pretrained model and fine-tunes it on Matbench data.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_gemnet", 
    Path(__file__).parent.parent / "src" / "gnn" / "model_gemnet.py"
)
model_gemnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_gemnet)
GemNetWrapper = model_gemnet.GemNetWrapper

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def structure_to_pyg_data(structure, target=None, norm_mean=None, norm_std=None):
    """Convert pymatgen Structure to PyG Data object."""
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    positions = torch.tensor(structure.cart_coords, dtype=torch.float32)
    
    # Create edge connectivity
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # Handle target normalization
    if target is not None:
        if norm_mean is not None and norm_std is not None:
            target_normalized = (target - norm_mean) / norm_std
        else:
            target_normalized = target
        energy_target = torch.tensor([target_normalized], dtype=torch.float32)
        energy_target_original = torch.tensor([target], dtype=torch.float32)
    else:
        energy_target = None
        energy_target_original = None
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=energy_target,
        energy_target_original=energy_target_original,
        n_atoms=torch.tensor([len(atomic_numbers)], dtype=torch.long)
    )
    
    # Add cell if periodic (ensure it's 3x3)
    if structure.is_ordered and hasattr(structure, 'lattice'):
        cell_matrix = structure.lattice.matrix
        if cell_matrix.shape == (3, 3):
            cell = torch.tensor(cell_matrix, dtype=torch.float32)
            data.cell = cell
        # If cell is not 3x3, don't add it (will use non-periodic computation)
    
    return data


def main():
    """Fine-tune model on Matbench task."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune GemNet on Matbench task')
    parser.add_argument('--pretrained-model', type=str,
                       default='models/gemnet_baseline/best_model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--task', type=str, default='perovskites',
                       choices=['perovskites'],
                       help='Matbench task')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--output-dir', type=str,
                       default='models/matbench_perovskites_finetuned',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load Matbench data
    try:
        from matminer.datasets import load_dataset
        logger.info(f"Loading Matbench {args.task} dataset...")
        df = load_dataset(f'matbench_{args.task}')
        logger.info(f"Loaded {len(df)} samples")
        
        # Split: use official Matbench splits if available, else 80/10/10
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:int(len(df) * 0.9)]
        test_df = df.iloc[int(len(df) * 0.9):]
        
        logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Determine target column
        if args.task == 'perovskites':
            structure_col = 'structure'
            target_col = 'e_form'
        
    except ImportError:
        logger.error("matminer not available. Install with: pip install matminer")
        return
    except Exception as e:
        logger.error(f"Error loading Matbench data: {e}")
        return
    
    # Compute normalization from training data
    train_targets = train_df[target_col].values
    norm_mean = np.mean(train_targets)
    norm_std = np.std(train_targets)
    logger.info(f"Normalization: mean={norm_mean:.6f}, std={norm_std:.6f}")
    
    # Load pretrained model
    logger.info(f"Loading pretrained model from {args.pretrained_model}")
    checkpoint = torch.load(args.pretrained_model, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('model_config', {})
    
    # Create model (use same architecture but update normalization)
    # Note: We need to use the pretrained model's architecture but with new normalization
    # However, GemNetWrapper stores mean/std as model parameters, so we'll initialize
    # with new stats but load pretrained weights
    model = GemNetWrapper(
        num_atoms=model_config.get('num_atoms', 120),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_filters=model_config.get('num_filters', 256),
        num_interactions=model_config.get('num_interactions', 6),
        cutoff=model_config.get('cutoff', 10.0),
        readout="sum",
        mean=norm_mean,  # Use Matbench normalization for training
        std=norm_std,
        use_film=model_config.get('use_film', False),
        num_domains=model_config.get('num_domains', 0),
        film_dim=model_config.get('film_dim', 16)
    ).to(device)
    
    # Load pretrained weights (may not match exactly due to normalization change)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("✓ Loaded pretrained weights (some may be missing due to normalization change)")
    except Exception as e:
        logger.warning(f"Could not load some weights: {e}")
        logger.info("Continuing with partial weights...")
    
    # Prepare data
    def prepare_data(df_subset):
        data_list = []
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Preparing data"):
            try:
                structure = row[structure_col]
                target = row[target_col]
                data = structure_to_pyg_data(structure, target, norm_mean, norm_std)
                data_list.append(data)
            except Exception as e:
                logger.debug(f"Error on sample {idx}: {e}")
                continue
        return data_list
    
    logger.info("Preparing training data...")
    train_data = prepare_data(train_df)
    logger.info("Preparing validation data...")
    val_data = prepare_data(val_df)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    logger.info("Starting fine-tuning...")
    for epoch in range(1, args.num_epochs + 1):
        # Training
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} - Train"):
            batch = batch.to(device)
            
            # Predict (no domain_id needed for baseline model)
            batch_size = batch.batch.max().item() + 1
            domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            try:
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
            except Exception as e:
                logger.debug(f"Error in forward pass: {e}")
                # Try without cell if PBC error
                batch_no_cell = batch.clone()
                batch_no_cell.cell = None
                energies_total, _, _ = model(batch_no_cell, compute_forces=False, domain_id=domain_id)
            
            # Convert from total (normalized) to per-atom (normalized)
            energies_per_atom = []
            targets_list = []
            
            for i in range(batch_size):
                mask = (batch.batch == i)
                n_atoms = mask.sum().item()
                if n_atoms > 0:
                    energies_per_atom.append(energies_total[i] / n_atoms)
                    # Get target from original data
                    batch_list = batch.to_data_list()
                    targets_list.append(batch_list[i].energy_target[0])
            
            energies_per_atom = torch.stack(energies_per_atom)
            targets = torch.stack(targets_list)
            
            loss = criterion(energies_per_atom, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.num_epochs} - Val"):
                batch = batch.to(device)
                batch_size = batch.batch.max().item() + 1
                domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                try:
                    energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
                except Exception as e:
                    # Try without cell if PBC error
                    batch_no_cell = batch.clone()
                    batch_no_cell.cell = None
                    energies_total, _, _ = model(batch_no_cell, compute_forces=False, domain_id=domain_id)
                
                # Convert from total to per-atom
                energies_per_atom = []
                targets_list = []
                
                for i in range(batch_size):
                    mask = (batch.batch == i)
                    n_atoms = mask.sum().item()
                    if n_atoms > 0:
                        energies_per_atom.append(energies_total[i] / n_atoms)
                        batch_list = batch.to_data_list()
                        targets_list.append(batch_list[i].energy_target[0])
                
                energies_per_atom = torch.stack(energies_per_atom)
                targets = torch.stack(targets_list)
                loss = criterion(energies_per_atom, targets)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}/{args.num_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'num_atoms': model.num_atoms,
                    'hidden_dim': model.hidden_dim,
                    'num_filters': 256,
                    'num_interactions': 6,
                    'cutoff': 10.0,
                    'use_film': False,
                    'num_domains': 0
                },
                'normalization': {'mean': norm_mean, 'std': norm_std},
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"✓ Saved best model (val_loss={best_val_loss:.6f})")
    
    logger.info(f"✅ Fine-tuning complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Model saved to {output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()

