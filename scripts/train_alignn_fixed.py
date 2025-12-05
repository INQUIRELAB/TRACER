#!/usr/bin/env python3
"""
Fixed ALIGNN Training Script
Fixes:
1. Proper validation with gradient handling
2. Energy normalization
3. Better regularization
4. Proper loss calculation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ase.io import read
from jarvis.core.atoms import Atoms as JAtoms
from alignn.graphs import Graph
from dgl import batch as dgl_batch

def compute_normalization_stats(train_energies):
    """Compute normalization statistics for energy targets."""
    energies = np.array(train_energies)
    mean = np.mean(energies)
    std = np.std(energies)
    
    logger.info(f"   Energy normalization: mean={mean:.6f}, std={std:.6f}")
    return mean, std


def load_and_format_data():
    """Load data in format ALIGNN can use, with proper normalization."""
    logger.info("📥 Loading unified dataset...")
    
    # Load from preprocessed JSON for consistency
    train_file = Path('data/preprocessed_full_unified/train_data.json')
    val_file = Path('data/preprocessed_full_unified/val_data.json')
    
    def load_json(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    train_samples = load_json(train_file) if train_file.exists() else []
    val_samples = load_json(val_file) if val_file.exists() else []
    
    logger.info(f"   Train: {len(train_samples)} samples")
    logger.info(f"   Val: {len(val_samples)} samples")
    
    # Convert to ALIGNN format
    train_structures = []
    train_energies_per_atom = []
    
    for sample in train_samples:
        atomic_numbers = sample['atomic_numbers']
        positions = np.array(sample['positions'])
        energy = sample.get('energy', sample.get('energy_target', sample.get('formation_energy_per_atom', 0.0)))
        n_atoms = len(atomic_numbers)
        
        if n_atoms == 0:
            continue
        
        # CRITICAL FIX: Data is already in per-atom format (formation_energy_per_atom)
        # Do NOT divide by n_atoms (this was the bug!)
        # Only convert if energy is suspiciously large (>50 eV suggests total energy)
        if abs(energy) > 50 and n_atoms > 0:
            # Heuristic: if energy is very large, assume it's total and convert
            energy_per_atom = energy / n_atoms
        else:
            # Already per-atom (typical range: -5 to 2 eV/atom)
            energy_per_atom = energy
        
        # Convert atomic numbers to elements
        from ase.data import chemical_symbols
        elements = [chemical_symbols[z] for z in atomic_numbers]
        
        # Create cell (for molecular systems, use bounding box)
        if 'cell' in sample and sample['cell']:
            cell = np.array(sample['cell'])
        else:
            max_dist = np.max(positions) - np.min(positions) if len(positions) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        # Create JARVIS Atoms (Cartesian coordinates)
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=elements,
            coords=positions.tolist()  # Cartesian coordinates
        )
        
        train_structures.append(j_atoms)
        train_energies_per_atom.append(energy_per_atom)
    
    val_structures = []
    val_energies_per_atom = []
    
    for sample in val_samples:
        atomic_numbers = sample['atomic_numbers']
        positions = np.array(sample['positions'])
        energy = sample.get('energy', sample.get('energy_target', sample.get('formation_energy_per_atom', 0.0)))
        n_atoms = len(atomic_numbers)
        
        if n_atoms == 0:
            continue
        
        # CRITICAL FIX: Data is already in per-atom format
        # Do NOT divide by n_atoms (this was the bug!)
        if abs(energy) > 50 and n_atoms > 0:
            energy_per_atom = energy / n_atoms
        else:
            energy_per_atom = energy
        
        from ase.data import chemical_symbols
        elements = [chemical_symbols[z] for z in atomic_numbers]
        
        if 'cell' in sample and sample['cell']:
            cell = np.array(sample['cell'])
        else:
            max_dist = np.max(positions) - np.min(positions) if len(positions) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=elements,
            coords=positions.tolist()
        )
        
        val_structures.append(j_atoms)
        val_energies_per_atom.append(energy_per_atom)
    
    # Compute normalization stats from training data
    energy_mean, energy_std = compute_normalization_stats(train_energies_per_atom)
    
    # Normalize energies
    train_energies_normalized = [(e - energy_mean) / (energy_std + 1e-8) for e in train_energies_per_atom]
    val_energies_normalized = [(e - energy_mean) / (energy_std + 1e-8) for e in val_energies_per_atom]
    
    logger.info("✅ Data loaded and normalized")
    
    return (train_structures, train_energies_normalized, 
            val_structures, val_energies_normalized,
            energy_mean, energy_std)


def create_alignn_model():
    """Create ALIGNN model with proper configuration."""
    from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
    
    config = ALIGNNAtomWiseConfig(
        name='alignn_atomwise',
        atom_input_features=92,
        embedding_features=256,
        hidden_features=256,
        alignn_layers=2,
        gcn_layers=2,
        output_features=1,
    )
    
    model = ALIGNNAtomWise(config=config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model created: {num_params:,} parameters")
    
    return model


def collate_batch(batch):
    """Custom collate for ALIGNN."""
    graphs = []
    line_graphs = []
    targets = []
    lattices = []
    
    for atoms, energy in batch:
        # Create DGL graph
        g, lg = Graph.atom_dgl_multigraph(atoms)
        
        # Lattice tensor - ALIGNN requires grad=True for its internal grad() calls
        lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
        lattices.append(lat)
        
        graphs.append(g)
        line_graphs.append(lg)
        targets.append(torch.tensor([energy], dtype=torch.float32))
    
    batched_graphs = dgl_batch(graphs)
    batched_line_graphs = dgl_batch(line_graphs)
    batched_targets = torch.cat(targets)
    batched_lattices = torch.stack(lattices)
    
    return (batched_graphs, batched_line_graphs, batched_lattices), batched_targets


def train():
    """Train ALIGNN model with fixes."""
    logger.info("🚀 FIXED ALIGNN TRAINING ON UNIFIED DATASET")
    logger.info("="*80)
    
    # Load data with normalization
    (train_structures, train_energies, val_structures, val_energies,
     energy_mean, energy_std) = load_and_format_data()
    
    if len(train_structures) == 0:
        logger.error("No training data found!")
        return
    
    # Create model
    model = create_alignn_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"   Device: {device}")
    
    # Setup training with better regularization
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-4,  # Slightly lower LR for stability
        weight_decay=1e-4  # Stronger weight decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Create dataloader
    from torch.utils.data import Dataset, DataLoader
    
    class StructureDataset(Dataset):
        def __init__(self, structures, energies):
            self.structures = structures
            self.energies = energies
        
        def __len__(self):
            return len(self.structures)
        
        def __getitem__(self, idx):
            return self.structures[idx], self.energies[idx]
    
    train_dataset = StructureDataset(train_structures, train_energies)
    val_dataset = StructureDataset(val_structures, val_energies)
    
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, 
        collate_fn=collate_batch, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, 
        collate_fn=collate_batch, num_workers=0
    )
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    
    # Training loop
    output_dir = Path('models/alignn_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = 10
    train_losses = []
    val_losses = []
    
    # FAIR COMPARISON: Train for same number of epochs as GemNet (50)
    num_epochs = 50
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for (g, lg, lat), targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            targets = targets.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward - ALIGNN might need gradients internally, but we wrap carefully
            try:
                # Try with no_grad wrapper
                with torch.enable_grad():
                    # Ensure lat requires grad if needed by ALIGNN
                    if not lat.requires_grad:
                        lat = lat.requires_grad_(True)
                    
                    output_dict = model((g, lat))
            except RuntimeError:
                # If that fails, try without wrapper
                lat = lat.requires_grad_(True)
                output_dict = model((g, lat))
            
            # Extract prediction
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            output = output.squeeze()
            
            # Loss on normalized targets
            loss = criterion(output, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_count += len(targets)
        
        # Validation - ALIGNN needs gradients internally, but we don't want to update model params
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        # Don't use torch.no_grad() because ALIGNN uses grad() internally
        # Instead, we'll just not call backward() - gradients computed internally won't affect model
        for (g, lg, lat), targets in val_loader:
            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            targets = targets.to(device).squeeze()
            
            # Forward pass - ALIGNN needs grad enabled for its internal grad() calls
            # But we won't call backward(), so model params won't be updated
            output_dict = model((g, lat))
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            output = output.squeeze()
            
            # Compute loss (detach to prevent any gradient computation if needed)
            with torch.no_grad():
                loss = criterion(output.detach(), targets)
            
            batch_size = targets.numel()
            val_loss += loss.item() * batch_size
            val_count += batch_size
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'epoch': epoch,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, output_dir / 'best_model.pt')
            logger.info(f"   ✓ Saved best model (val_loss: {best_val_loss:.6f})")
        else:
            patience += 1
            if patience >= max_patience:
                logger.info(f"   Early stopping at epoch {epoch+1}")
                break
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.6f}")
    logger.info(f"   Energy normalization: mean={energy_mean:.6f}, std={energy_std:.6f}")


if __name__ == "__main__":
    train()

