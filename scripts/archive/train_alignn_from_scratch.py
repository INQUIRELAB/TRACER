#!/usr/bin/env python3
"""
Train ALIGNN model architecture on our unified dataset
Using ALIGNN models but our own data loading to bypass issues
"""

import os
# Force RTX 4090 (GPU 1) for better CUDA compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ase.io import read
from jarvis.core.atoms import Atoms as JAtoms
from alignn.models.alignn_atomwise import ALIGNNAtomWise

def load_and_format_data():
    """Load data in format ALIGNN can use."""
    logger.info("📥 Loading unified dataset...")
    
    # Load extxyz
    train_ase = read('data/alignn_unified/train_fixed.extxyz', ':')
    val_ase = read('data/alignn_unified/val_fixed.extxyz', ':')
    
    logger.info(f"   Train: {len(train_ase)} samples")
    logger.info(f"   Val: {len(val_ase)} samples")
    
    # Extract structures and energies
    train_structures = []
    train_energies = []
    
    for atoms in train_ase:
        # Convert to JARVIS Atoms
        pos = atoms.get_positions()
        elements = atoms.get_chemical_symbols()
        cell = atoms.get_cell().array
        
        if cell.sum() == 0:
            # Create bounding box
            max_dist = np.max(pos) - np.min(pos) if len(pos) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=elements,
            coords=pos.tolist()
        )
        
        energy = atoms.info.get('energy', 0.0) / len(atoms)  # Per atom
        train_structures.append(j_atoms)
        train_energies.append(energy)
    
    val_structures = []
    val_energies = []
    for atoms in val_ase:
        pos = atoms.get_positions()
        elements = atoms.get_chemical_symbols()
        cell = atoms.get_cell().array
        
        if cell.sum() == 0:
            max_dist = np.max(pos) - np.min(pos) if len(pos) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=elements,
            coords=pos.tolist()
        )
        
        energy = atoms.info.get('energy', 0.0) / len(atoms)
        val_structures.append(j_atoms)
        val_energies.append(energy)
    
    logger.info("✅ Data loaded")
    return train_structures, train_energies, val_structures, val_energies


def create_alignn_model():
    """Create ALIGNN model."""
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
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def collate_batch(batch):
    """Custom collate for ALIGNN."""
    from alignn.graphs import Graph
    from dgl import batch as dgl_batch
    import numpy as np
    
    graphs = []
    line_graphs = []
    targets = []
    lattices = []
    
    for atoms, energy in batch:
        # Create DGL graph (returns tuple: graph, line_graph)
        g, lg = Graph.atom_dgl_multigraph(atoms)
        
        # Add lattice info - ALIGNN requires grad=True
        lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
        lattices.append(lat)
        
        graphs.append(g)
        line_graphs.append(lg)
        targets.append(torch.tensor([energy], dtype=torch.float32))
    
    batched_graphs = dgl_batch(graphs)
    batched_line_graphs = dgl_batch(line_graphs)
    batched_targets = torch.cat(targets)
    batched_lattices = torch.stack(lattices)  # (batch_size, 3, 3)
    
    return (batched_graphs, batched_line_graphs, batched_lattices), batched_targets


def train():
    """Train ALIGNN model."""
    logger.info("🚀 TRAINING ALIGNN ON UNIFIED DATASET")
    logger.info("="*80)
    
    # Load data
    train_structures, train_energies, val_structures, val_energies = load_and_format_data()
    
    # Create model
    model = create_alignn_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logger.info(f"   Device: {device}")
    
    # Setup training
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch, num_workers=0)
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    
    # Training loop
    output_dir = Path('models/alignn_official')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for (g, lg, lat), targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            targets = targets.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Forward
            output_dict = model((g, lat))
            # Extract energy prediction from dict
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            output = output.squeeze()
            loss = criterion(output, targets)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
            train_count += len(targets)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        # NOTE: Don't use torch.no_grad() because ALIGNN uses grad() internally for forces
        for (g, lg, lat), targets in val_loader:
            g = g.to(device)
            lg = lg.to(device)
            lat = lat.to(device)
            targets = targets.to(device).squeeze()
            
            output_dict = model((g, lat))
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            output = output.squeeze()
            loss = criterion(output, targets)
            
            # Use numel() instead of len() to handle 0-d tensors (batch size 1)
            batch_size = targets.numel()
            val_loss += loss.item() * batch_size
            val_count += batch_size
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, output_dir / 'best_model.pt')
            logger.info(f"   ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

