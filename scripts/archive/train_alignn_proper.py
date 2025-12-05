#!/usr/bin/env python3
"""
Train ALIGNN properly on our unified dataset
Uses ALIGNN's dataset_array parameter to feed custom data
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ase.io import read
from jarvis.core.atoms import Atoms as JAtoms
import json

def ase_to_jarvis(atoms, energy):
    """Convert ASE Atoms to JARVIS Atoms with energy."""
    pos = atoms.get_positions().tolist()
    elements = list(atoms.get_chemical_symbols())
    
    # Get cell
    cell = atoms.get_cell().array
    
    # If no cell (molecule), use bounding box + padding
    if cell.sum() == 0:
        pos_arr = np.array(pos)
        max_dist = np.max(pos_arr) - np.min(pos_arr) if len(pos_arr) > 0 else 10.0
        padding = 5.0
        cell = np.eye(3) * (max_dist + 2 * padding)
    
    cell = cell.tolist()
    
    j_atoms = JAtoms(
        lattice_mat=cell,
        elements=elements,
        coords=pos
    )
    
    # Store energy in properties
    j_atoms.props = {'target': energy}
    
    return j_atoms


def load_custom_dataset():
    """Load our unified dataset."""
    logger.info("📥 Loading unified dataset...")
    
    # Load extxyz
    train_ase = read('data/alignn_unified/train_fixed.extxyz', ':')
    val_ase = read('data/alignn_unified/val_fixed.extxyz', ':')
    test_ase = read('data/alignn_unified/test_fixed.extxyz', ':')
    
    logger.info(f"   Train: {len(train_ase)} samples")
    logger.info(f"   Val: {len(val_ase)} samples")
    logger.info(f"   Test: {len(test_ase)} samples")
    
    # Convert and extract energies
    train_data = []
    val_data = []
    test_data = []
    
    # ALIGNN expects specific target keys like 'formation_energy_peratom'
    # Store energy per atom
    for idx, atoms in enumerate(train_ase):
        energy_total = atoms.info.get('energy', 0.0)
        n_atoms = len(atoms)
        energy_per_atom = energy_total / n_atoms if n_atoms > 0 else 0.0
        
        j_atoms = ase_to_jarvis(atoms, energy_per_atom)
        train_data.append({
            'atoms': j_atoms.to_dict(),
            'jid': f'train_{idx}',
            'formation_energy_peratom': energy_per_atom
        })
    
    for idx, atoms in enumerate(val_ase):
        energy_total = atoms.info.get('energy', 0.0)
        n_atoms = len(atoms)
        energy_per_atom = energy_total / n_atoms if n_atoms > 0 else 0.0
        
        j_atoms = ase_to_jarvis(atoms, energy_per_atom)
        val_data.append({
            'atoms': j_atoms.to_dict(),
            'jid': f'val_{idx}',
            'formation_energy_peratom': energy_per_atom
        })
    
    for idx, atoms in enumerate(test_ase):
        energy_total = atoms.info.get('energy', 0.0)
        n_atoms = len(atoms)
        energy_per_atom = energy_total / n_atoms if n_atoms > 0 else 0.0
        
        j_atoms = ase_to_jarvis(atoms, energy_per_atom)
        test_data.append({
            'atoms': j_atoms.to_dict(),
            'jid': f'test_{idx}',
            'formation_energy_peratom': energy_per_atom
        })
    
    logger.info("✅ Data loaded")
    
    return train_data, val_data, test_data


def create_alignn_model():
    """Create ALIGNN model architecture."""
    from alignn.models.alignn_atomwise import ALIGNNAtomwise
    
    config = {
        'name': 'alignn_atomwise',
        'atom_input_features': 92,
        'embedding_features': 256,
        'hidden_features': 256,
        'alignn_layers': 2,
        'gcn_layers': 2,
        'output_features': 1,
    }
    
    model = ALIGNNAtomwise(**config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_alignn_proper():
    """Train ALIGNN on our dataset."""
    logger.info("🚀 TRAINING ALIGNN (PROPER)")
    logger.info("="*80)
    
    # Load data
    train_data, val_data, test_data = load_custom_dataset()
    
    # Use ALIGNN's data loaders
    logger.info("\n🔧 Setting up ALIGNN data loaders...")
    
    try:
        from alignn.data import get_train_val_loaders
        
        # Use dataset_array parameter to feed our custom data
        # Skip test for now to avoid file issues
        # Use ALIGNN's built-in train/val split
        train_loader, val_loader, test_loader = get_train_val_loaders(
            dataset_array=train_data + val_data + test_data,
            target='formation_energy_peratom',
            batch_size=16,
            workers=4,
            cutoff=8.0,
            line_graph=True,
            output_dir='models/alignn_official',
            n_train=len(train_data),
            n_val=len(val_data),
            n_test=0,  # Skip test for now
        )
        
        logger.info("✅ Data loaders created")
        
        # Create model
        model = create_alignn_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup training
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training loop
        logger.info("\n🚀 Starting training...")
        
        best_val_loss = float('inf')
        for epoch in range(50):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Forward pass
                output = model(batch.to(device))
                target = batch['target'].to(device)
                
                # Loss
                loss = criterion(output, target)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    output = model(batch.to(device))
                    target = batch['target'].to(device)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1:3d} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, 'models/alignn_official/best_model.pt')
        
        logger.info("\n✅ Training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_alignn_proper()

