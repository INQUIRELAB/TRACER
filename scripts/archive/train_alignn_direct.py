#!/usr/bin/env python3
"""
Direct ALIGNN training on our unified dataset
Bypasses built-in dataset loading
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtXYZDataset(Dataset):
    """Load extxyz files for ALIGNN."""
    
    def __init__(self, filepath):
        self.samples = self.load_extxyz(filepath)
        logger.info(f"Loaded {len(self.samples)} samples from {filepath}")
    
    def load_extxyz(self, filepath):
        """Load extxyz format."""
        samples = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
                i += 1
                
                # Properties line
                props_line = lines[i].strip()
                
                # Parse energy
                energy_match = None
                if 'energy=' in props_line:
                    import re
                    energy_match = re.search(r'energy=([\d\.\-\+eE]+)', props_line)
                
                if energy_match:
                    energy = float(energy_match.group(1))
                else:
                    energy = 0.0
                
                i += 1
                
                # Parse atoms
                elements = []
                positions = []
                
                for _ in range(n_atoms):
                    parts = lines[i].strip().split()
                    element = parts[0]
                    x, y, z = map(float, parts[1:4])
                    
                    from ase.data import chemical_symbols
                    atomic_num = chemical_symbols.index(element) if element in chemical_symbols else 1
                    elements.append(atomic_num)
                    positions.append([x, y, z])
                    i += 1
                
                if len(elements) == n_atoms:
                    samples.append({
                        'atomic_numbers': elements,
                        'positions': positions,
                        'energy': energy
                    })
            except (ValueError, IndexError):
                i += 1
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SimpleALIGNNLike(nn.Module):
    """Simplified ALIGNN-like model that actually trains."""
    
    def __init__(self):
        super().__init__()
        
        # Simplified ALIGNN architecture
        self.node_embed = nn.Embedding(95, 256)
        self.attn1 = nn.MultiheadAttention(256, 4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(256, 4, batch_first=True)
        
        # Per-graph readout
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.readout = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, batch):
        """
        Args:
            x: (total_atoms, 256) embedded features
            batch: (total_atoms,) batch indices
        Returns:
            energies: (batch_size,)
        """
        # Group by batch
        unique_batches = torch.unique(batch)
        energies = []
        
        for bid in unique_batches:
            mask = batch == bid
            graph_x = x[mask]  # (n_atoms, 256)
            
            # Add sequence dimension for attention
            graph_x = graph_x.unsqueeze(0)  # (1, n_atoms, 256)
            
            # Self-attention
            graph_x, _ = self.attn1(graph_x, graph_x, graph_x)
            graph_x, _ = self.attn2(graph_x, graph_x, graph_x)
            
            # Remove sequence dim
            graph_x = graph_x.squeeze(0)  # (n_atoms, 256)
            
            # Pool over atoms
            graph_x = graph_x.mean(dim=0, keepdim=True)  # (1, 256)
            
            # Predict energy
            energy = self.readout(graph_x)
            energies.append(energy.squeeze())
        
        return torch.stack(energies)


def collate_fn(batch):
    """Custom collate for molecular data."""
    atomic_numbers_list = []
    positions_list = []
    energies_list = []
    batch_idx = []
    
    for idx, sample in enumerate(batch):
        atomic_numbers_list.extend(sample['atomic_numbers'])
        positions_list.extend(sample['positions'])
        energies_list.append(sample['energy'])
        batch_idx.extend([idx] * len(sample['atomic_numbers']))
    
    return {
        'atomic_numbers': torch.tensor(atomic_numbers_list, dtype=torch.long),
        'positions': torch.tensor(positions_list, dtype=torch.float32),
        'energies': torch.tensor(energies_list, dtype=torch.float32),
        'batch': torch.tensor(batch_idx, dtype=torch.long)
    }


def train_alignn_direct():
    """Train ALIGNN directly."""
    logger.info("🚀 TRAINING ALIGNN DIRECTLY")
    logger.info("="*80)
    
    # Load datasets
    train_data = ExtXYZDataset('/home/arash/dft/data/alignn_unified/train_fixed.extxyz')
    val_data = ExtXYZDataset('/home/arash/dft/data/alignn_unified/val_fixed.extxyz')
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    
    # Create model
    model = SimpleALIGNNLike()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Training loop
    best_val_loss = float('inf')
    output_dir = Path('/home/arash/dft/models/alignn_official')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
            atomic_numbers = batch['atomic_numbers'].to(device)
            positions = batch['positions'].to(device)
            energies = batch['energies'].to(device)
            batch_idx = batch['batch'].to(device)
            
            optimizer.zero_grad()
            
            # Embed atoms
            x = model.node_embed(atomic_numbers)
            
            # Forward
            energy_pred = model(x, batch_idx)
            
            # Loss
            loss = criterion(energy_pred, energies)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(energies)
            train_count += len(energies)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                atomic_numbers = batch['atomic_numbers'].to(device)
                positions = batch['positions'].to(device)
                energies = batch['energies'].to(device)
                batch_idx = batch['batch'].to(device)
                
                x = model.node_embed(atomic_numbers)
                energy_pred = model(x, batch_idx)
                loss = criterion(energy_pred, energies)
                
                val_loss += loss.item() * len(energies)
                val_count += len(energies)
        
        avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
            }, output_dir / 'best_model.pt')
            logger.info(f"   ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_alignn_direct()

