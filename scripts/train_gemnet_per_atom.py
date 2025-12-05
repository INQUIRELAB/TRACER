#!/usr/bin/env python3
"""
Train GemNet for per-atom energy predictions (for fair comparison with ALIGNN)
This version trains the model to predict per-atom energy instead of total energy.
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


def custom_collate_fn(batch_list):
    """Custom collate function that properly handles domain_id for FiLM."""
    # Create batch
    batch = Batch.from_data_list(batch_list)
    
    # Extract domain_id per graph (stored as per-node, but same for all nodes in graph)
    batch_domain_ids = []
    for i, data in enumerate(batch_list):
        if hasattr(data, 'domain_id') and data.domain_id is not None:
            domain_id = data.domain_id[0].item() if len(data.domain_id) > 0 else 0
        else:
            domain_id = 0
        batch_domain_ids.append(domain_id)
    
    # Store as batch attribute (list, not tensor, to avoid batching issues)
    batch.graph_domain_ids = batch_domain_ids
    
    return batch

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
    
    # CRITICAL FIX: Check if energy is already per-atom or total
    # JARVIS-DFT provides formation_energy_per_atom directly
    import numpy as np
    
    all_per_atom_energies = []
    for s in train_data:
        # Prefer explicit formation_energy_per_atom field
        if 'formation_energy_per_atom' in s:
            per_atom_energy = s['formation_energy_per_atom']
        elif 'energy' in s:
            # Check if energy is reasonable for per-atom (typical range: -5 to 2 eV/atom)
            # or total (typical range: -500 to 500 eV for ~100 atoms)
            energy = s['energy']
            n_atoms = len(s.get('positions', []))
            
            # Heuristic: if energy magnitude is large (>50 eV), assume it's total
            if abs(energy) > 50 and n_atoms > 0:
                per_atom_energy = energy / n_atoms
                logger.warning(f"Energy {energy:.2f} eV seems like total energy, converting to per-atom")
            else:
                # Assume it's already per-atom
                per_atom_energy = energy
        else:
            per_atom_energy = s.get('energy_target', 0.0)
        
        all_per_atom_energies.append(per_atom_energy)
    
    if len(all_per_atom_energies) > 0:
        norm_mean = np.mean(all_per_atom_energies)
        norm_std = np.std(all_per_atom_energies)
        
        norm_stats = {'mean': norm_mean, 'std': norm_std}
        
        logger.info(f"   Computed per-atom normalization: mean={norm_mean:.6f}, std={norm_std:.6f}")
    else:
        norm_stats = {'mean': 0.0, 'std': 1.0}
    
    logger.info(f"✅ Loaded:")
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Validation: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    logger.info(f"   Normalization (per-atom): mean={norm_stats.get('mean', 0):.6f}, std={norm_stats.get('std', 1):.6f}")
    
    return train_data, val_data, test_data, norm_stats


def get_domain_id(domain_str: str) -> int:
    """Map domain string to domain ID for FiLM adaptation.
    
    Mapping:
        0: jarvis_dft
        1: jarvis_elastic
        2: oc20_s2ef
        3: oc22_s2ef
        4: ani1x
    """
    domain_lower = domain_str.lower()
    
    if 'jarvis' in domain_lower:
        if 'elastic' in domain_lower:
            return 1  # JARVIS-Elastic
        else:
            return 0  # JARVIS-DFT
    elif 'oc20' in domain_lower:
        return 2  # OC20-S2EF
    elif 'oc22' in domain_lower:
        return 3  # OC22-S2EF
    elif 'ani' in domain_lower:
        return 4  # ANI1x
    else:
        # Default to JARVIS-DFT
        return 0


def sample_to_pyg_data(sample, norm_mean=None, norm_std=None):
    """Convert sample dict to PyTorch Geometric Data object.
    
    CRITICAL: Converts total energy to per-atom energy for fair comparison with ALIGNN.
    Also extracts domain_id for FiLM adaptation.
    """
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Create simple edge connectivity based on distance
    # Use fallback method that doesn't require torch-cluster
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # CRITICAL FIX: Use formation_energy_per_atom directly (do not divide by n_atoms)
    # Check if we have explicit per-atom field
    if 'formation_energy_per_atom' in sample:
        per_atom_energy = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', sample.get('energy_target', 0.0))
        n_atoms = len(positions)
        # Heuristic: if energy is large, assume total; otherwise assume per-atom
        if abs(energy) > 50 and n_atoms > 0:
            per_atom_energy = energy / n_atoms  # Convert total to per-atom
        else:
            per_atom_energy = energy  # Already per-atom
    
    # Normalize if stats provided
    if norm_mean is not None and norm_std is not None:
        per_atom_energy = (per_atom_energy - norm_mean) / norm_std
    
    # Extract domain_id for FiLM adaptation
    domain_str = sample.get('domain', 'jarvis_dft')
    domain_id = get_domain_id(domain_str)
    
    # Create Data object with per-atom energy target and domain_id
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        energy_target=torch.tensor([per_atom_energy], dtype=torch.float32),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),  # Store for later conversion
        domain_id=torch.tensor([domain_id], dtype=torch.long)  # For FiLM adaptation
    )
    
    return data


def train_gemnet_per_atom(
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    output_dir: str = "models/gemnet_per_atom"
):
    """Train GemNet for per-atom energy predictions (for fair comparison with ALIGNN)."""
    
    logger.info("🚀 TRAINING GEMNET FOR PER-ATOM ENERGY (Fair Comparison with ALIGNN)")
    logger.info("="*80)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"   Device: {device}")
    
    # Force RTX 4090 for compatibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    train_data, val_data, test_data, norm_stats = load_preprocessed_data()
    
    # 2. Convert to PyG Data objects with per-atom targets
    logger.info("\n🔄 Converting to PyG format (per-atom targets)...")
    train_pyg = [sample_to_pyg_data(s, norm_stats.get('mean'), norm_stats.get('std')) for s in train_data]
    val_pyg = [sample_to_pyg_data(s, norm_stats.get('mean'), norm_stats.get('std')) for s in val_data]
    test_pyg = [sample_to_pyg_data(s, norm_stats.get('mean'), norm_stats.get('std')) for s in test_data]
    
    # 3. Create dataloaders with custom collate function for domain_id
    train_loader = PyGDataLoader(train_pyg, batch_size=min(batch_size, len(train_pyg)), shuffle=True, collate_fn=custom_collate_fn)
    val_loader = PyGDataLoader(val_pyg, batch_size=min(batch_size, len(val_pyg)), shuffle=False, collate_fn=custom_collate_fn)
    test_loader = PyGDataLoader(test_pyg, batch_size=min(batch_size, len(test_pyg)), shuffle=False, collate_fn=custom_collate_fn)
    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    # 4. Create model
    logger.info("\n🔧 Creating GemNet model (per-atom output)...")
    
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
        use_film=True,  # Enable FiLM adaptation
        num_domains=5,  # JARVIS-DFT, JARVIS-Elastic, OC20, OC22, ANI1x
        film_dim=16     # Domain embedding dimension
    ).to(device)
    
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   ⚠️ Model will predict PER-ATOM energy (same as ALIGNN)")
    logger.info(f"   ✅ FiLM adaptation: ENABLED (domain-aware predictions)")
    
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
            
            # Extract domain_id from batch for FiLM (using custom collate function)
            if hasattr(batch, 'graph_domain_ids'):
                batch_domain_ids = batch.graph_domain_ids
            else:
                # Fallback: default to JARVIS-DFT
                batch_size = batch.batch.max().item() + 1
                batch_domain_ids = [0] * batch_size
            
            domain_id = torch.tensor(batch_domain_ids, dtype=torch.long, device=device)
            
            # Forward with domain_id for FiLM adaptation
            energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
            
            # Convert total energy to per-atom energy (model outputs total via sum readout)
            # Get number of atoms per graph in batch using batch indices
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                # Single graph
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_total.dtype, device=energies_total.device)
            
            energies_per_atom = energies_total / n_atoms_per_graph
            
            # Loss - comparing per-atom predictions to per-atom targets
            loss = criterion(energies_per_atom, batch.energy_target)
            
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
                
                # Extract domain_id for FiLM (using custom collate function)
                if hasattr(batch, 'graph_domain_ids'):
                    batch_domain_ids = batch.graph_domain_ids
                else:
                    batch_size = batch.batch.max().item() + 1
                    batch_domain_ids = [0] * batch_size
                domain_id = torch.tensor(batch_domain_ids, dtype=torch.long, device=device)
                
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
                
                # Convert to per-atom
                if hasattr(batch, 'batch'):
                    n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                    n_atoms_per_graph = n_atoms_per_graph.float()
                else:
                    n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                      dtype=energies_total.dtype, device=energies_total.device)
                
                energies_per_atom = energies_total / n_atoms_per_graph
                loss = criterion(energies_per_atom, batch.energy_target)
                val_loss += loss.item() * batch.num_graphs
                val_count += batch.num_graphs
        
        avg_val_loss = val_loss / val_count if val_count > 0 else 0.0
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
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
                'target_type': 'per_atom_energy',  # Mark as per-atom model
                'model_config': {
                    'num_atoms': model.num_atoms,
                    'hidden_dim': model.hidden_dim,
                    'num_interactions': len(model.blocks),
                    'cutoff': model.cutoff,
                    'mean': model.mean,
                    'std': model.std,
                    'use_film': model.use_film,  # FiLM adaptation flag
                    'num_domains': model.num_domains,
                    'film_dim': 16,  # film_dim from model creation
                }
            }, output_path / 'best_model.pt')
            logger.info(f"   ✓ Saved best model (val_loss: {best_val_loss:.6f})")
    
    # 7. Test evaluation
    logger.info("\n📊 Evaluating on test set (per-atom)...")
    model.eval()
    test_loss = 0.0
    test_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Extract domain_id for FiLM
            batch_size = batch.batch.max().item() + 1
            batch_domain_ids = []
            for b in range(batch_size):
                mask = (batch.batch == b)
                domain_idx = batch.domain_id[mask][0].item() if hasattr(batch, 'domain_id') else 0
                batch_domain_ids.append(domain_idx)
            domain_id = torch.tensor(batch_domain_ids, dtype=torch.long, device=device)
            
            energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
            
            # Convert to per-atom
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_total.dtype, device=energies_total.device)
            
            energies_per_atom = energies_total / n_atoms_per_graph
            loss = criterion(energies_per_atom, batch.energy_target)
            test_loss += loss.item() * batch.num_graphs
            test_count += batch.num_graphs
    
    avg_test_loss = test_loss / test_count if test_count > 0 else 0.0
    logger.info(f"   Test Loss (per-atom): {avg_test_loss:.6f} eV/atom")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.6f} eV/atom")
    logger.info(f"   Test loss: {avg_test_loss:.6f} eV/atom")
    logger.info(f"   Model saved to: {output_path / 'best_model.pt'}")
    logger.info(f"   ⚠️ This model predicts PER-ATOM energy (for fair comparison with ALIGNN)")
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
        output_dir: str = typer.Option("models/gemnet_per_atom", help="Output directory")
    ):
        train_gemnet_per_atom(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir
        )
    
    app()

