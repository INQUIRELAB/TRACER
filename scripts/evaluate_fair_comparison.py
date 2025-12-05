#!/usr/bin/env python3
"""
Fair Comparison Evaluation: ALIGNN vs Our GemNet Pipeline
Both models evaluated on PER-ATOM energy predictions for fair comparison.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import json
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from ase.io import read
from jarvis.core.atoms import Atoms as JAtoms
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Force RTX 4090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_test_data():
    """Load test dataset and convert to per-atom targets."""
    logger.info("📥 Loading test dataset...")
    
    test_file = Path('data/preprocessed_full_unified/test_data.json')
    with open(test_file, 'r') as f:
        test_samples = json.load(f)
    
    logger.info(f"   Loaded {len(test_samples)} test samples")
    
    # CRITICAL FIX: Test data is already in per-atom format (formation_energy_per_atom)
    # Do NOT divide by n_atoms (this was the bug!)
    test_data_per_atom = []
    for sample in test_samples:
        pos = np.array(sample['positions'])
        elements = sample['atomic_numbers']
        n_atoms = len(pos)
        
        if 'formation_energy_per_atom' in sample:
            per_atom_energy = sample['formation_energy_per_atom']
        else:
            energy = sample.get('energy', sample.get('energy_target', 0.0))
            # Heuristic: if energy is large (>50 eV), assume total and convert
            if abs(energy) > 50 and n_atoms > 0:
                per_atom_energy = energy / n_atoms
            else:
                # Already per-atom (typical range: -5 to 2 eV/atom)
                per_atom_energy = energy
        
        test_data_per_atom.append({
            'positions': pos,
            'atomic_numbers': elements,
            'energy_per_atom': per_atom_energy,
            'n_atoms': n_atoms
        })
    
    return test_data_per_atom


def evaluate_gemnet_per_atom(model_path: str, test_data):
    """Evaluate our GemNet model on per-atom predictions."""
    logger.info("🔧 Loading GemNet per-atom model...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    norm_stats = checkpoint.get('normalization', {})
    model_config = checkpoint.get('model_config', {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = GemNetWrapper(
        num_atoms=model_config.get('num_atoms', 120),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_filters=model_config.get('num_filters', 256),
        num_interactions=model_config.get('num_interactions', 6),
        cutoff=model_config.get('cutoff', 10.0),
        readout="sum",
        mean=norm_stats.get('mean'),
        std=norm_stats.get('std'),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"   Model loaded from {model_path}")
    logger.info(f"   Device: {device}")
    
    # Convert test data to PyG format
    def sample_to_pyg(sample):
        atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
        positions = torch.tensor(sample['positions'], dtype=torch.float32)
        
        from torch_geometric.nn import radius_graph
        edge_index = radius_graph(positions, r=10.0, max_num_neighbors=32)
        
        data = Data(
            atomic_numbers=atomic_numbers,
            pos=positions,
            edge_index=edge_index,
            batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
            energy_target=torch.tensor([sample['energy_per_atom']], dtype=torch.float32),
            n_atoms=torch.tensor([sample['n_atoms']], dtype=torch.long)
        )
        return data
    
    test_pyg = [sample_to_pyg(s) for s in test_data]
    test_loader = PyGDataLoader(test_pyg, batch_size=16, shuffle=False)
    
    # Evaluate
    logger.info("🧪 Evaluating GemNet on test set (per-atom)...")
    predictions = []
    targets = []
    errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="GemNet"):
            batch = batch.to(device)
            energies_total, _, _ = model(batch, compute_forces=False)
            
            # Convert to per-atom
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_total.dtype, device=energies_total.device)
            
            energies_per_atom = energies_total / n_atoms_per_graph
            
            # Denormalize if needed
            if norm_stats.get('mean') is not None and norm_stats.get('std') is not None:
                energies_per_atom = energies_per_atom * norm_stats['std'] + norm_stats['mean']
            
            preds = energies_per_atom.cpu().numpy()
            targs = batch.energy_target.cpu().numpy()
            
            predictions.extend(preds)
            targets.extend(targs)
            errors.extend(np.abs(preds - targs))
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    # R²
    ss_res = np.sum((targets - predictions)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    logger.info(f"   MAE (per-atom): {mae:.6f} eV/atom")
    logger.info(f"   RMSE (per-atom): {rmse:.6f} eV/atom")
    logger.info(f"   R²: {r2:.6f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def evaluate_alignn(model_path: str, test_data):
    """Evaluate ALIGNN on per-atom predictions."""
    logger.info("🔧 Loading ALIGNN model...")
    
    from ase.data import chemical_symbols
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ALIGNN model
    checkpoint = torch.load(model_path, map_location='cpu')
    
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"   Model loaded from {model_path}")
    logger.info(f"   Device: {device}")
    
    # Convert test data to JARVIS Atoms format
    def sample_to_jarvis(sample):
        pos = np.array(sample['positions'])
        elements = sample['atomic_numbers']
        
        max_dist = np.max(pos) - np.min(pos) if len(pos) > 0 else 10.0
        cell = np.eye(3) * (max_dist + 10.0)
        
        element_list = [chemical_symbols[z] for z in elements]
        
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=element_list,
            coords=pos.tolist()
        )
        
        return j_atoms, sample['energy_per_atom']
    
    test_jarvis = [sample_to_jarvis(s) for s in test_data]
    
    # Evaluate
    logger.info("🧪 Evaluating ALIGNN on test set (per-atom)...")
    predictions = []
    targets = []
    errors = []
    
    model.eval()
    for j_atoms, target_per_atom in tqdm(test_jarvis, desc="ALIGNN"):
        try:
            g, lg = Graph.atom_dgl_multigraph(j_atoms)
            lat = torch.tensor(j_atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
            
            g = g.to(device)
            lat = lat.to(device)
            
            output_dict = model((g, lat))
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            pred_energy = output.squeeze().cpu().item()
            
            predictions.append(pred_energy)
            targets.append(target_per_atom)
            errors.append(abs(pred_energy - target_per_atom))
        except Exception as e:
            logger.warning(f"Failed to predict: {e}")
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    # R²
    ss_res = np.sum((targets - predictions)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    logger.info(f"   MAE (per-atom): {mae:.6f} eV/atom")
    logger.info(f"   RMSE (per-atom): {rmse:.6f} eV/atom")
    logger.info(f"   R²: {r2:.6f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }


def print_comparison_table(results_gemnet, results_alignn):
    """Print fair comparison table."""
    print("\n" + "="*80)
    print("  FAIR COMPARISON: ALIGNN vs Our GemNet Pipeline")
    print("  Both models optimized for PER-ATOM energy predictions")
    print("="*80)
    
    print(f"\n📊 Results on {len(results_gemnet['targets'])} test samples:\n")
    
    print(f"{'Metric':<25} {'ALIGNN':<20} {'Our Pipeline':<20} {'Winner':<15}")
    print("-" * 80)
    
    # MAE comparison
    mae_alignn = results_alignn['mae']
    mae_gemnet = results_gemnet['mae']
    mae_winner = "ALIGNN ✓" if mae_alignn < mae_gemnet else "GemNet ✓" if mae_gemnet < mae_alignn else "Tie"
    print(f"{'MAE (eV/atom)':<25} {mae_alignn:<20.6f} {mae_gemnet:<20.6f} {mae_winner:<15}")
    
    # RMSE comparison
    rmse_alignn = results_alignn['rmse']
    rmse_gemnet = results_gemnet['rmse']
    rmse_winner = "ALIGNN ✓" if rmse_alignn < rmse_gemnet else "GemNet ✓" if rmse_gemnet < rmse_alignn else "Tie"
    print(f"{'RMSE (eV/atom)':<25} {rmse_alignn:<20.6f} {rmse_gemnet:<20.6f} {rmse_winner:<15}")
    
    # R² comparison
    r2_alignn = results_alignn['r2']
    r2_gemnet = results_gemnet['r2']
    r2_winner = "GemNet ✓" if r2_gemnet > r2_alignn else "ALIGNN ✓" if r2_alignn > r2_gemnet else "Tie"
    print(f"{'R² Score':<25} {r2_alignn:<20.6f} {r2_gemnet:<20.6f} {r2_winner:<15}")
    
    print("\n" + "="*80)
    
    # Summary
    print("\n📈 Summary:")
    if mae_gemnet < mae_alignn:
        print(f"   ✅ Our GemNet has better MAE ({mae_gemnet:.6f} vs {mae_alignn:.6f} eV/atom)")
    else:
        print(f"   ⚠️  ALIGNN has better MAE ({mae_alignn:.6f} vs {mae_gemnet:.6f} eV/atom)")
    
    if r2_gemnet > r2_alignn:
        print(f"   ✅ Our GemNet has better correlation (R²={r2_gemnet:.6f} vs {r2_alignn:.6f})")
    else:
        print(f"   ⚠️  ALIGNN has better correlation (R²={r2_alignn:.6f} vs {r2_gemnet:.6f})")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    logger.info("🚀 FAIR COMPARISON EVALUATION")
    logger.info("="*80)
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate both models
    logger.info("\n" + "="*80)
    results_gemnet = evaluate_gemnet_per_atom(
        'models/gemnet_per_atom/best_model.pt',
        test_data
    )
    
    logger.info("\n" + "="*80)
    results_alignn = evaluate_alignn(
        'models/alignn_official/best_model.pt',
        test_data
    )
    
    # Print comparison
    print_comparison_table(results_gemnet, results_alignn)


if __name__ == "__main__":
    main()

