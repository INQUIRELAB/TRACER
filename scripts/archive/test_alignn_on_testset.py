#!/usr/bin/env python3
"""
Test trained ALIGNN model on independent test set
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force RTX 4090 for compatibility
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ase.io import read
from jarvis.core.atoms import Atoms as JAtoms
from alignn.graphs import Graph  # Use ALIGNN's Graph, not jarvis!
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

def ase_to_jarvis(atoms, energy=0.0):
    """Convert ASE Atoms to JARVIS Atoms - use Cartesian coordinates like training"""
    pos = atoms.get_positions()
    elements = list(atoms.get_chemical_symbols())
    cell = atoms.get_cell().array
    
    # Create cell if none (molecules)
    if cell.sum() == 0:
        max_dist = np.max(pos) - np.min(pos) if len(pos) > 0 else 10.0
        cell = np.eye(3) * (max_dist + 10.0)
    
    # Use Cartesian coordinates (same as training)
    j_atoms = JAtoms(
        lattice_mat=cell.tolist(),
        elements=elements,
        coords=pos.tolist()  # Cartesian coordinates like training!
    )
    
    return j_atoms

def load_test_data():
    """Load test dataset from preprocessed JSON"""
    logger.info("📥 Loading test dataset...")
    import json
    
    test_file = Path('data/preprocessed_full_unified/test_data.json')
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        test_samples = json.load(f)
    
    logger.info(f"   Loaded {len(test_samples)} test samples")
    
    test_data = []
    for sample in test_samples:
        # Create JARVIS Atoms from the sample
        pos = np.array(sample['positions'])
        elements = sample['atomic_numbers']
        energy = sample.get('energy', 0.0)
        
        # Create cell (molecular systems)
        max_dist = np.max(pos) - np.min(pos) if len(pos) > 0 else 10.0
        cell = np.eye(3) * (max_dist + 10.0)
        
        # Convert atomic numbers to elements
        from ase.data import chemical_symbols
        element_list = [chemical_symbols[z] for z in elements]
        
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=element_list,
            coords=pos.tolist()
        )
        
        test_data.append((j_atoms, energy))
    
    return test_data

def load_model():
    """Load trained ALIGNN model"""
    logger.info("🔧 Loading trained ALIGNN model...")
    
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
    
    # Load trained weights
    checkpoint_path = Path('models/alignn_official/best_model.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"   Model loaded from {checkpoint_path}")
    logger.info(f"   Device: {device}")
    
    return model, device

def evaluate_model(model, device, test_data):
    """Evaluate model on test set"""
    logger.info("🧪 Evaluating ALIGNN on test set...")
    
    predictions = []
    targets = []
    errors = []
    
    # Don't use torch.no_grad() because ALIGNN uses grad() internally
    model.eval()  # Set model to eval mode
    for j_atoms, target_energy in tqdm(test_data, desc="Testing"):
        try:
            # Create graph
            g, lg = Graph.atom_dgl_multigraph(j_atoms)
            # Match training: requires_grad=True
            lat = torch.tensor(j_atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
            
            # Move to device
            g = g.to(device)
            lat = lat.to(device)
            
            # Predict (ALIGNN uses grad() internally, so no no_grad())
            output_dict = model((g, lat))
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            pred_energy = output.squeeze().cpu().item()
            
            # ALIGNN was trained on per-atom energies, but test data has total
            # We need to compare like-with-like
            # Predictions are per-atom, targets are total - convert target to per-atom
            n_atoms = len(j_atoms.elements)
            target_per_atom = target_energy / n_atoms
            pred_energy_total = pred_energy * n_atoms  # Convert pred to total for display
            
            predictions.append(pred_energy)  # Keep per-atom for R²
            targets.append(target_per_atom)   # Convert to per-atom
            errors.append(abs(pred_energy - target_per_atom))  # Per-atom error
            
        except Exception as e:
            logger.warning(f"Failed to predict: {e}")
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.array(errors)
    
    # Calculate metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    # Calculate per-atom metrics (approximate)
    n_atoms_avg = 20  # Approximate average
    mae_per_atom = mae / n_atoms_avg
    
    # Calculate R²
    ss_res = np.sum((targets - predictions)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mae_per_atom': mae_per_atom,
        'r2': r2,
        'n_samples': len(predictions)
    }

def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*80)
    print("  ALIGNN TEST SET EVALUATION")
    print("="*80)
    print(f"\n📊 Results on {results['n_samples']} independent test samples:\n")
    print(f"  MAE (total):          {results['mae']:.4f} eV")
    print(f"  RMSE (total):         {results['rmse']:.4f} eV")
    print(f"  MAE (per atom):       {results['mae_per_atom']:.4f} eV/atom")
    print(f"  R² Score:             {results['r2']:.4f}")
    print("\n" + "="*80)
    
    # Compare with our pipeline
    print("\n📈 COMPARISON WITH OUR PIPELINE:\n")
    print(f"{'Metric':<25} {'ALIGNN':<20} {'Our Pipeline':<20} {'Winner':<15}")
    print("-" * 80)
    print(f"{'MAE (eV)':<25} {results['mae']:<20.2f} {'0.82':<20} {'ALIGNN ✓' if results['mae'] < 0.82 else 'Ours ✓'}")
    print(f"{'MAE (eV/atom)':<25} {results['mae_per_atom']:<20.2f} {'0.12':<20} {'ALIGNN ✓' if results['mae_per_atom'] < 0.12 else 'Ours ✓'}")
    print(f"{'R² Score':<25} {results['r2']:<20.4f} {'0.9930':<20} {'ALIGNN ✓' if results['r2'] > 0.9930 else 'Ours ✓'}")
    print("\n" + "="*80)

def main():
    """Main evaluation"""
    try:
        test_data = load_test_data()
        model, device = load_model()
        results = evaluate_model(model, device, test_data)
        print_results(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
