#!/usr/bin/env python3
"""
Fixed ALIGNN Evaluation Script
Fixes:
1. Proper energy denormalization
2. Correct metric calculation (no double division)
3. Proper R² calculation
4. Energy scale consistency
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ase.data import chemical_symbols
from jarvis.core.atoms import Atoms as JAtoms
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

def load_model():
    """Load trained ALIGNN model with normalization stats."""
    logger.info("🔧 Loading trained ALIGNN model...")
    
    checkpoint_path = Path('models/alignn_fixed/best_model.pt')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get normalization stats
    energy_mean = checkpoint.get('energy_mean', 0.0)
    energy_std = checkpoint.get('energy_std', 1.0)
    
    logger.info(f"   Energy normalization: mean={energy_mean:.6f}, std={energy_std:.6f}")
    
    # Create model
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"   Model loaded from {checkpoint_path}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Best validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, device, energy_mean, energy_std


def load_test_data():
    """Load test dataset from preprocessed JSON."""
    logger.info("📥 Loading test dataset...")
    
    test_file = Path('data/preprocessed_full_unified/test_data.json')
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        test_samples = json.load(f)
    
    logger.info(f"   Loaded {len(test_samples)} test samples")
    
    test_data = []
    for sample in test_samples:
        atomic_numbers = sample['atomic_numbers']
        positions = np.array(sample['positions'])
        n_atoms = len(atomic_numbers)
        
        if n_atoms == 0:
            continue
        
        # CRITICAL FIX: Test data is already in per-atom format (formation_energy_per_atom)
        # Do NOT divide by n_atoms (this was the bug!)
        if 'formation_energy_per_atom' in sample:
            energy_per_atom = sample['formation_energy_per_atom']
        else:
            energy = sample.get('energy', sample.get('energy_target', 0.0))
            # Heuristic: if energy is large (>50 eV), assume total and convert
            if abs(energy) > 50 and n_atoms > 0:
                energy_per_atom = energy / n_atoms
            else:
                # Already per-atom (typical range: -5 to 2 eV/atom)
                energy_per_atom = energy
        
        # Create cell
        if 'cell' in sample and sample['cell']:
            cell = np.array(sample['cell'])
        else:
            max_dist = np.max(positions) - np.min(positions) if len(positions) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        # Convert atomic numbers to elements
        element_list = [chemical_symbols[z] for z in atomic_numbers]
        
        # Create JARVIS Atoms (Cartesian coordinates, matching training)
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=element_list,
            coords=positions.tolist()  # Cartesian, matching training
        )
        
        # Calculate total energy from per-atom
        energy_total = energy_per_atom * n_atoms
        
        test_data.append({
            'atoms': j_atoms,
            'energy_per_atom': energy_per_atom,
            'energy_total': energy_total,
            'n_atoms': n_atoms,
            'sample_id': sample.get('sample_id', 'unknown')
        })
    
    return test_data


def evaluate_model(model, device, test_data, energy_mean, energy_std):
    """Evaluate model on test set with proper denormalization."""
    logger.info("🧪 Evaluating ALIGNN on test set...")
    
    predictions_per_atom = []
    targets_per_atom = []
    predictions_total = []
    targets_total = []
    
    model.eval()
    
    # NOTE: Do NOT use torch.no_grad() here because ALIGNN performs internal grad() calls
    for sample in tqdm(test_data, desc="Evaluating"):
            try:
                j_atoms = sample['atoms']
                target_per_atom = sample['energy_per_atom']
                target_total = sample['energy_total']
                n_atoms = sample['n_atoms']
                
                # Create graph
                g, lg = Graph.atom_dgl_multigraph(j_atoms)
                
                # Lattice - requires_grad=True for ALIGNN internal grad() calls
                lat = torch.tensor(j_atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
                
                # Move to device
                g = g.to(device)
                lg = lg.to(device)
                lat = lat.to(device)
                
                # Predict (model outputs normalized per-atom energy)
                output_dict = model((g, lat))
                
                output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
                pred_normalized = output.squeeze().cpu().item()
                
                # DENORMALIZE: Convert from normalized space to actual per-atom energy
                pred_per_atom = pred_normalized * energy_std + energy_mean
                
                # Convert to total energy for comparison
                pred_total = pred_per_atom * n_atoms
                
                predictions_per_atom.append(pred_per_atom)
                targets_per_atom.append(target_per_atom)
                predictions_total.append(pred_total)
                targets_total.append(target_total)
                
            except Exception as e:
                logger.warning(f"Failed to predict {sample.get('sample_id', 'unknown')}: {e}")
                continue
    
    if len(predictions_per_atom) == 0:
        logger.error("No successful predictions!")
        return None
    
    predictions_per_atom = np.array(predictions_per_atom)
    targets_per_atom = np.array(targets_per_atom)
    predictions_total = np.array(predictions_total)
    targets_total = np.array(targets_total)
    
    # Calculate metrics (FIXED: No double division!)
    mae_per_atom = np.mean(np.abs(predictions_per_atom - targets_per_atom))
    rmse_per_atom = np.sqrt(np.mean((predictions_per_atom - targets_per_atom)**2))
    
    mae_total = np.mean(np.abs(predictions_total - targets_total))
    rmse_total = np.sqrt(np.mean((predictions_total - targets_total)**2))
    
    # Calculate R² (FIXED: Proper calculation)
    # For per-atom
    ss_res_per_atom = np.sum((targets_per_atom - predictions_per_atom)**2)
    ss_tot_per_atom = np.sum((targets_per_atom - np.mean(targets_per_atom))**2)
    r2_per_atom = 1 - (ss_res_per_atom / (ss_tot_per_atom + 1e-8))
    
    # For total
    ss_res_total = np.sum((targets_total - predictions_total)**2)
    ss_tot_total = np.sum((targets_total - np.mean(targets_total))**2)
    r2_total = 1 - (ss_res_total / (ss_tot_total + 1e-8))
    
    return {
        'mae_per_atom': mae_per_atom,
        'rmse_per_atom': rmse_per_atom,
        'r2_per_atom': r2_per_atom,
        'mae_total': mae_total,
        'rmse_total': rmse_total,
        'r2_total': r2_total,
        'n_samples': len(predictions_per_atom),
        'predictions_per_atom': predictions_per_atom,
        'targets_per_atom': targets_per_atom
    }


def print_results(results):
    """Print evaluation results."""
    if results is None:
        logger.error("No results to print!")
        return
    
    print("\n" + "="*80)
    print("  FIXED ALIGNN TEST SET EVALUATION")
    print("="*80)
    print(f"\n📊 Results on {results['n_samples']} independent test samples:\n")
    
    print(f"{'Metric':<30} {'Value':>15} {'Unit':<15}")
    print("-" * 60)
    print(f"{'MAE (per-atom)':<30} {results['mae_per_atom']:>15.6f} {'eV/atom':<15}")
    print(f"{'RMSE (per-atom)':<30} {results['rmse_per_atom']:>15.6f} {'eV/atom':<15}")
    print(f"{'R² Score (per-atom)':<30} {results['r2_per_atom']:>15.6f} {'':<15}")
    print()
    print(f"{'MAE (total)':<30} {results['mae_total']:>15.6f} {'eV':<15}")
    print(f"{'RMSE (total)':<30} {results['rmse_total']:>15.6f} {'eV':<15}")
    print(f"{'R² Score (total)':<30} {results['r2_total']:>15.6f} {'':<15}")
    
    print("\n" + "="*80)
    print("📈 COMPARISON WITH GEMNET:")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'ALIGNN (Fixed)':>20} {'GemNet':>20}")
    print("-" * 70)
    print(f"{'MAE (eV/atom)':<30} {results['mae_per_atom']:>20.6f} {'0.005977':>20}")
    print(f"{'RMSE (eV/atom)':<30} {results['rmse_per_atom']:>20.6f} {'0.020605':>20}")
    print(f"{'R² Score':<30} {results['r2_per_atom']:>20.6f} {'0.992627':>20}")
    
    # Analysis
    print("\n" + "="*80)
    print("📊 ANALYSIS:")
    print("="*80)
    
    if results['r2_per_atom'] > 0.9:
        print("✅ R² > 0.9: Excellent correlation!")
    elif results['r2_per_atom'] > 0.5:
        print("⚠️  R² > 0.5: Moderate correlation")
    else:
        print("❌ R² < 0.5: Poor correlation - still has issues")
    
    if results['r2_per_atom'] < 0:
        print("❌ Negative R²: Model worse than mean predictor!")
    
    if results['mae_per_atom'] < 0.01:
        print("✅ MAE < 0.01 eV/atom: Good accuracy")
    elif results['mae_per_atom'] < 0.1:
        print("⚠️  MAE < 0.1 eV/atom: Moderate accuracy")
    else:
        print("❌ MAE > 0.1 eV/atom: Poor accuracy")
    
    print("="*80)


def main():
    """Main evaluation."""
    try:
        model, device, energy_mean, energy_std = load_model()
        test_data = load_test_data()
        results = evaluate_model(model, device, test_data, energy_mean, energy_std)
        print_results(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

