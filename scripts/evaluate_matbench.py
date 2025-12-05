#!/usr/bin/env python3
"""
Evaluate GemNet model on Matbench tasks.

This script loads Matbench datasets and evaluates the trained GemNet model
on standard materials property prediction tasks.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def structure_to_pyg_data(structure, target=None, norm_mean=None, norm_std=None):
    """
    Convert pymatgen Structure to PyTorch Geometric Data object.
    
    Args:
        structure: pymatgen Structure object
        target: Target property value (e.g., formation energy per atom)
        norm_mean: Normalization mean
        norm_std: Normalization std
    
    Returns:
        PyG Data object
    """
    from pymatgen.core import Structure
    
    # Get atomic numbers and positions
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
    
    # Create Data object
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=energy_target,
        energy_target_original=energy_target_original,
        n_atoms=torch.tensor([len(atomic_numbers)], dtype=torch.long)
    )
    
    # Add cell if periodic
    if structure.is_ordered and hasattr(structure, 'lattice'):
        cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
        data.cell = cell
    
    return data


def load_trained_model(model_path: str, device: torch.device):
    """Load trained GemNet model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model_config = checkpoint.get('model_config', {})
    norm_stats = checkpoint.get('normalization', {})
    
    use_film = model_config.get('use_film', False)
    num_domains = model_config.get('num_domains', 0)
    film_dim = model_config.get('film_dim', 16)
    
    model = GemNetWrapper(
        num_atoms=model_config.get('num_atoms', 120),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_filters=model_config.get('num_filters', 256),
        num_interactions=model_config.get('num_interactions', 6),
        cutoff=model_config.get('cutoff', 10.0),
        readout="sum",
        mean=norm_stats.get('mean'),
        std=norm_stats.get('std'),
        use_film=use_film,
        num_domains=num_domains,
        film_dim=film_dim
    ).to(device)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    return model, norm_stats


def evaluate_on_matbench_task(model, norm_stats, task_name, train_df, test_df, 
                              structure_col='structure', target_col='formation_energy_per_atom',
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                              use_matbench_normalization=True):
    """
    Evaluate model on a Matbench task.
    
    Args:
        model: Trained GemNet model
        norm_stats: Normalization statistics (from training, may be overridden)
        task_name: Name of the Matbench task
        train_df: Training DataFrame with structures and targets
        test_df: Test DataFrame with structures and targets
        structure_col: Column name for structure objects
        target_col: Column name for target property
        device: Device to run on
        use_matbench_normalization: If True, compute normalization from Matbench train data
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating on {task_name}")
    logger.info(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    # Compute normalization from Matbench training data (CRITICAL)
    if use_matbench_normalization:
        import numpy as np
        matbench_targets = train_df[target_col].values
        matbench_mean = np.mean(matbench_targets)
        matbench_std = np.std(matbench_targets)
        logger.info(f"  Using Matbench normalization: mean={matbench_mean:.6f}, std={matbench_std:.6f}")
        norm_mean = matbench_mean
        norm_std = matbench_std
    else:
        norm_mean = norm_stats.get('mean', 0.0)
        norm_std = norm_stats.get('std', 1.0)
        logger.info(f"  Using model normalization: mean={norm_mean:.6f}, std={norm_std:.6f}")
    
    predictions = []
    targets = []
    failed = 0
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {task_name}"):
            try:
                structure = row[structure_col]
                target = row[target_col]
                
                # Convert structure to PyG Data
                # Note: We normalize using Matbench stats, but model was trained with different stats
                # This is a limitation - ideally model should be fine-tuned on Matbench
                data = structure_to_pyg_data(
                    structure, 
                    target=target,
                    norm_mean=norm_mean,
                    norm_std=norm_std
                )
                data = data.to(device)
                
                # Create batch
                batch = Batch.from_data_list([data])
                
                # Predict (no domain_id needed for baseline model)
                domain_id = torch.tensor([0], dtype=torch.long, device=device)
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
                
                # Model outputs total energy, convert to per-atom
                n_atoms = len(data.atomic_numbers)
                energy_per_atom_normalized = energies_total.item() / n_atoms
                
                # Denormalize
                energy_per_atom_pred = energy_per_atom_normalized * norm_std + norm_mean
                
                predictions.append(energy_per_atom_pred)
                targets.append(target)
                
            except Exception as e:
                logger.debug(f"Error on sample {idx}: {e}")
                failed += 1
                continue
    
    if len(predictions) == 0:
        logger.error(f"No successful predictions for {task_name}")
        return None
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    logger.info(f"  Results for {task_name}:")
    logger.info(f"    MAE: {mae:.6f}")
    logger.info(f"    RMSE: {rmse:.6f}")
    logger.info(f"    R²: {r2:.6f}")
    logger.info(f"    Failed: {failed}/{len(test_df)}")
    
    return {
        'task': task_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(predictions),
        'n_failed': failed,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }


def evaluate_matbench_perovskites(model, norm_stats, device):
    """Evaluate on Matbench Perovskites task."""
    try:
        from matminer.datasets import load_dataset
        
        logger.info("Loading Matbench Perovskites dataset...")
        df = load_dataset('matbench_perovskites')
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Matbench Perovskites structure: structure column + e_form (formation energy per atom)
        structure_col = 'structure'
        target_col = 'e_form'  # Formation energy per atom
        
        # Use first 80% as train (for normalization), last 20% as test
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return evaluate_on_matbench_task(
            model, norm_stats, 'matbench_perovskites',
            train_df, test_df, structure_col, target_col, device
        )
        
    except ImportError:
        logger.error("matminer not available. Install with: pip install matminer")
        return None
    except Exception as e:
        logger.error(f"Error loading Matbench Perovskites: {e}")
        return None


def evaluate_matbench_mp_is_metal(model, norm_stats, device):
    """Evaluate on Matbench MP Is Metal task (binary classification)."""
    try:
        from matminer.datasets import load_dataset
        
        logger.info("Loading Matbench MP Is Metal dataset...")
        df = load_dataset('matbench_mp_is_metal')
        
        logger.info(f"Loaded {len(df)} samples")
        
        # This is binary classification, not regression
        # Skip for now - would need classification head
        logger.warning("MP Is Metal is classification task - skipping")
        return None
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GemNet on Matbench tasks')
    parser.add_argument('--model-path', type=str, 
                       default='models/gemnet_baseline/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--tasks', type=str, nargs='+',
                       default=['perovskites'],
                       choices=['perovskites', 'mp_is_metal'],
                       help='Matbench tasks to evaluate')
    parser.add_argument('--output-dir', type=str,
                       default='artifacts/matbench',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, norm_stats = load_trained_model(args.model_path, device)
    logger.info(f"Model loaded (mean={norm_stats.get('mean', 0):.6f}, std={norm_stats.get('std', 1):.6f})")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Evaluate on requested tasks
    for task in args.tasks:
        logger.info("=" * 80)
        if task == 'perovskites':
            result = evaluate_matbench_perovskites(model, norm_stats, device)
        elif task == 'mp_is_metal':
            result = evaluate_matbench_mp_is_metal(model, norm_stats, device)
        else:
            logger.warning(f"Unknown task: {task}")
            continue
        
        if result:
            results[task] = result
    
    # Save results
    output_file = output_dir / 'matbench_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info(f"✅ Results saved to {output_file}")
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("  SUMMARY")
    logger.info("=" * 80)
    for task, result in results.items():
        logger.info(f"{task}:")
        logger.info(f"  MAE: {result['mae']:.6f}")
        logger.info(f"  RMSE: {result['rmse']:.6f}")
        logger.info(f"  R²: {result['r2']:.6f}")
        logger.info(f"  N samples: {result['n_samples']}")


if __name__ == '__main__':
    main()

