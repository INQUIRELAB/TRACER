#!/usr/bin/env python3
"""
Evaluate GemNet with FiLM adaptation on test dataset.
Reports per-atom energy metrics (MAE, RMSE, R²) overall and per domain.
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Force RTX 4090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_domain_id(domain_str: str) -> int:
    """Map domain string to domain ID for FiLM adaptation."""
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
        return 0  # Default to JARVIS-DFT


def custom_collate_fn(batch_list):
    """Custom collate function that properly handles domain_id for FiLM."""
    batch = Batch.from_data_list(batch_list)
    
    batch_domain_ids = []
    for i, data in enumerate(batch_list):
        if hasattr(data, 'domain_id') and data.domain_id is not None:
            domain_id = data.domain_id[0].item() if len(data.domain_id) > 0 else 0
        else:
            domain_id = 0
        batch_domain_ids.append(domain_id)
    
    batch.graph_domain_ids = batch_domain_ids
    return batch


def sample_to_pyg_data(sample, norm_mean=None, norm_std=None):
    """Convert sample dict to PyTorch Geometric Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Create edge connectivity (fallback method without torch-cluster)
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # CRITICAL FIX: Use formation_energy_per_atom directly
    # Verify we're using the correct target (formation energy per atom, not total)
    if 'formation_energy_per_atom' in sample:
        per_atom_energy = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', sample.get('energy_target', 0.0))
        n_atoms = len(positions)
        # Heuristic: if energy is large (>50 eV), assume total; otherwise per-atom
        if abs(energy) > 50 and n_atoms > 0:
            per_atom_energy = energy / n_atoms
        else:
            per_atom_energy = energy  # Already per-atom
    
    # Normalize if stats provided
    if norm_mean is not None and norm_std is not None:
        per_atom_energy_normalized = (per_atom_energy - norm_mean) / norm_std
    else:
        per_atom_energy_normalized = per_atom_energy
    
    # Extract domain_id for FiLM
    domain_str = sample.get('domain', 'jarvis_dft')
    domain_id = get_domain_id(domain_str)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=torch.tensor([per_atom_energy_normalized], dtype=torch.float32),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),
        domain_id=torch.tensor([domain_id], dtype=torch.long),
        energy_target_original=torch.tensor([per_atom_energy], dtype=torch.float32)  # Store original for metrics
    )
    
    return data


def compute_metrics(predictions, targets):
    """Compute MAE, RMSE, and R²."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_model(model_path: str, test_data_path: str, batch_size: int = 32, max_samples: int = None):
    """Evaluate GemNet model with FiLM on test dataset."""
    
    logger.info("=" * 80)
    logger.info("  GEMNET WITH FILM EVALUATION")
    logger.info("=" * 80)
    logger.info("")
    
    # 1. Load test data
    logger.info("📥 Loading test dataset...")
    with open(test_data_path, 'r') as f:
        test_samples = json.load(f)
    
    if max_samples:
        test_samples = test_samples[:max_samples]
        logger.info(f"   Limited to {max_samples} samples for evaluation")
    
    logger.info(f"   Loaded {len(test_samples)} test samples")
    
    # 2. Load model
    logger.info("🔧 Loading GemNet model with FiLM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"   Device: {device}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    norm_stats = checkpoint.get('normalization', {})
    model_config = checkpoint.get('model_config', {})
    
    # Determine if FiLM is enabled (check state dict or use default)
    has_film_params = any('film' in k.lower() or 'domain_embedding' in k.lower() 
                          for k in checkpoint['model_state_dict'].keys())
    use_film = model_config.get('use_film', has_film_params)
    
    logger.info(f"   FiLM enabled: {use_film}")
    logger.info(f"   Normalization: mean={norm_stats.get('mean', 0):.6f}, std={norm_stats.get('std', 1):.6f}")
    
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
        use_film=use_film,
        num_domains=model_config.get('num_domains', 5),
        film_dim=model_config.get('film_dim', 16)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"   Model loaded successfully")
    logger.info("")
    
    # 3. Convert test data to PyG format
    logger.info("🔄 Converting test data to PyG format...")
    test_pyg_data = []
    for sample in test_samples:
        try:
            pyg_data = sample_to_pyg_data(sample, 
                                         norm_mean=norm_stats.get('mean'),
                                         norm_std=norm_stats.get('std'))
            test_pyg_data.append(pyg_data)
        except Exception as e:
            logger.warning(f"   Skipped sample due to error: {e}")
            continue
    
    logger.info(f"   Converted {len(test_pyg_data)} samples")
    logger.info("")
    
    # 4. Create dataloader
    test_loader = PyGDataLoader(
        test_pyg_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # 5. Evaluate
    logger.info("🔮 Generating predictions...")
    all_predictions = []
    all_targets = []
    all_domains = []
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            # Extract domain IDs for FiLM
            if hasattr(batch, 'graph_domain_ids'):
                batch_domain_ids = batch.graph_domain_ids
            else:
                batch_size_val = batch.batch.max().item() + 1
                batch_domain_ids = [0] * batch_size_val
            
            domain_id_tensor = torch.tensor(batch_domain_ids, dtype=torch.long, device=device)
            
            # Forward pass with domain IDs
            energies_normalized, _, _ = model(batch, compute_forces=False, domain_id=domain_id_tensor)
            
            # CRITICAL: Check if model outputs total or per-atom energy
            # Based on training script, model outputs TOTAL energy, then we convert to per-atom
            # Get number of atoms per graph
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_normalized))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_normalized.dtype, 
                                                  device=energies_normalized.device)
            
            # Convert from total (normalized) to per-atom (normalized)
            energies_per_atom_normalized = energies_normalized / n_atoms_per_graph
            
            # Denormalize predictions (from normalized per-atom to actual per-atom)
            norm_mean = norm_stats.get('mean', 0.0)
            norm_std = norm_stats.get('std', 1.0)
            energies_per_atom_pred = energies_per_atom_normalized * norm_std + norm_mean
            
            # Get targets (already denormalized in energy_target_original)
            if hasattr(batch, 'energy_target_original'):
                targets = batch.energy_target_original.cpu().numpy()
            else:
                # Fallback: denormalize from normalized targets
                targets_normalized = batch.energy_target.cpu().numpy()
                targets = targets_normalized * norm_std + norm_mean
            
            # CRITICAL VERIFICATION: Check if targets are reasonable
            # Formation energy per atom: typically -5 to 2 eV/atom
            if len(targets) > 0:
                max_abs_target = np.max(np.abs(targets))
                if max_abs_target > 100:
                    logger.warning(f"⚠️  SUSPICIOUS: Target energy max(abs) = {max_abs_target:.2f} eV/atom")
                    logger.warning("   Expected formation energy per atom: typically -5 to 2 eV/atom")
                    logger.warning("   This suggests targets might be total energy instead of per-atom!")
                elif max_abs_target > 10:
                    logger.warning(f"⚠️  WARNING: Target energy max(abs) = {max_abs_target:.2f} eV/atom")
                    logger.warning("   This is higher than typical formation energies. Verify data format.")
            
            # Store predictions and targets
            predictions = energies_per_atom_pred.cpu().numpy()
            
            # Optional debug: Print first batch predictions (can be enabled with --debug flag)
            # Only print if explicitly requested to avoid cluttering output
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_domains.extend(batch_domain_ids)
            all_errors.extend(np.abs(predictions - targets).tolist())
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info("")
    
    # Overall metrics
    overall_metrics = compute_metrics(all_predictions, all_targets)
    logger.info("📊 Overall Metrics (Per-Atom Energy):")
    logger.info(f"   MAE:  {overall_metrics['mae']:.6f} eV/atom")
    logger.info(f"   RMSE: {overall_metrics['rmse']:.6f} eV/atom")
    logger.info(f"   R²:   {overall_metrics['r2']:.6f}")
    logger.info("")
    
    # Per-domain metrics
    domain_names = ['JARVIS-DFT', 'JARVIS-Elastic', 'OC20-S2EF', 'OC22-S2EF', 'ANI1x']
    domain_stats = defaultdict(lambda: {'predictions': [], 'targets': []})
    
    for pred, target, domain_id in zip(all_predictions, all_targets, all_domains):
        domain_stats[domain_id]['predictions'].append(pred)
        domain_stats[domain_id]['targets'].append(target)
    
    logger.info("📊 Per-Domain Metrics:")
    logger.info("")
    logger.info(f"{'Domain':<20} {'Samples':<10} {'MAE (eV/atom)':<15} {'RMSE (eV/atom)':<15} {'R²':<10}")
    logger.info("-" * 70)
    
    for domain_id in range(5):
        if domain_id in domain_stats:
            stats = domain_stats[domain_id]
            metrics = compute_metrics(stats['predictions'], stats['targets'])
            n_samples = len(stats['predictions'])
            domain_name = domain_names[domain_id] if domain_id < len(domain_names) else f'Domain-{domain_id}'
            logger.info(f"{domain_name:<20} {n_samples:<10} {metrics['mae']:<15.6f} {metrics['rmse']:<15.6f} {metrics['r2']:<10.6f}")
        else:
            domain_name = domain_names[domain_id] if domain_id < len(domain_names) else f'Domain-{domain_id}'
            logger.info(f"{domain_name:<20} {0:<10} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 80)
    
    return {
        'overall': overall_metrics,
        'per_domain': {domain_names[i] if i < len(domain_names) else f'Domain-{i}': 
                      compute_metrics(domain_stats[i]['predictions'], domain_stats[i]['targets'])
                      for i in range(5) if i in domain_stats}
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GemNet with FiLM')
    parser.add_argument('--model-path', type=str, 
                       default='models/gemnet_per_atom_film/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str,
                       default='data/preprocessed_full_unified/test_data.json',
                       help='Path to test data JSON file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Override device
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    results = evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

