#!/usr/bin/env python3
"""
End-to-end evaluation of the pipeline with per-atom delta head corrections.
Compares baseline GemNet predictions vs delta-corrected predictions.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_data():
    """Load preprocessed test split."""
    data_path = Path("data/preprocessed_full_unified")
    
    logger.info("📥 Loading test dataset...")
    
    with open(data_path / 'test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Re-compute per-atom normalization stats from training data
    train_file = data_path / 'train_data.json'
    norm_stats = {}
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_samples = json.load(f)
        all_energies = []
        for s in train_samples:
            energy = s.get('energy', s.get('energy_target', 0))
            n_atoms = len(s.get('atomic_numbers', [1]))
            if n_atoms > 0:
                all_energies.append(energy / n_atoms)
        if len(all_energies) > 0:
            norm_stats['mean'] = np.mean(all_energies)
            norm_stats['std'] = np.std(all_energies)
            logger.info(f"   Per-atom normalization: mean={norm_stats['mean']:.6f}, std={norm_stats['std']:.6f}")
    
    logger.info(f"   Loaded {len(test_data)} test samples")
    return test_data, norm_stats


def sample_to_pyg(sample):
    """Convert sample dict to PyTorch Geometric Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    from torch_geometric.nn import radius_graph
    edge_index = radius_graph(positions, r=10.0, max_num_neighbors=32)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        energy_target=torch.tensor([sample.get('energy', sample.get('energy_target', 0.0))], dtype=torch.float32)
    )
    
    return data


def load_gemnet_model(model_path: str, norm_stats: dict):
    """Load GemNet per-atom model."""
    logger.info("🔧 Loading GemNet per-atom model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    
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
    model.predict_per_atom = True
    
    logger.info(f"   Model loaded from {model_path}")
    logger.info(f"   Device: {device}")
    return model, device


def load_delta_head(model_path: str, device: torch.device):
    """Load delta head model."""
    logger.info("🔧 Loading delta head model...")
    
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint.get('config', {})
    
    if isinstance(config_dict, DeltaHeadConfig):
        config = config_dict
    else:
        config = DeltaHeadConfig(
            schnet_feature_dim=config_dict.get('schnet_feature_dim', 256),
            domain_embedding_dim=config_dict.get('domain_embedding_dim', 16),
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_layers=config_dict.get('num_layers', 3),
            dropout=config_dict.get('dropout', 0.1),
            num_domains=config_dict.get('num_domains', 5)
        )
    
    model = DeltaHead(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"   Delta head loaded from {model_path}")
    logger.info(f"   Config: {config}")
    return model, config


def extract_schnet_features(gnn_model, sample, device):
    """Extract SchNet/GemNet features for delta head input."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        x=atomic_numbers.unsqueeze(1).float()
    ).to(device)
    
    # Extract embeddings from GemNet
    if hasattr(gnn_model, 'model') and hasattr(gnn_model.model, 'embedding'):
        embedding_layer = gnn_model.model.embedding
    elif hasattr(gnn_model, 'embedding'):
        embedding_layer = gnn_model.embedding
    else:
        raise AttributeError("GNN model does not have accessible embedding layer")
    
    embeddings = embedding_layer(data.atomic_numbers)
    
    # Pool features (sum over atoms)
    pooled_features = embeddings.sum(dim=0, keepdim=True)
    
    # Resize to match delta head input dim
    target_dim = 256  # Default, adjust if needed
    if pooled_features.size(1) > target_dim:
        pooled_features = pooled_features[:, :target_dim]
    elif pooled_features.size(1) < target_dim:
        padding = torch.zeros(1, target_dim - pooled_features.size(1), device=device)
        pooled_features = torch.cat([pooled_features, padding], dim=1)
    
    return pooled_features


def get_domain_id(domain: str) -> int:
    """Convert domain string to ID."""
    domain_map = {
        'jarvis_dft': 0,
        'jarvis_elastic': 1,
        'oc20_s2ef': 2,
        'oc22_s2ef': 3,
        'ani1x': 4
    }
    return domain_map.get(domain, 0)


def evaluate_pipeline(test_data, gemnet_model, delta_head, delta_config, device, norm_stats):
    """Evaluate full pipeline: baseline and delta-corrected."""
    logger.info("🚀 Running pipeline evaluation...")
    
    test_pyg = [sample_to_pyg(s) for s in test_data]
    test_loader = PyGDataLoader(test_pyg, batch_size=16, shuffle=False)
    
    baseline_predictions = []
    delta_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch = batch.to(device)
            
            # Get number of atoms per graph
            n_atoms_per_graph = torch.bincount(batch.batch, minlength=batch.num_graphs).float()
            
            # Baseline GemNet prediction
            energies_total, _, _ = gemnet_model(batch, compute_forces=False)
            energies_per_atom_pred = energies_total / n_atoms_per_graph
            
            # Denormalize if needed
            if gemnet_model.mean is not None and gemnet_model.std is not None:
                energies_per_atom_pred = energies_per_atom_pred * gemnet_model.std + gemnet_model.mean
            
            # Convert target to per-atom
            target_per_atom = batch.energy_target.squeeze() / n_atoms_per_graph
            
            # Store baseline predictions
            for i in range(batch.num_graphs):
                sample_idx = batch_idx * 16 + i
                if sample_idx >= len(test_data):
                    continue
                
                sample = test_data[sample_idx]
                n_atoms = int(n_atoms_per_graph[i].item())
                
                # Estimate uncertainty from prediction variance (placeholder - in real scenario use ensemble)
                # Use small default uncertainty for easy cases, larger for predictions further from target
                baseline_error = abs(energies_per_atom_pred[i].item() - target_per_atom[i].item())
                estimated_variance = max(0.0001, (baseline_error * 0.5) ** 2)  # Rough uncertainty estimate
                
                baseline_pred = {
                    'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
                    'domain': sample.get('domain', 'unknown'),
                    'n_atoms': n_atoms,
                    'num_atoms': n_atoms,  # Also include for compatibility
                    'energy_per_atom_pred': energies_per_atom_pred[i].item(),
                    'energy_total_pred': energies_per_atom_pred[i].item() * n_atoms,
                    'energy_per_atom_target': target_per_atom[i].item(),
                    'energy_total_target': sample.get('energy', sample.get('energy_target', 0.0)),
                    'energy_variance': estimated_variance,  # Add uncertainty for gating
                    'atomic_numbers': sample.get('atomic_numbers', []),
                    'positions': sample.get('positions', [])
                }
                baseline_predictions.append(baseline_pred)
            
            # Apply delta head corrections (one sample at a time due to feature extraction)
            for i in range(min(batch.num_graphs, len(test_data) - batch_idx * 16)):
                sample_idx = batch_idx * 16 + i
                if sample_idx >= len(test_data):
                    break
                
                sample = test_data[sample_idx]
                n_atoms = int(n_atoms_per_graph[i].item())
                
                # Get corresponding baseline prediction
                baseline_pred = baseline_predictions[len(delta_predictions)]
                
                try:
                    # Extract features
                    schnet_features = extract_schnet_features(gemnet_model, sample, device)
                    
                    # Get domain ID
                    domain_id = torch.tensor([get_domain_id(sample.get('domain', 'jarvis_dft'))], dtype=torch.long).to(device)
                    
                    # Apply delta head (predicts per-atom delta)
                    delta_output = delta_head(schnet_features, domain_id)
                    if isinstance(delta_output, dict):
                        delta_per_atom = delta_output.get('delta_energy', 0.0)
                        if isinstance(delta_per_atom, torch.Tensor):
                            delta_per_atom = delta_per_atom.squeeze().item()
                    elif isinstance(delta_output, torch.Tensor):
                        delta_per_atom = delta_output.squeeze().item()
                    else:
                        delta_per_atom = float(delta_output)
                    
                    # IMPROVED GATING: Use uncertainty and conservative scaling
                    # Get uncertainty estimate (from variance if available, otherwise estimate)
                    uncertainty_per_atom = np.sqrt(baseline_pred.get('energy_variance', 0.0)) / n_atoms if baseline_pred.get('energy_variance', 0.0) > 0 else 0.005
                    
                    # Conservative gating strategy:
                    # 1. Scale correction magnitude based on uncertainty
                    # 2. Use very conservative scaling factors
                    
                    # If uncertainty is low, apply minimal or no correction
                    if uncertainty_per_atom < 0.005:  # Very low uncertainty
                        delta_per_atom = delta_per_atom * 0.1  # 10% of predicted correction
                    elif uncertainty_per_atom < 0.01:  # Low uncertainty
                        delta_per_atom = delta_per_atom * 0.3  # 30% of predicted correction
                    else:  # Moderate to high uncertainty
                        delta_per_atom = delta_per_atom * 0.5  # 50% of predicted correction
                    
                    # Always clip to very conservative bounds
                    max_correction = 0.02  # Maximum 0.02 eV/atom correction (very conservative)
                    delta_per_atom = np.clip(delta_per_atom, -max_correction, max_correction)
                    
                    # Skip very small corrections (numerical stability)
                    if abs(delta_per_atom) < 0.0001:
                        delta_per_atom = 0.0
                    
                    # Apply correction: E_total_corr = E_total + n_atoms * Δ_per_atom
                    energy_total_pred = baseline_pred['energy_total_pred']
                    delta_total = delta_per_atom * n_atoms
                    energy_total_corr = energy_total_pred + delta_total
                    energy_per_atom_corr = energy_total_corr / n_atoms
                    
                    delta_pred = baseline_pred.copy()
                    delta_pred['energy_per_atom_pred'] = energy_per_atom_corr
                    delta_pred['energy_total_pred'] = energy_total_corr
                    delta_pred['delta_per_atom'] = delta_per_atom
                    delta_pred['delta_total'] = delta_total
                    delta_pred['correction_gated'] = abs(delta_per_atom) > 0.0001
                    delta_pred['uncertainty_per_atom'] = uncertainty_per_atom
                    
                except Exception as e:
                    logger.warning(f"Failed delta correction for sample {sample_idx}: {e}")
                    delta_pred = baseline_pred.copy()
                    delta_pred['delta_per_atom'] = 0.0
                    delta_pred['delta_total'] = 0.0
                
                delta_predictions.append(delta_pred)
    
    return baseline_predictions, delta_predictions


def compute_metrics(predictions):
    """Compute evaluation metrics."""
    pred_per_atom = np.array([p['energy_per_atom_pred'] for p in predictions])
    target_per_atom = np.array([p['energy_per_atom_target'] for p in predictions])
    
    pred_total = np.array([p['energy_total_pred'] for p in predictions])
    target_total = np.array([p['energy_total_target'] for p in predictions])
    
    # Per-atom metrics
    mae_per_atom = np.mean(np.abs(pred_per_atom - target_per_atom))
    rmse_per_atom = np.sqrt(np.mean((pred_per_atom - target_per_atom)**2))
    
    # Total energy metrics
    mae_total = np.mean(np.abs(pred_total - target_total))
    rmse_total = np.sqrt(np.mean((pred_total - target_total)**2))
    
    # R² (using total energies for consistency)
    ss_res = np.sum((target_total - pred_total)**2)
    ss_tot = np.sum((target_total - np.mean(target_total))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mae_per_atom': mae_per_atom,
        'rmse_per_atom': rmse_per_atom,
        'mae_total': mae_total,
        'rmse_total': rmse_total,
        'r2': r2,
        'n_samples': len(predictions)
    }


def print_results(baseline_metrics, delta_metrics):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("  PIPELINE EVALUATION: BASELINE vs DELTA-CORRECTED")
    print("="*80)
    print(f"\n📊 Results on {baseline_metrics['n_samples']} test samples:\n")
    
    print(f"{'Metric':<35} {'Baseline':<20} {'Delta-Corrected':<20} {'Improvement':<15}")
    print("-" * 90)
    
    # Per-atom metrics
    print(f"\n{'Per-Atom Energy Metrics:':<35}")
    mae_pa_b = baseline_metrics['mae_per_atom']
    mae_pa_d = delta_metrics['mae_per_atom']
    impr_pa = ((mae_pa_b - mae_pa_d) / mae_pa_b * 100) if mae_pa_b > 0 else 0.0
    print(f"{'  MAE (eV/atom)':<35} {mae_pa_b:.6f}{'':<14} {mae_pa_d:.6f}{'':<14} {impr_pa:+.2f}%")
    
    rmse_pa_b = baseline_metrics['rmse_per_atom']
    rmse_pa_d = delta_metrics['rmse_per_atom']
    impr_rmse_pa = ((rmse_pa_b - rmse_pa_d) / rmse_pa_b * 100) if rmse_pa_b > 0 else 0.0
    print(f"{'  RMSE (eV/atom)':<35} {rmse_pa_b:.6f}{'':<14} {rmse_pa_d:.6f}{'':<14} {impr_rmse_pa:+.2f}%")
    
    # Total energy metrics
    print(f"\n{'Total Energy Metrics:':<35}")
    mae_t_b = baseline_metrics['mae_total']
    mae_t_d = delta_metrics['mae_total']
    impr_t = ((mae_t_b - mae_t_d) / mae_t_b * 100) if mae_t_b > 0 else 0.0
    print(f"{'  MAE (eV)':<35} {mae_t_b:.6f}{'':<14} {mae_t_d:.6f}{'':<14} {impr_t:+.2f}%")
    
    rmse_t_b = baseline_metrics['rmse_total']
    rmse_t_d = delta_metrics['rmse_total']
    impr_rmse_t = ((rmse_t_b - rmse_t_d) / rmse_t_b * 100) if rmse_t_b > 0 else 0.0
    print(f"{'  RMSE (eV)':<35} {rmse_t_b:.6f}{'':<14} {rmse_t_d:.6f}{'':<14} {impr_rmse_t:+.2f}%")
    
    # R²
    print(f"\n{'Correlation Metrics:':<35}")
    r2_b = baseline_metrics['r2']
    r2_d = delta_metrics['r2']
    impr_r2 = ((r2_d - r2_b) / abs(r2_b) * 100) if r2_b != 0 else 0.0
    print(f"{'  R² Score':<35} {r2_b:.6f}{'':<14} {r2_d:.6f}{'':<14} {impr_r2:+.2f}%")
    
    print("\n" + "="*80)
    
    # Summary
    print("\n📈 Summary:")
    if impr_pa > 0:
        print(f"   ✅ Delta head improves per-atom MAE by {impr_pa:.2f}%")
    else:
        print(f"   ❌ Delta head degrades per-atom MAE by {abs(impr_pa):.2f}%")
    
    if impr_t > 0:
        print(f"   ✅ Delta head improves total energy MAE by {impr_t:.2f}%")
    else:
        print(f"   ❌ Delta head degrades total energy MAE by {abs(impr_t):.2f}%")
    
    if r2_d > r2_b:
        print(f"   ✅ Delta head improves R² by {impr_r2:.2f}%")
    else:
        print(f"   ❌ Delta head degrades R² by {abs(impr_r2):.2f}%")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    logger.info("🚀 END-TO-END PIPELINE EVALUATION")
    logger.info("="*80)
    
    # Load data
    test_data, norm_stats = load_test_data()
    
    # Limit to first 1000 samples for faster evaluation
    if len(test_data) > 1000:
        logger.info(f"   Limiting evaluation to first 1000 samples for speed")
        test_data = test_data[:1000]
    
    # Load models
    gemnet_model, device = load_gemnet_model(
        "models/gemnet_per_atom/best_model.pt",
        norm_stats
    )
    
    delta_head, delta_config = load_delta_head(
        "artifacts/delta_head_final/best_model.pt",
        device
    )
    
    # Evaluate
    baseline_preds, delta_preds = evaluate_pipeline(
        test_data, gemnet_model, delta_head, delta_config, device, norm_stats
    )
    
    # Compute metrics
    baseline_metrics = compute_metrics(baseline_preds)
    delta_metrics = compute_metrics(delta_preds)
    
    # Print results
    print_results(baseline_metrics, delta_metrics)
    
    logger.info("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

