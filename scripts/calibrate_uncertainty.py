#!/usr/bin/env python3
"""
Uncertainty Calibration Analysis for GNN Models
Computes ECE, NLL, coverage curves, and reliability diagrams for publication.

Usage:
    python scripts/calibrate_uncertainty.py \
        --model-path models/gemnet_per_atom/best_model.pt \
        --test-data data/preprocessed_full_unified/test_data.json \
        --output-dir artifacts/uncertainty_calibration
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.data import Data, Batch
from ase import Atoms
from ase.data import chemical_symbols

# Direct import to avoid MACE dependencies via gnn.__init__
import importlib.util
spec = importlib.util.spec_from_file_location("model_gemnet", str(Path(__file__).parent.parent / "src" / "gnn" / "model_gemnet.py"))
model_gemnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_gemnet)
GemNetWrapper = model_gemnet.GemNetWrapper

from src.graphs.periodic_graph import PeriodicGraph


def load_test_data(test_file: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test dataset from JSON."""
    print(f"📥 Loading test data from {test_file}...")
    
    with open(test_file, 'r') as f:
        samples = json.load(f)
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"   Loaded {len(samples)} test samples")
    return samples


def create_graph_from_sample(sample: Dict) -> Data:
    """Create PyTorch Geometric graph from sample dict."""
    atomic_numbers = np.array(sample['atomic_numbers'], dtype=np.int64)
    positions = np.array(sample['positions'], dtype=np.float32)
    
    # Create cell if available
    if 'cell' in sample and sample['cell']:
        cell = np.array(sample['cell'], dtype=np.float32)
    else:
        # Default cell
        if len(positions) > 0:
            max_dist = float(np.max(positions) - np.min(positions)) + 10.0
        else:
            max_dist = 10.0
        cell = np.eye(3, dtype=np.float32) * max_dist
    
    # Create graph using PeriodicGraph
    graph_builder = PeriodicGraph(cutoff_radius=10.0, max_neighbors=100)
    graph = graph_builder.build_graph(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell_vectors=cell
    )
    
    return graph


def load_model(model_path: Path, device: str = 'cuda') -> GemNetWrapper:
    """Load trained GemNet model."""
    print(f"🔧 Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config and normalization stats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        energy_mean = checkpoint.get('energy_mean', 0.0)
        energy_std = checkpoint.get('energy_std', 1.0)
        model_config = checkpoint.get('model_config', {})
    else:
        state_dict = checkpoint
        energy_mean = 0.0
        energy_std = 1.0
        model_config = {}
    
    # Infer num_atoms from embedding weight size if available
    if 'embedding.embedding.weight' in state_dict:
        num_atoms = state_dict['embedding.embedding.weight'].shape[0]
    else:
        num_atoms = model_config.get('num_atoms', 100)
    
    # Create model with correct config
    model = GemNetWrapper(
        num_atoms=num_atoms,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_filters=model_config.get('num_filters', 256),
        num_interactions=model_config.get('num_interactions', 6),
        cutoff=model_config.get('cutoff', 10.0),
        mean=energy_mean,
        std=energy_std
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Patch _compute_edges to work without torch-cluster
    original_compute_edges = model._compute_edges
    def patched_compute_edges(positions, batch_indices):
        """Fallback edge computation without torch-cluster."""
        n_atoms = len(positions)
        # Use distance-based edge creation
        distances = torch.cdist(positions, positions)
        edge_mask = (distances < model.cutoff) & (distances > 1e-8)
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        
        # Compute distances and attributes
        row, col = edge_index
        edge_vec = positions[row] - positions[col]
        distances = torch.norm(edge_vec, dim=-1)
        edge_attr = edge_vec / (distances.unsqueeze(-1) + 1e-8)
        
        return edge_index, edge_attr, distances
    
    model._compute_edges = patched_compute_edges
    
    print(f"   Model loaded (mean={energy_mean:.6f}, std={energy_std:.6f})")
    return model


def simple_predict(
    model: GemNetWrapper,
    batch: Batch,
    n_samples: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple uncertainty estimation via prediction variance.
    Since GemNet doesn't have dropout, we use a simple variance estimate
    by adding small noise to positions.
    
    Note: We skip noise addition if torch-cluster is unavailable and
    just use single deterministic prediction.
    """
    predictions = []
    
    # Check if we can add noise (requires proper edge computation)
    try:
        from torch_geometric.nn import radius_graph
        can_add_noise = True
    except ImportError:
        can_add_noise = False
    
    original_pos = batch.pos.clone() if hasattr(batch, 'pos') and can_add_noise else None
    
    with torch.no_grad():
        for i in range(n_samples):
            # Add small random noise to positions (stochastic uncertainty)
            if original_pos is not None and i > 0:  # Keep first sample as baseline
                noise_scale = 0.01  # 0.01 Å noise
                noise = torch.randn_like(original_pos) * noise_scale
                batch.pos = original_pos + noise
            
            pred, _, _ = model(batch, compute_forces=False)
            # Denormalize if needed
            if model.mean is not None and model.std is not None:
                pred = model.denormalize_energy(pred)
            predictions.append(pred.cpu())
    
    # Restore original positions
    if original_pos is not None:
        batch.pos = original_pos
    
    predictions = torch.stack(predictions)  # (n_samples, batch_size)
    mean_pred = torch.mean(predictions, dim=0)
    var_pred = torch.var(predictions, dim=0)
    
    # If variance is too small (single deterministic prediction), add small constant
    if torch.mean(var_pred) < 1e-8:
        var_pred = torch.full_like(var_pred, 0.01)  # Small baseline uncertainty
    
    return mean_pred, var_pred


def compute_calibration_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics:
    - ECE (Expected Calibration Error)
    - NLL (Negative Log Likelihood)
    - Coverage curves
    """
    # Convert uncertainties to confidence intervals
    std = np.sqrt(np.maximum(uncertainties, 1e-10))  # Avoid division by zero
    
    # Compute errors
    errors = np.abs(predictions - targets)
    
    # Normalize uncertainties to [0, 1] for calibration
    max_std = np.max(std)
    if max_std > 0:
        normalized_std = std / max_std
    else:
        normalized_std = np.ones_like(std)
    
    # ECE: Expected Calibration Error
    # Bin predictions by confidence and compare to accuracy
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = (normalized_std >= bin_lower) & (normalized_std < bin_upper)
        if bin_upper == 1.0:  # Include upper bound
            in_bin = (normalized_std >= bin_lower) & (normalized_std <= bin_upper)
        
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            # Accuracy in this bin (inverse of error)
            bin_error = np.mean(errors[in_bin])
            bin_accuracy = 1.0 - (bin_error / (np.max(errors) + 1e-8))  # Normalized accuracy
            bin_confidence = 1.0 - np.mean(normalized_std[in_bin])  # Confidence = 1 - uncertainty
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            
            # ECE contribution
            ece += np.abs(bin_accuracy - bin_confidence) * bin_count / len(predictions)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    
    # NLL: Negative Log Likelihood (assuming Gaussian)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * uncertainties) + 
        (predictions - targets)**2 / (uncertainties + 1e-8)
    )
    
    # Correlation between uncertainty and error
    corr = np.corrcoef(std, errors)[0, 1]
    
    # Coverage: fraction of predictions within k*std
    coverage_68 = np.mean(np.abs(predictions - targets) <= std)  # 1 sigma
    coverage_95 = np.mean(np.abs(predictions - targets) <= 2 * std)  # 2 sigma
    
    return {
        'ece': ece,
        'nll': nll,
        'correlation': corr,
        'coverage_68': coverage_68,
        'coverage_95': coverage_95,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'mean_error': np.mean(errors),
        'mean_uncertainty': np.mean(std)
    }


def plot_reliability_diagram(
    metrics: Dict,
    output_path: Path,
    n_bins: int = 10
):
    """Plot reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bin_accuracies = metrics['bin_accuracies']
    bin_confidences = metrics['bin_confidences']
    bin_counts = metrics['bin_counts']
    
    # Plot perfect calibration line
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot calibration curve
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    ax.plot(bin_centers, bin_accuracies, 'o-', label='Model', linewidth=2, markersize=8)
    
    # Add count annotations
    for i, (center, acc, conf, count) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            ax.text(center, acc, f'  n={count}', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Confidence (1 - Normalized Uncertainty)', fontsize=12)
    ax.set_ylabel('Accuracy (1 - Normalized Error)', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add ECE text
    ece_text = f'ECE = {metrics["ece"]:.4f}'
    ax.text(0.05, 0.95, ece_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved reliability diagram to {output_path / 'reliability_diagram.png'}")


def plot_coverage_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    output_path: Path
):
    """Plot coverage vs confidence interval."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    errors = np.abs(predictions - targets)
    
    # Vary k from 0 to 3
    k_values = np.linspace(0, 3, 100)
    coverages = []
    
    for k in k_values:
        coverage = np.mean(errors <= k * std)
        coverages.append(coverage)
    
    ax.plot(k_values, coverages, 'b-', linewidth=2, label='Model')
    
    # Add reference lines
    ax.axhline(0.68, color='r', linestyle='--', label='68% (1σ)', alpha=0.7)
    ax.axhline(0.95, color='r', linestyle=':', label='95% (2σ)', alpha=0.7)
    ax.axvline(1.0, color='g', linestyle='--', alpha=0.7)
    ax.axvline(2.0, color='g', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Confidence Interval (k × σ)', fontsize=12)
    ax.set_ylabel('Coverage Fraction', fontsize=12)
    ax.set_title('Coverage Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'coverage_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved coverage curve to {output_path / 'coverage_curve.png'}")


def plot_error_vs_uncertainty(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    output_path: Path
):
    """Plot error vs uncertainty scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    errors = np.abs(predictions - targets)
    
    # Scatter plot with density
    ax.scatter(std, errors, alpha=0.3, s=10)
    
    # Add correlation line
    z = np.polyfit(std, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(std.min(), std.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Fit (corr={np.corrcoef(std, errors)[0,1]:.3f})')
    
    ax.set_xlabel('Uncertainty (σ)', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Error vs Uncertainty', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_vs_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved error vs uncertainty plot to {output_path / 'error_vs_uncertainty.png'}")


def main():
    parser = argparse.ArgumentParser(description='Uncertainty Calibration Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--output-dir', type=str, default='artifacts/uncertainty_calibration',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of test samples (None for all)')
    parser.add_argument('--n-mc-samples', type=int, default=10,
                       help='Number of Monte Carlo dropout samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  UNCERTAINTY CALIBRATION ANALYSIS")
    print("="*80)
    
    # Load model and data
    model = load_model(Path(args.model_path), device=device)
    test_samples = load_test_data(Path(args.test_data), max_samples=args.max_samples)
    
    # Predict with uncertainty
    print(f"\n🔮 Generating predictions with uncertainty estimation (n={args.n_mc_samples})...")
    predictions = []
    targets = []
    uncertainties = []
    
    model.eval()
    for i in tqdm(range(0, len(test_samples), args.batch_size), desc="Predicting"):
        batch_samples = test_samples[i:i+args.batch_size]
        
        # Create graphs
        graphs = [create_graph_from_sample(s) for s in batch_samples]
        batch = Batch.from_data_list(graphs).to(device)
        
        # Get targets
        # CRITICAL FIX: Test data is already in per-atom format (formation_energy_per_atom)
        batch_targets = []
        for s in batch_samples:
            n_atoms = len(s['atomic_numbers'])
            if 'formation_energy_per_atom' in s:
                energy_per_atom = s['formation_energy_per_atom']
            else:
                energy = s.get('energy', s.get('energy_target', 0.0))
                # Heuristic: if energy is large (>50 eV), assume total and convert
                if abs(energy) > 50 and n_atoms > 0:
                    energy_per_atom = energy / n_atoms
                else:
                    # Already per-atom (typical range: -5 to 2 eV/atom)
                    energy_per_atom = energy
            batch_targets.append(energy_per_atom)
        
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)
        
        # Uncertainty estimation via prediction variance
        mean_pred, var_pred = simple_predict(
            model, batch, n_samples=args.n_mc_samples
        )
        
        predictions.extend(mean_pred.cpu().numpy())
        targets.extend(batch_targets.numpy())
        uncertainties.extend(var_pred.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    uncertainties = np.array(uncertainties)
    
    print(f"   Generated {len(predictions)} predictions")
    
    # Compute calibration metrics
    print(f"\n📊 Computing calibration metrics...")
    metrics = compute_calibration_metrics(predictions, targets, uncertainties)
    
    # Print results
    print("\n" + "="*80)
    print("  CALIBRATION METRICS")
    print("="*80)
    print(f"Expected Calibration Error (ECE): {metrics['ece']:.6f}")
    print(f"Negative Log Likelihood (NLL): {metrics['nll']:.6f}")
    print(f"Uncertainty-Error Correlation: {metrics['correlation']:.6f}")
    print(f"Coverage @ 1σ (68%): {metrics['coverage_68']:.4f}")
    print(f"Coverage @ 2σ (95%): {metrics['coverage_95']:.4f}")
    print(f"Mean Absolute Error: {metrics['mean_error']:.6f} eV/atom")
    print(f"Mean Uncertainty: {metrics['mean_uncertainty']:.6f} eV/atom")
    
    # Generate plots
    print(f"\n📈 Generating calibration plots...")
    plot_reliability_diagram(metrics, output_dir)
    plot_coverage_curve(predictions, targets, uncertainties, output_dir)
    plot_error_vs_uncertainty(predictions, targets, uncertainties, output_dir)
    
    # Save metrics
    metrics_save = {
        'ece': float(metrics['ece']),
        'nll': float(metrics['nll']),
        'correlation': float(metrics['correlation']),
        'coverage_68': float(metrics['coverage_68']),
        'coverage_95': float(metrics['coverage_95']),
        'mean_error': float(metrics['mean_error']),
        'mean_uncertainty': float(metrics['mean_uncertainty']),
        'n_samples': len(predictions)
    }
    
    with open(output_dir / 'calibration_metrics.json', 'w') as f:
        json.dump(metrics_save, f, indent=2)
    
    print(f"\n✅ Calibration analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

