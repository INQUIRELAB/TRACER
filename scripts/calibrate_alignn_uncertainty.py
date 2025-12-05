#!/usr/bin/env python3
"""
Uncertainty Calibration for ALIGNN Model
For fair comparison with GemNet uncertainty calibration.

Usage:
    python scripts/calibrate_alignn_uncertainty.py \
        --model-path models/alignn_fixed/best_model.pt \
        --test-data data/preprocessed_full_unified/test_data.json \
        --output-dir artifacts/uncertainty_calibration_alignn \
        --max-samples 1000 \
        --n-mc-samples 10 \
        --device cuda
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ase.data import chemical_symbols
from jarvis.core.atoms import Atoms as JAtoms
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

sns.set_style('whitegrid')


def load_test_data(test_file: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test dataset from JSON."""
    print(f"📥 Loading test data from {test_file}...")
    
    with open(test_file, 'r') as f:
        samples = json.load(f)
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"   Loaded {len(samples)} test samples")
    return samples


def load_model(model_path: Path, device: str = 'cuda') -> Tuple[ALIGNNAtomWise, Dict]:
    """Load trained ALIGNN model."""
    print(f"🔧 Loading ALIGNN model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get normalization stats
    energy_mean = checkpoint.get('energy_mean', checkpoint.get('normalization', {}).get('mean', 0.0))
    energy_std = checkpoint.get('energy_std', checkpoint.get('normalization', {}).get('std', 1.0))
    
    # Create model (match evaluation script config)
    config = ALIGNNAtomWiseConfig(
        name='alignn_atomwise',
        atom_input_features=92,
        embedding_features=256,
        hidden_features=256,
        alignn_layers=2,
        gcn_layers=2,
        output_features=1,
    )
    
    model = ALIGNNAtomWise(config=config).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    norm_stats = {'mean': energy_mean, 'std': energy_std}
    print(f"   Model loaded (mean={energy_mean:.6f}, std={energy_std:.6f})")
    
    return model, norm_stats


def ensemble_predict_via_noise(
    model: ALIGNNAtomWise,
    j_atoms: JAtoms,
    energy_mean: float,
    energy_std: float,
    n_samples: int = 10,
    noise_scale: float = 0.01
) -> Tuple[float, float]:
    """
    Estimate uncertainty via input noise perturbation.
    Fixed: Handle ALIGNN's requires_grad requirement properly.
    """
    model.eval()
    
    # Store original coordinates and lattice
    original_coords = np.array(j_atoms.coords).copy()
    original_lattice = np.array(j_atoms.lattice_mat).copy()
    
    predictions = []
    
    # Don't use torch.no_grad() - ALIGNN needs gradients on lattice
    for i in range(n_samples):
        # Add noise to coordinates (except first sample - baseline)
        if i > 0:
            noise = np.random.randn(*original_coords.shape) * noise_scale
            j_atoms.coords = (original_coords + noise).tolist()
        else:
            j_atoms.coords = original_coords.tolist()
        
        # Ensure lattice is set (may be modified by noise)
        j_atoms.lattice_mat = original_lattice.tolist()
        
        try:
            # Create graph - must be inside try to handle gradient issues
            g, lg = Graph.atom_dgl_multigraph(j_atoms)
            
            # CRITICAL: Lattice must have requires_grad=True for ALIGNN
            lat = torch.tensor(j_atoms.lattice_mat, dtype=torch.float32, requires_grad=True)
            
            # Move to device
            device = next(model.parameters()).device
            g = g.to(device)
            lat = lat.to(device)
            
            # Predict - ALIGNN may need gradients even in eval mode
            output_dict = model((g, lat))
            output = output_dict['out'] if isinstance(output_dict, dict) else output_dict
            
            # Detach to avoid gradient accumulation
            pred_normalized = output.squeeze().detach().cpu().item()
            
            # Denormalize
            pred_per_atom = pred_normalized * energy_std + energy_mean
            predictions.append(pred_per_atom)
            
        except RuntimeError as e:
            # Skip samples that fail due to gradient issues
            if "grad" in str(e).lower() or "requires_grad" in str(e).lower():
                # Use baseline prediction for failed samples
                if i == 0:
                    # First sample failed - can't proceed
                    raise
                # Use previous prediction as fallback
                predictions.append(predictions[-1] if predictions else 0.0)
                continue
            else:
                raise
    
    # Restore original coordinates
    j_atoms.coords = original_coords.tolist()
    j_atoms.lattice_mat = original_lattice.tolist()
    
    if len(predictions) == 0:
        raise RuntimeError("All predictions failed")
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions) + 1e-6  # Add minimum variance
    
    return mean_pred, var_pred


def compute_ece_regression(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """Compute Expected Calibration Error (ECE) for regression."""
    errors = np.abs(predictions - targets)
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    
    # Normalize uncertainties to confidence levels
    max_std = np.percentile(std, 95)
    if max_std > 0:
        normalized_confidence = 1.0 - (std / max_std)
        normalized_confidence = np.clip(normalized_confidence, 0, 1)
    else:
        normalized_confidence = np.ones_like(std)
    
    # Normalize errors to accuracy
    max_error = np.percentile(errors, 95)
    if max_error > 0:
        normalized_accuracy = 1.0 - (errors / max_error)
        normalized_accuracy = np.clip(normalized_accuracy, 0, 1)
    else:
        normalized_accuracy = np.ones_like(errors)
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (normalized_confidence >= bin_lower) & (normalized_confidence < bin_upper)
        if bin_upper == 1.0:
            in_bin = (normalized_confidence >= bin_lower) & (normalized_confidence <= bin_upper)
        
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_accuracy = np.mean(normalized_accuracy[in_bin])
            bin_confidence = np.mean(normalized_confidence[in_bin])
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            
            ece += np.abs(bin_accuracy - bin_confidence) * bin_count / len(predictions)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
    
    # NLL
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * uncertainties + 1e-8) + 
        (predictions - targets)**2 / (uncertainties + 1e-8)
    )
    
    # Correlation
    correlation = np.corrcoef(std, errors)[0, 1] if len(std) > 1 and np.std(std) > 0 else 0.0
    
    # Coverage
    coverage_68 = np.mean(errors <= std)
    coverage_95 = np.mean(errors <= 2 * std)
    
    return {
        'ece': ece,
        'nll': nll,
        'correlation': correlation,
        'coverage_68': coverage_68,
        'coverage_95': coverage_95,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'mean_error': np.mean(errors),
        'mean_uncertainty': np.mean(std),
        'median_error': np.median(errors),
        'median_uncertainty': np.median(std)
    }


def plot_reliability_diagram(metrics: Dict, output_path: Path, n_bins: int = 10):
    """Plot reliability diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bin_accuracies = np.array(metrics['bin_accuracies'])
    bin_confidences = np.array(metrics['bin_confidences'])
    bin_counts = np.array(metrics['bin_counts'])
    
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
    
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    ax.plot(bin_centers, bin_accuracies, 'o-', label='ALIGNN', linewidth=2, markersize=10, color='red')
    
    for i, (center, acc, conf, count) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            ax.text(center, acc + 0.05, f'n={count}', fontsize=9, ha='center', alpha=0.7)
    
    ax.set_xlabel('Mean Confidence', fontsize=14)
    ax.set_ylabel('Mean Accuracy', fontsize=14)
    ax.set_title('ALIGNN Reliability Diagram', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ece_text = f'ECE = {metrics["ece"]:.4f}'
    ax.text(0.02, 0.98, ece_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved reliability diagram")


def plot_coverage_curve(predictions: np.ndarray, targets: np.ndarray,
                       uncertainties: np.ndarray, output_path: Path):
    """Plot coverage vs confidence interval."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    errors = np.abs(predictions - targets)
    
    k_values = np.linspace(0, 3, 200)
    coverages = [np.mean(errors <= k * std) for k in k_values]
    
    ax.plot(k_values, coverages, 'r-', linewidth=2, label='ALIGNN Coverage', zorder=3)
    
    ax.axhline(0.68, color='r', linestyle='--', label='68% Target (1σ)', alpha=0.7, linewidth=1.5)
    ax.axhline(0.95, color='r', linestyle=':', label='95% Target (2σ)', alpha=0.7, linewidth=1.5)
    ax.axvline(1.0, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(2.0, color='g', linestyle=':', alpha=0.5, linewidth=1)
    
    coverage_1sigma = np.mean(errors <= 1.0 * std)
    coverage_2sigma = np.mean(errors <= 2.0 * std)
    
    ax.plot([1.0], [coverage_1sigma], 'go', markersize=10, label=f'1σ = {coverage_1sigma:.1%}', zorder=4)
    ax.plot([2.0], [coverage_2sigma], 'gs', markersize=10, label=f'2σ = {coverage_2sigma:.1%}', zorder=4)
    
    ax.set_xlabel('Confidence Interval Multiplier (k × σ)', fontsize=14)
    ax.set_ylabel('Coverage Fraction', fontsize=14)
    ax.set_title('ALIGNN Coverage Curve', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'coverage_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved coverage curve")


def plot_error_vs_uncertainty(predictions: np.ndarray, targets: np.ndarray,
                              uncertainties: np.ndarray, output_path: Path):
    """Plot error vs uncertainty scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    errors = np.abs(predictions - targets)
    
    ax.scatter(std, errors, alpha=0.4, s=15, edgecolors='none', color='red', zorder=2)
    
    if len(std) > 1 and np.std(std) > 0:
        z = np.polyfit(std, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(std.min(), std.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                label=f'Fit (corr={np.corrcoef(std, errors)[0,1]:.3f})', zorder=3)
    
    ax.set_xlabel('Predicted Uncertainty (σ)', fontsize=14)
    ax.set_ylabel('Absolute Error |y - ŷ|', fontsize=14)
    ax.set_title('ALIGNN Error vs Uncertainty', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_vs_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved error vs uncertainty plot")


def main():
    parser = argparse.ArgumentParser(description='ALIGNN Uncertainty Calibration')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='artifacts/uncertainty_calibration_alignn')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--n-mc-samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-bins', type=int, default=10)
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  ALIGNN UNCERTAINTY CALIBRATION ANALYSIS")
    print("="*80)
    print()
    
    # Load model
    model, norm_stats = load_model(Path(args.model_path), args.device)
    device = next(model.parameters()).device
    
    # Load test data
    test_samples = load_test_data(Path(args.test_data), args.max_samples)
    
    # Generate predictions
    print(f"\n🔮 Generating predictions with noise perturbation (n={args.n_mc_samples})...")
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    for sample in tqdm(test_samples, desc="Predicting"):
        atomic_numbers = sample['atomic_numbers']
        positions = np.array(sample['positions'])
        n_atoms = len(atomic_numbers)
        
        if n_atoms == 0:
            continue
        
        # Energy target (per-atom)
        if 'formation_energy_per_atom' in sample:
            energy_per_atom = sample['formation_energy_per_atom']
        else:
            energy = sample.get('energy', sample.get('energy_target', 0.0))
            if abs(energy) > 50 and n_atoms > 0:
                energy_per_atom = energy / n_atoms
            else:
                energy_per_atom = energy
        
        # Create cell
        if 'cell' in sample and sample['cell']:
            cell = np.array(sample['cell'])
        else:
            max_dist = np.max(positions) - np.min(positions) if len(positions) > 0 else 10.0
            cell = np.eye(3) * (max_dist + 10.0)
        
        # Convert to JARVIS Atoms
        element_list = [chemical_symbols[z] for z in atomic_numbers]
        j_atoms = JAtoms(
            lattice_mat=cell.tolist(),
            elements=element_list,
            coords=positions.tolist()
        )
        
        try:
            mean_pred, var_pred = ensemble_predict_via_noise(
                model, j_atoms, norm_stats['mean'], norm_stats['std'], 
                args.n_mc_samples
            )
            
            all_predictions.append(mean_pred)
            all_targets.append(energy_per_atom)
            all_uncertainties.append(var_pred)
        except Exception as e:
            print(f"   Warning: Failed to predict sample: {e}")
            continue
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    uncertainties = np.array(all_uncertainties)
    
    print(f"\n📊 Computing calibration metrics...")
    metrics = compute_ece_regression(predictions, targets, uncertainties, args.n_bins)
    
    # Print results
    print("\n" + "="*80)
    print("  ALIGNN CALIBRATION RESULTS")
    print("="*80)
    print(f"\n📈 Metrics:")
    print(f"   ECE:                {metrics['ece']:.6f}")
    print(f"   NLL:                {metrics['nll']:.6f}")
    print(f"   Correlation:        {metrics['correlation']:.4f}")
    print(f"   Coverage @ 1σ:      {metrics['coverage_68']:.2%}")
    print(f"   Coverage @ 2σ:      {metrics['coverage_95']:.2%}")
    print(f"   Mean Error:         {metrics['mean_error']:.6f} eV/atom")
    print(f"   Mean Uncertainty:   {metrics['mean_uncertainty']:.6f} eV/atom")
    print(f"   Median Error:       {metrics['median_error']:.6f} eV/atom")
    print(f"   Median Uncertainty: {metrics['median_uncertainty']:.6f} eV/atom")
    
    # Save metrics
    def convert_numpy(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(output_path / 'calibration_metrics.json', 'w') as f:
        json.dump(convert_numpy(metrics), f, indent=2)
    
    # Generate plots
    print("\n📊 Generating plots...")
    plot_reliability_diagram(metrics, output_path, args.n_bins)
    plot_coverage_curve(predictions, targets, uncertainties, output_path)
    plot_error_vs_uncertainty(predictions, targets, uncertainties, output_path)
    
    print("\n" + "="*80)
    print("  ✅ ALIGNN Calibration analysis complete!")
    print(f"   Results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

