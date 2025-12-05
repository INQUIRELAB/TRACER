#!/usr/bin/env python3
"""
Improved Uncertainty Calibration for GemNet with FiLM
Computes ECE, reliability diagrams, coverage curves for publication.

Usage:
    python scripts/calibrate_uncertainty_improved.py \
        --model-path models/gemnet_per_atom_fixed/best_model.pt \
        --test-data data/preprocessed_full_unified/test_data.json \
        --output-dir artifacts/uncertainty_calibration_improved \
        --max-samples 1000 \
        --n-mc-samples 10 \
        --device cuda
"""

import torch
import torch.nn as nn
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
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from ase import Atoms
from ase.data import chemical_symbols

# Import GemNet model
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gnn.model_gemnet import GemNetWrapper

# Force RTX 4090
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


def get_domain_id(domain_str: str) -> int:
    """Map domain string to domain ID for FiLM adaptation."""
    domain_lower = domain_str.lower()
    if 'jarvis' in domain_lower:
        if 'elastic' in domain_lower:
            return 1
        else:
            return 0
    elif 'oc20' in domain_lower:
        return 2
    elif 'oc22' in domain_lower:
        return 3
    elif 'ani' in domain_lower:
        return 4
    else:
        return 0


def sample_to_pyg_data(sample: Dict, energy_mean: float, energy_std: float) -> Data:
    """Convert sample dict to PyG Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Energy target (already per-atom)
    if 'formation_energy_per_atom' in sample:
        energy_per_atom = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', sample.get('energy_target', 0.0))
        n_atoms = len(atomic_numbers)
        if abs(energy) > 50 and n_atoms > 0:
            energy_per_atom = energy / n_atoms
        else:
            energy_per_atom = energy
    
    # Normalize energy
    energy_normalized = (energy_per_atom - energy_mean) / energy_std
    
    # Domain ID for FiLM
    domain_str = sample.get('domain', 'jarvis-dft')
    domain_id = torch.tensor([get_domain_id(domain_str)], dtype=torch.long)
    
    # Cell for PBC
    cell = None
    if 'cell' in sample and sample['cell']:
        cell = torch.tensor(sample['cell'], dtype=torch.float32)
    
    data = Data(
        x=atomic_numbers.unsqueeze(1),
        pos=positions,
        energy_target=torch.tensor([energy_normalized], dtype=torch.float32),
        energy_target_original=torch.tensor([energy_per_atom], dtype=torch.float32),
        domain_id=domain_id,
        num_nodes=len(atomic_numbers)
    )
    
    if cell is not None:
        data.cell = cell
    
    return data


def custom_collate_fn(batch_list):
    """Custom collate function that properly handles domain_id for FiLM."""
    batch = Batch.from_data_list(batch_list)
    
    # Extract domain_id per graph
    batch_domain_ids = []
    for data in batch_list:
        if hasattr(data, 'domain_id') and data.domain_id is not None:
            domain_id = data.domain_id[0].item() if len(data.domain_id) > 0 else 0
        else:
            domain_id = 0
        batch_domain_ids.append(domain_id)
    
    batch.graph_domain_ids = batch_domain_ids
    return batch


def load_model(model_path: Path, device: str = 'cuda') -> Tuple[GemNetWrapper, Dict]:
    """Load trained GemNet model."""
    print(f"🔧 Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config and normalization stats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        energy_mean = checkpoint.get('normalization', {}).get('mean', checkpoint.get('energy_mean', 0.0))
        energy_std = checkpoint.get('normalization', {}).get('std', checkpoint.get('energy_std', 1.0))
        model_config = checkpoint.get('model_config', {})
    else:
        state_dict = checkpoint
        energy_mean = 0.0
        energy_std = 1.0
        model_config = {}
    
    # Create model with same config
    model = GemNetWrapper(
        num_atoms=model_config.get('num_atoms', 95),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_filters=model_config.get('num_filters', 256),
        num_interactions=model_config.get('num_interactions', 6),
        cutoff=model_config.get('cutoff', 10.0),
        mean=energy_mean,
        std=energy_std,
        use_film=model_config.get('use_film', True),
        num_domains=model_config.get('num_domains', 5),
        film_dim=model_config.get('film_dim', 16)
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    norm_stats = {'mean': energy_mean, 'std': energy_std}
    print(f"   Model loaded (mean={energy_mean:.6f}, std={energy_std:.6f})")
    print(f"   FiLM enabled: {model_config.get('use_film', False)}")
    
    return model, norm_stats


def ensemble_predict_via_noise(
    model: GemNetWrapper,
    batch: Batch,
    domain_id: torch.Tensor,
    n_samples: int = 10,
    noise_scale: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate uncertainty via input noise perturbation (since model has no dropout).
    Adds small random noise to atomic positions and measures prediction variance.
    """
    model.eval()
    
    # Store original positions
    original_pos = batch.pos.clone()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Add noise to positions (except first sample - baseline)
            if i > 0:
                noise = torch.randn_like(original_pos) * noise_scale
                batch.pos = original_pos + noise
            else:
                batch.pos = original_pos
            
            energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
            
            # Convert to per-atom
            if hasattr(batch, 'batch'):
                n_atoms_per_graph = torch.bincount(batch.batch, minlength=len(energies_total))
                n_atoms_per_graph = n_atoms_per_graph.float()
            else:
                n_atoms_per_graph = torch.tensor([len(batch.atomic_numbers)], 
                                                  dtype=energies_total.dtype, device=energies_total.device)
            energies_per_atom = energies_total / n_atoms_per_graph
            predictions.append(energies_per_atom.cpu())
    
    # Restore original positions
    batch.pos = original_pos
    
    predictions = torch.stack(predictions)  # (n_samples, batch_size)
    mean_pred = torch.mean(predictions, dim=0)
    var_pred = torch.var(predictions, dim=0)
    
    # Add minimum variance to avoid zero uncertainty
    var_pred = var_pred + 1e-6
    
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
    # Higher uncertainty -> lower confidence
    max_std = np.percentile(std, 95)  # Use 95th percentile to avoid outliers
    if max_std > 0:
        normalized_confidence = 1.0 - (std / max_std)
        normalized_confidence = np.clip(normalized_confidence, 0, 1)
    else:
        normalized_confidence = np.ones_like(std)
    
    # Normalize errors to accuracy (higher error -> lower accuracy)
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
    
    # NLL (Negative Log Likelihood) assuming Gaussian
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * uncertainties + 1e-8) + 
        (predictions - targets)**2 / (uncertainties + 1e-8)
    )
    
    # Correlation between uncertainty and error
    correlation = np.corrcoef(std, errors)[0, 1] if len(std) > 1 and np.std(std) > 0 else 0.0
    
    # Coverage
    coverage_68 = np.mean(errors <= std)  # 1 sigma
    coverage_95 = np.mean(errors <= 2 * std)  # 2 sigma
    
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
    """Plot reliability diagram (calibration curve)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bin_accuracies = np.array(metrics['bin_accuracies'])
    bin_confidences = np.array(metrics['bin_confidences'])
    bin_counts = np.array(metrics['bin_counts'])
    
    # Perfect calibration line
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
    
    # Calibration curve
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    ax.plot(bin_centers, bin_accuracies, 'o-', label='Model', linewidth=2, markersize=10, color='blue')
    
    # Add count annotations for bins with data
    for i, (center, acc, conf, count) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            ax.text(center, acc + 0.05, f'n={count}', fontsize=9, ha='center', alpha=0.7)
    
    ax.set_xlabel('Mean Confidence', fontsize=14)
    ax.set_ylabel('Mean Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram (Calibration Curve)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add ECE text
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
    
    # Vary k from 0 to 3
    k_values = np.linspace(0, 3, 200)
    coverages = []
    
    for k in k_values:
        coverage = np.mean(errors <= k * std)
        coverages.append(coverage)
    
    coverages = np.array(coverages)
    
    ax.plot(k_values, coverages, 'b-', linewidth=2, label='Model Coverage', zorder=3)
    
    # Reference lines
    ax.axhline(0.68, color='r', linestyle='--', label='68% Target (1σ)', alpha=0.7, linewidth=1.5)
    ax.axhline(0.95, color='r', linestyle=':', label='95% Target (2σ)', alpha=0.7, linewidth=1.5)
    ax.axvline(1.0, color='g', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(2.0, color='g', linestyle=':', alpha=0.5, linewidth=1)
    
    # Find actual coverage at k=1 and k=2
    coverage_1sigma = np.mean(errors <= 1.0 * std)
    coverage_2sigma = np.mean(errors <= 2.0 * std)
    
    ax.plot([1.0], [coverage_1sigma], 'go', markersize=10, label=f'1σ Coverage = {coverage_1sigma:.1%}', zorder=4)
    ax.plot([2.0], [coverage_2sigma], 'gs', markersize=10, label=f'2σ Coverage = {coverage_2sigma:.1%}', zorder=4)
    
    ax.set_xlabel('Confidence Interval Multiplier (k × σ)', fontsize=14)
    ax.set_ylabel('Coverage Fraction', fontsize=14)
    ax.set_title('Coverage Curve', fontsize=16, fontweight='bold')
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
    """Plot error vs uncertainty scatter with correlation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    std = np.sqrt(np.maximum(uncertainties, 1e-10))
    errors = np.abs(predictions - targets)
    
    # Scatter plot
    ax.scatter(std, errors, alpha=0.4, s=15, edgecolors='none', zorder=2)
    
    # Correlation line
    if len(std) > 1 and np.std(std) > 0:
        z = np.polyfit(std, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(std.min(), std.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Fit (corr={np.corrcoef(std, errors)[0,1]:.3f})', zorder=3)
    
    ax.set_xlabel('Predicted Uncertainty (σ)', fontsize=14)
    ax.set_ylabel('Absolute Error |y - ŷ|', fontsize=14)
    ax.set_title('Error vs Uncertainty', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_vs_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved error vs uncertainty plot")


def main():
    parser = argparse.ArgumentParser(description='Improved Uncertainty Calibration')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--output-dir', type=str, default='artifacts/uncertainty_calibration_improved',
                       help='Output directory for results')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of test samples to evaluate')
    parser.add_argument('--n-mc-samples', type=int, default=10,
                       help='Number of Monte Carlo samples for uncertainty')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--n-bins', type=int, default=10,
                       help='Number of bins for calibration')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  IMPROVED UNCERTAINTY CALIBRATION ANALYSIS")
    print("="*80)
    print()
    
    # Load model
    model, norm_stats = load_model(Path(args.model_path), args.device)
    
    # Load test data
    test_samples = load_test_data(Path(args.test_data), args.max_samples)
    
    # Convert to PyG format
    print("\n🔄 Converting to PyG format...")
    test_pyg = [sample_to_pyg_data(s, norm_stats['mean'], norm_stats['std']) 
                for s in tqdm(test_samples, desc="Converting")]
    test_loader = PyGDataLoader(test_pyg, batch_size=args.batch_size, shuffle=False, 
                               collate_fn=custom_collate_fn)
    
    # Generate predictions with uncertainty
    print(f"\n🔮 Generating predictions with MC dropout (n={args.n_mc_samples})...")
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    model.eval()  # Ensure eval mode for inference
    
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = batch.to(args.device)
        
        # Extract domain_id for FiLM
        if hasattr(batch, 'graph_domain_ids'):
            batch_domain_ids = batch.graph_domain_ids
        else:
            batch_size = batch.batch.max().item() + 1 if hasattr(batch, 'batch') else 1
            batch_domain_ids = [0] * batch_size
        
        domain_id = torch.tensor(batch_domain_ids, dtype=torch.long, device=args.device)
        
        # Uncertainty via input noise perturbation
        mean_pred, var_pred = ensemble_predict_via_noise(model, batch, domain_id, args.n_mc_samples)
        
        # Get targets
        if hasattr(batch, 'energy_target_original'):
            targets = batch.energy_target_original.cpu()
        else:
            # Denormalize
            targets_normalized = batch.energy_target.cpu()
            targets = targets_normalized * norm_stats['std'] + norm_stats['mean']
        
        all_predictions.append(mean_pred.numpy())
        all_targets.append(targets.numpy().flatten())
        all_uncertainties.append(var_pred.numpy().flatten())
    
    # Concatenate
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    uncertainties = np.concatenate(all_uncertainties)
    
    print(f"\n📊 Computing calibration metrics...")
    metrics = compute_ece_regression(predictions, targets, uncertainties, args.n_bins)
    
    # Print results
    print("\n" + "="*80)
    print("  CALIBRATION RESULTS")
    print("="*80)
    print(f"\n📈 Metrics:")
    print(f"   ECE (Expected Calibration Error):  {metrics['ece']:.6f}")
    print(f"   NLL (Negative Log Likelihood):    {metrics['nll']:.6f}")
    print(f"   Uncertainty-Error Correlation:    {metrics['correlation']:.4f}")
    print(f"   Coverage @ 1σ (68%):             {metrics['coverage_68']:.2%}")
    print(f"   Coverage @ 2σ (95%):              {metrics['coverage_95']:.2%}")
    print(f"   Mean Absolute Error:              {metrics['mean_error']:.6f} eV/atom")
    print(f"   Mean Uncertainty:                 {metrics['mean_uncertainty']:.6f} eV/atom")
    print(f"   Median Absolute Error:            {metrics['median_error']:.6f} eV/atom")
    print(f"   Median Uncertainty:               {metrics['median_uncertainty']:.6f} eV/atom")
    
    # Save metrics (convert numpy types to Python native types)
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
    print("  ✅ Calibration analysis complete!")
    print(f"   Results saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

