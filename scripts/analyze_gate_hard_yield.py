#!/usr/bin/env python3
"""
Gate-Hard Yield Analysis
Compare gate-hard selection vs variance-only and random baselines.

Usage:
    python scripts/analyze_gate_hard_yield.py \
        --gate-hard-file artifacts/gate_hard_gemnet/topK_all.jsonl \
        --full-predictions data/predictions_full.jsonl \
        --output-dir artifacts/gate_hard_analysis
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_gate_hard_samples(file_path: Path) -> List[Dict]:
    """Load gate-hard selected samples."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def load_full_predictions(file_path: Path) -> List[Dict]:
    """Load full prediction dataset."""
    if not file_path.exists():
        # If file doesn't exist, try to load from test data and generate predictions
        print(f"⚠️  Full predictions file not found: {file_path}")
        print("   Will generate synthetic comparison from gate-hard data")
        return None
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_absolute_error(pred: float, target: float) -> float:
    """Compute absolute error."""
    return abs(pred - target)


def variance_only_ranking(predictions: List[Dict], k: int) -> List[Dict]:
    """Rank by variance only (baseline)."""
    sorted_by_variance = sorted(
        predictions,
        key=lambda x: x.get('energy_variance', 0.0),
        reverse=True
    )
    return sorted_by_variance[:k]


def random_ranking(predictions: List[Dict], k: int, seed: int = 42) -> List[Dict]:
    """Random ranking (baseline)."""
    np.random.seed(seed)
    indices = np.random.choice(len(predictions), size=min(k, len(predictions)), replace=False)
    return [predictions[i] for i in indices]


def compute_yield_metrics(selected_samples: List[Dict]) -> Dict:
    """Compute yield metrics for selected samples."""
    errors = [compute_absolute_error(s['energy_pred'], s['energy_target']) 
              for s in selected_samples]
    variances = [s.get('energy_variance', 0.0) for s in selected_samples]
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'p90_error': np.percentile(errors, 90),
        'p95_error': np.percentile(errors, 95),
        'mean_variance': np.mean(variances),
        'n_samples': len(selected_samples),
        'errors': errors,
        'variances': variances
    }


def compute_error_vs_yield_curve(predictions: List[Dict], method: str, 
                                 k_values: List[int]) -> Dict:
    """Compute error vs yield curve for different K values."""
    errors_at_k = []
    variances_at_k = []
    
    for k in k_values:
        if method == 'gate_hard':
            # Use existing gate-hard ranking (approximate from samples)
            selected = sorted(
                predictions,
                key=lambda x: x.get('score', x.get('energy_variance', 0.0)),
                reverse=True
            )[:k]
        elif method == 'variance_only':
            selected = variance_only_ranking(predictions, k)
        elif method == 'random':
            selected = random_ranking(predictions, k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        errors = [compute_absolute_error(s['energy_pred'], s['energy_target']) 
                  for s in selected]
        errors_at_k.append(np.mean(errors))
        variances_at_k.append(np.mean([s.get('energy_variance', 0.0) for s in selected]))
    
    return {
        'k_values': k_values,
        'mean_errors': errors_at_k,
        'mean_variances': variances_at_k
    }


def plot_yield_comparison(gate_hard_metrics: Dict, variance_metrics: Dict, 
                         random_metrics: Dict, output_path: Path):
    """Plot comparison of yield metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mean error comparison
    ax = axes[0, 0]
    methods = ['Gate-Hard', 'Variance-Only', 'Random']
    mean_errors = [
        gate_hard_metrics['mean_error'],
        variance_metrics['mean_error'],
        random_metrics['mean_error']
    ]
    ax.bar(methods, mean_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax.set_ylabel('Mean Absolute Error (eV)', fontsize=12)
    ax.set_title('Mean Error Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_errors):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Error distribution boxplot
    ax = axes[0, 1]
    error_data = [
        gate_hard_metrics['errors'],
        variance_metrics['errors'],
        random_metrics['errors']
    ]
    bp = ax.boxplot(error_data, labels=methods, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Absolute Error (eV)', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Percentile errors
    ax = axes[1, 0]
    percentiles = ['P90', 'P95', 'Max']
    gate_hard_percentiles = [
        gate_hard_metrics['p90_error'],
        gate_hard_metrics['p95_error'],
        gate_hard_metrics['max_error']
    ]
    variance_percentiles = [
        variance_metrics['p90_error'],
        variance_metrics['p95_error'],
        variance_metrics['max_error']
    ]
    random_percentiles = [
        random_metrics['p90_error'],
        random_metrics['p95_error'],
        random_metrics['max_error']
    ]
    x = np.arange(len(percentiles))
    width = 0.25
    ax.bar(x - width, gate_hard_percentiles, width, label='Gate-Hard', 
           color='#1f77b4', alpha=0.7)
    ax.bar(x, variance_percentiles, width, label='Variance-Only', 
           color='#ff7f0e', alpha=0.7)
    ax.bar(x + width, random_percentiles, width, label='Random', 
           color='#2ca02c', alpha=0.7)
    ax.set_xlabel('Error Percentile', fontsize=12)
    ax.set_ylabel('Absolute Error (eV)', fontsize=12)
    ax.set_title('High-Percentile Error Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Variance comparison
    ax = axes[1, 1]
    mean_variances = [
        gate_hard_metrics['mean_variance'],
        variance_metrics['mean_variance'],
        random_metrics['mean_variance']
    ]
    ax.bar(methods, mean_variances, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax.set_ylabel('Mean Variance', fontsize=12)
    ax.set_title('Uncertainty (Variance) Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_variances):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'yield_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved yield comparison to {output_path / 'yield_comparison.png'}")


def plot_error_vs_yield_curves(curves: Dict[str, Dict], output_path: Path):
    """Plot error vs yield curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'gate_hard': '#1f77b4', 'variance_only': '#ff7f0e', 'random': '#2ca02c'}
    labels = {'gate_hard': 'Gate-Hard', 'variance_only': 'Variance-Only', 'random': 'Random'}
    
    for method, curve_data in curves.items():
        k_values = curve_data['k_values']
        mean_errors = curve_data['mean_errors']
        ax.plot(k_values, mean_errors, 'o-', label=labels[method], 
               color=colors[method], linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Samples (K)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (eV)', fontsize=12)
    ax.set_title('Error vs Yield Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_vs_yield_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved error vs yield curves to {output_path / 'error_vs_yield_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='Gate-Hard Yield Analysis')
    parser.add_argument('--gate-hard-file', type=str, required=True,
                       help='Path to gate-hard selected samples JSONL')
    parser.add_argument('--full-predictions', type=str, default=None,
                       help='Path to full predictions JSONL (optional)')
    parser.add_argument('--output-dir', type=str, default='artifacts/gate_hard_analysis',
                       help='Output directory')
    parser.add_argument('--k-values', type=int, nargs='+', 
                       default=[50, 100, 150, 200, 250, 270],
                       help='K values for yield curves')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  GATE-HARD YIELD ANALYSIS")
    print("="*80)
    
    # Load gate-hard samples
    print(f"\n📥 Loading gate-hard samples from {args.gate_hard_file}...")
    gate_hard_samples = load_gate_hard_samples(Path(args.gate_hard_file))
    k_gate_hard = len(gate_hard_samples)
    print(f"   Loaded {k_gate_hard} gate-hard samples")
    
    # Load or generate full predictions
    if args.full_predictions and Path(args.full_predictions).exists():
        print(f"\n📥 Loading full predictions from {args.full_predictions}...")
        full_predictions = load_full_predictions(Path(args.full_predictions))
        print(f"   Loaded {len(full_predictions)} full predictions")
    else:
        print(f"\n⚠️  Full predictions not available. Using gate-hard samples as pool.")
        print(f"   (This limits comparison but still provides baseline metrics)")
        full_predictions = gate_hard_samples + random_ranking(
            gate_hard_samples, min(1000, len(gate_hard_samples) * 5), seed=999
        )
    
    # Compute metrics for each method
    print(f"\n📊 Computing yield metrics...")
    
    # Gate-hard metrics (use all samples)
    gate_hard_metrics = compute_yield_metrics(gate_hard_samples)
    
    # Variance-only baseline
    variance_selected = variance_only_ranking(full_predictions, k_gate_hard)
    variance_metrics = compute_yield_metrics(variance_selected)
    
    # Random baseline
    random_selected = random_ranking(full_predictions, k_gate_hard, seed=42)
    random_metrics = compute_yield_metrics(random_selected)
    
    # Print comparison table
    print("\n" + "="*80)
    print("  YIELD METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {'Gate-Hard':>15} {'Variance-Only':>18} {'Random':>15}")
    print("-" * 80)
    print(f"{'Mean Error (eV)':<30} {gate_hard_metrics['mean_error']:>15.4f} {variance_metrics['mean_error']:>18.4f} {random_metrics['mean_error']:>15.4f}")
    print(f"{'Median Error (eV)':<30} {gate_hard_metrics['median_error']:>15.4f} {variance_metrics['median_error']:>18.4f} {random_metrics['median_error']:>15.4f}")
    print(f"{'Std Error (eV)':<30} {gate_hard_metrics['std_error']:>15.4f} {variance_metrics['std_error']:>18.4f} {random_metrics['std_error']:>15.4f}")
    print(f"{'P90 Error (eV)':<30} {gate_hard_metrics['p90_error']:>15.4f} {variance_metrics['p90_error']:>18.4f} {random_metrics['p90_error']:>15.4f}")
    print(f"{'P95 Error (eV)':<30} {gate_hard_metrics['p95_error']:>15.4f} {variance_metrics['p95_error']:>18.4f} {random_metrics['p95_error']:>15.4f}")
    print(f"{'Max Error (eV)':<30} {gate_hard_metrics['max_error']:>15.4f} {variance_metrics['max_error']:>18.4f} {random_metrics['max_error']:>15.4f}")
    print(f"{'Mean Variance':<30} {gate_hard_metrics['mean_variance']:>15.4f} {variance_metrics['mean_variance']:>18.4f} {random_metrics['mean_variance']:>15.4f}")
    
    # Compute improvement
    improvement_vs_variance = ((variance_metrics['mean_error'] - gate_hard_metrics['mean_error']) 
                              / variance_metrics['mean_error'] * 100)
    improvement_vs_random = ((random_metrics['mean_error'] - gate_hard_metrics['mean_error']) 
                            / random_metrics['mean_error'] * 100)
    
    print("\n" + "="*80)
    print("  IMPROVEMENT OVER BASELINES")
    print("="*80)
    print(f"Gate-Hard vs Variance-Only: {improvement_vs_variance:+.2f}% error reduction")
    print(f"Gate-Hard vs Random: {improvement_vs_random:+.2f}% error reduction")
    
    # Compute error vs yield curves
    print(f"\n📈 Computing error vs yield curves...")
    curves = {}
    for method in ['gate_hard', 'variance_only', 'random']:
        if method == 'gate_hard':
            # Use gate-hard samples with score-based ranking
            pool = sorted(full_predictions, 
                         key=lambda x: x.get('score', x.get('energy_variance', 0.0)), 
                         reverse=True)
        else:
            pool = full_predictions
        curves[method] = compute_error_vs_yield_curve(pool, method, args.k_values)
    
    # Generate plots
    print(f"\n📊 Generating comparison plots...")
    plot_yield_comparison(gate_hard_metrics, variance_metrics, random_metrics, output_dir)
    plot_error_vs_yield_curves(curves, output_dir)
    
    # Save metrics
    metrics_summary = {
        'gate_hard': {
            'mean_error': float(gate_hard_metrics['mean_error']),
            'median_error': float(gate_hard_metrics['median_error']),
            'p90_error': float(gate_hard_metrics['p90_error']),
            'p95_error': float(gate_hard_metrics['p95_error']),
            'max_error': float(gate_hard_metrics['max_error']),
            'mean_variance': float(gate_hard_metrics['mean_variance']),
            'n_samples': gate_hard_metrics['n_samples']
        },
        'variance_only': {
            'mean_error': float(variance_metrics['mean_error']),
            'median_error': float(variance_metrics['median_error']),
            'p90_error': float(variance_metrics['p90_error']),
            'p95_error': float(variance_metrics['p95_error']),
            'max_error': float(variance_metrics['max_error']),
            'mean_variance': float(variance_metrics['mean_variance']),
            'n_samples': variance_metrics['n_samples']
        },
        'random': {
            'mean_error': float(random_metrics['mean_error']),
            'median_error': float(random_metrics['median_error']),
            'p90_error': float(random_metrics['p90_error']),
            'p95_error': float(random_metrics['p95_error']),
            'max_error': float(random_metrics['max_error']),
            'mean_variance': float(random_metrics['mean_variance']),
            'n_samples': random_metrics['n_samples']
        },
        'improvements': {
            'vs_variance_percent': float(improvement_vs_variance),
            'vs_random_percent': float(improvement_vs_random)
        }
    }
    
    with open(output_dir / 'yield_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n✅ Gate-hard yield analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()



