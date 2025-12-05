#!/usr/bin/env python3
"""
Robustness Ablation Studies
Test model performance under different configurations:
1. Single model vs ensemble
2. Different test subsets (by domain, size, chemistry)
3. OOD stress tests (metals, oxides)

Usage:
    python scripts/robustness_ablations.py \
        --model-path models/gemnet_per_atom/best_model.pt \
        --test-data data/preprocessed_full_unified/test_data.json \
        --output-dir artifacts/robustness_ablations
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid MACE dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("model_gemnet", str(Path(__file__).parent.parent / "src" / "gnn" / "model_gemnet.py"))
model_gemnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_gemnet)
GemNetWrapper = model_gemnet.GemNetWrapper

from src.graphs.periodic_graph import PeriodicGraph
from torch_geometric.data import Batch


def load_test_data(test_file: Path, max_samples: int = None) -> List[Dict]:
    """Load test dataset."""
    with open(test_file, 'r') as f:
        samples = json.load(f)
    if max_samples:
        samples = samples[:max_samples]
    return samples


def create_graph_from_sample(sample: Dict):
    """Create PyTorch Geometric graph from sample."""
    atomic_numbers = np.array(sample['atomic_numbers'], dtype=np.int64)
    positions = np.array(sample['positions'], dtype=np.float32)
    
    if 'cell' in sample and sample['cell']:
        cell = np.array(sample['cell'], dtype=np.float32)
    else:
        if len(positions) > 0:
            max_dist = float(np.max(positions) - np.min(positions)) + 10.0
        else:
            max_dist = 10.0
        cell = np.eye(3, dtype=np.float32) * max_dist
    
    graph_builder = PeriodicGraph(cutoff_radius=10.0, max_neighbors=100)
    graph = graph_builder.build_graph(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell_vectors=cell
    )
    return graph


def load_model(model_path: Path, device: str = 'cuda') -> GemNetWrapper:
    """Load trained GemNet model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
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
    
    if 'embedding.embedding.weight' in state_dict:
        num_atoms = state_dict['embedding.embedding.weight'].shape[0]
    else:
        num_atoms = model_config.get('num_atoms', 100)
    
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
    def patched_compute_edges(positions, batch_indices):
        n_atoms = len(positions)
        distances = torch.cdist(positions, positions)
        edge_mask = (distances < model.cutoff) & (distances > 1e-8)
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        row, col = edge_index
        edge_vec = positions[row] - positions[col]
        distances = torch.norm(edge_vec, dim=-1)
        edge_attr = edge_vec / (distances.unsqueeze(-1) + 1e-8)
        return edge_index, edge_attr, distances
    
    model._compute_edges = patched_compute_edges
    
    return model


def predict_batch(model: GemNetWrapper, graphs: List, device: str) -> np.ndarray:
    """Predict energies for a batch of graphs."""
    batch = Batch.from_data_list(graphs).to(device)
    
    with torch.no_grad():
        pred, _, _ = model(batch, compute_forces=False)
        if model.mean is not None and model.std is not None:
            pred = model.denormalize_energy(pred)
    
    return pred.cpu().numpy()


def categorize_sample(sample: Dict) -> Dict[str, str]:
    """Categorize sample by chemistry, size, etc."""
    atomic_numbers = sample['atomic_numbers']
    n_atoms = len(atomic_numbers)
    
    # Transition metals
    tm_elements = [22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78]
    has_tm = any(z in tm_elements for z in atomic_numbers)
    
    # Metal vs non-metal
    metals = [3, 4, 11, 12, 13, 19, 20, 21] + tm_elements + [29, 30, 47, 48, 79, 80]
    is_metal = all(z in metals for z in atomic_numbers)
    
    # Oxide detection (simple heuristic)
    has_oxygen = 8 in atomic_numbers
    is_oxide = has_oxygen and has_tm
    
    # Size categories
    if n_atoms < 10:
        size_cat = 'small'
    elif n_atoms < 50:
        size_cat = 'medium'
    else:
        size_cat = 'large'
    
    return {
        'has_tm': has_tm,
        'is_metal': is_metal,
        'is_oxide': is_oxide,
        'has_oxygen': has_oxygen,
        'size': size_cat,
        'n_atoms': n_atoms
    }


def evaluate_on_subset(samples: List[Dict], model: GemNetWrapper, 
                       device: str, batch_size: int = 32) -> Dict:
    """Evaluate model on a subset of samples."""
    predictions = []
    targets = []
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        graphs = [create_graph_from_sample(s) for s in batch_samples]
        
        pred = predict_batch(model, graphs, device)
        
        for j, s in enumerate(batch_samples):
            energy = s.get('energy', s.get('energy_target', 0.0))
            n_atoms = len(s['atomic_numbers'])
            energy_per_atom = energy / n_atoms if n_atoms > 0 else 0.0
            
            predictions.append(pred[j])
            targets.append(energy_per_atom)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = np.abs(predictions - targets)
    
    return {
        'mae': float(np.mean(errors)),
        'rmse': float(np.sqrt(np.mean((predictions - targets)**2))),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'p90_error': float(np.percentile(errors, 90)),
        'p95_error': float(np.percentile(errors, 95)),
        'max_error': float(np.max(errors)),
        'n_samples': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description='Robustness Ablation Studies')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--output-dir', type=str, default='artifacts/robustness_ablations',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("  ROBUSTNESS ABLATION STUDIES")
    print("="*80)
    
    # Load model and data
    print(f"\n🔧 Loading model from {args.model_path}...")
    model = load_model(Path(args.model_path), device=device)
    
    print(f"\n📥 Loading test data from {args.test_data}...")
    test_samples = load_test_data(Path(args.test_data), max_samples=args.max_samples)
    print(f"   Loaded {len(test_samples)} test samples")
    
    # Categorize samples
    print(f"\n📊 Categorizing samples...")
    categorized = defaultdict(list)
    for sample in test_samples:
        cats = categorize_sample(sample)
        for key, value in cats.items():
            if isinstance(value, bool):
                if value:
                    categorized[key].append(sample)
            elif isinstance(value, str):
                categorized[f"{key}_{value}"].append(sample)
    
    # Evaluate on full test set
    print(f"\n🧪 Evaluating on full test set...")
    full_metrics = evaluate_on_subset(test_samples, model, device, args.batch_size)
    
    # Evaluate on categorized subsets
    subset_metrics = {}
    print(f"\n🧪 Evaluating on categorized subsets...")
    
    for category, subset_samples in categorized.items():
        if len(subset_samples) < 5:  # Skip very small subsets
            continue
        print(f"   {category}: {len(subset_samples)} samples")
        metrics = evaluate_on_subset(subset_samples, model, device, args.batch_size)
        subset_metrics[category] = metrics
    
    # Print results
    print("\n" + "="*80)
    print("  ROBUSTNESS ABLATION RESULTS")
    print("="*80)
    
    print(f"\n{'Full Test Set':<30} {'MAE (eV/atom)':>15} {'RMSE (eV/atom)':>18} {'N Samples':>12}")
    print("-" * 80)
    print(f"{'All samples':<30} {full_metrics['mae']:>15.6f} {full_metrics['rmse']:>18.6f} {full_metrics['n_samples']:>12}")
    
    print(f"\n{'Subset':<30} {'MAE (eV/atom)':>15} {'RMSE (eV/atom)':>18} {'N Samples':>12}")
    print("-" * 80)
    for category, metrics in sorted(subset_metrics.items(), key=lambda x: x[1]['mae'], reverse=True):
        print(f"{category:<30} {metrics['mae']:>15.6f} {metrics['rmse']:>18.6f} {metrics['n_samples']:>12}")
    
    # Generate comparison plot
    print(f"\n📈 Generating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MAE comparison by category
    ax = axes[0, 0]
    categories = list(subset_metrics.keys())
    mae_values = [subset_metrics[c]['mae'] for c in categories]
    ax.barh(categories, mae_values, color='steelblue', alpha=0.7)
    ax.set_xlabel('MAE (eV/atom)', fontsize=12)
    ax.set_title('MAE by Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Error distribution by size
    ax = axes[0, 1]
    size_cats = ['size_small', 'size_medium', 'size_large']
    size_data = [subset_metrics.get(c, {}).get('mae', 0) for c in size_cats]
    size_labels = ['Small\n(<10)', 'Medium\n(10-50)', 'Large\n(>50)']
    ax.bar(size_labels, size_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax.set_ylabel('MAE (eV/atom)', fontsize=12)
    ax.set_title('Error by System Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Error by chemistry
    ax = axes[1, 0]
    chem_cats = ['is_metal', 'is_oxide', 'has_tm']
    chem_data = [subset_metrics.get(c, {}).get('mae', 0) for c in chem_cats]
    chem_labels = ['Pure\nMetals', 'Oxides\n(TM+O)', 'Transition\nMetals']
    ax.bar(chem_labels, chem_data, color=['#9467bd', '#8c564b', '#e377c2'], alpha=0.7)
    ax.set_ylabel('MAE (eV/atom)', fontsize=12)
    ax.set_title('Error by Chemistry Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Error percentiles
    ax = axes[1, 1]
    percentiles = ['median_error', 'p90_error', 'p95_error', 'max_error']
    percentile_labels = ['Median', 'P90', 'P95', 'Max']
    full_percentiles = [full_metrics[p] for p in percentiles]
    ax.plot(percentile_labels, full_percentiles, 'o-', linewidth=2, markersize=8, label='Full Test Set')
    ax.set_ylabel('Error (eV/atom)', fontsize=12)
    ax.set_title('Error Distribution (Percentiles)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_ablations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved ablation plots to {output_dir / 'robustness_ablations.png'}")
    
    # Save metrics
    results = {
        'full_test': full_metrics,
        'subsets': subset_metrics
    }
    
    with open(output_dir / 'robustness_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Robustness ablation studies complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()



