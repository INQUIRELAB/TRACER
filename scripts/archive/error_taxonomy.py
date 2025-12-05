#!/usr/bin/env python3
"""
Error Taxonomy Analysis
Analyze errors by chemistry type, system size, and domain.

Usage:
    python scripts/error_taxonomy.py \
        --model-path models/gemnet_per_atom/best_model.pt \
        --test-data data/preprocessed_full_unified/test_data.json \
        --output-dir artifacts/error_taxonomy
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
from ase.data import chemical_symbols

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
    
    # Patch _compute_edges
    def patched_compute_edges(positions, batch_indices):
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


def classify_chemistry(atomic_numbers: List[int]) -> Dict[str, any]:
    """Classify chemistry type."""
    elements = set(atomic_numbers)
    element_names = [chemical_symbols[z] for z in elements]
    n_atoms = len(atomic_numbers)
    
    # Transition metals
    tm_z = {22, 23, 24, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 72, 73, 74, 75, 76, 77, 78}
    has_tm = bool(elements & tm_z)
    
    # Alkali/alkaline earth
    alkali_earth = {3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88}
    has_alkali = bool(elements & alkali_earth)
    
    # Halogens
    halogens = {9, 17, 35, 53, 85}
    has_halogen = bool(elements & halogens)
    
    # Oxygen
    has_oxygen = 8 in elements
    
    # Metals (simplified)
    metals_z = {3, 4, 11, 12, 13, 19, 20, 21} | tm_z | {29, 30, 47, 48, 79, 80}
    is_pure_metal = len(elements) == 1 and elements & metals_z
    is_alloy = len(elements & metals_z) > 1 and not has_oxygen
    
    # Oxides
    is_oxide = has_oxygen and has_tm
    is_simple_oxide = has_oxygen and not has_tm
    
    # Organic/C-H systems
    has_carbon = 6 in elements
    has_hydrogen = 1 in elements
    is_organic = has_carbon and has_hydrogen
    
    # Size categories
    if n_atoms < 10:
        size = 'small'
    elif n_atoms < 50:
        size = 'medium'
    else:
        size = 'large'
    
    # Complexity
    n_elements = len(elements)
    if n_elements == 1:
        complexity = 'pure'
    elif n_elements == 2:
        complexity = 'binary'
    elif n_elements <= 4:
        complexity = 'ternary'
    else:
        complexity = 'multinary'
    
    # Chemistry class
    if is_pure_metal:
        chem_class = 'pure_metal'
    elif is_alloy:
        chem_class = 'alloy'
    elif is_oxide:
        chem_class = 'transition_metal_oxide'
    elif is_simple_oxide:
        chem_class = 'simple_oxide'
    elif is_organic:
        chem_class = 'organic'
    elif has_halogen and has_tm:
        chem_class = 'tm_halide'
    elif has_tm:
        chem_class = 'tm_compound'
    else:
        chem_class = 'other'
    
    return {
        'chem_class': chem_class,
        'size': size,
        'complexity': complexity,
        'n_atoms': n_atoms,
        'n_elements': n_elements,
        'has_tm': has_tm,
        'has_oxygen': has_oxygen,
        'is_pure_metal': is_pure_metal,
        'is_oxide': is_oxide,
        'is_organic': is_organic
    }


def analyze_errors_by_category(samples: List[Dict], predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict:
    """Analyze errors by category."""
    errors = np.abs(predictions - targets)
    
    categories = defaultdict(lambda: {'errors': [], 'predictions': [], 'targets': []})
    
    for i, sample in enumerate(samples):
        chem_info = classify_chemistry(sample['atomic_numbers'])
        
        # Categorize by chemistry class
        categories[f"class_{chem_info['chem_class']}"]['errors'].append(errors[i])
        categories[f"class_{chem_info['chem_class']}"]['predictions'].append(predictions[i])
        categories[f"class_{chem_info['chem_class']}"]['targets'].append(targets[i])
        
        # Categorize by size
        categories[f"size_{chem_info['size']}"]['errors'].append(errors[i])
        
        # Categorize by complexity
        categories[f"complexity_{chem_info['complexity']}"]['errors'].append(errors[i])
        
        # Categorize by flags
        if chem_info['has_tm']:
            categories['has_tm']['errors'].append(errors[i])
        if chem_info['is_oxide']:
            categories['is_oxide']['errors'].append(errors[i])
        if chem_info['is_organic']:
            categories['is_organic']['errors'].append(errors[i])
        if chem_info['is_pure_metal']:
            categories['is_pure_metal']['errors'].append(errors[i])
    
    # Compute statistics per category
    category_stats = {}
    for cat, data in categories.items():
        if len(data['errors']) >= 5:  # Minimum samples
            err_array = np.array(data['errors'])
            category_stats[cat] = {
                'mean_error': float(np.mean(err_array)),
                'median_error': float(np.median(err_array)),
                'std_error': float(np.std(err_array)),
                'p90_error': float(np.percentile(err_array, 90)),
                'n_samples': len(data['errors'])
            }
    
    return category_stats


def generate_taxonomy_report(category_stats: Dict, output_path: Path):
    """Generate error taxonomy report."""
    # Sort by mean error
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['mean_error'], reverse=True)
    
    print("\n" + "="*80)
    print("  ERROR TAXONOMY REPORT")
    print("="*80)
    print(f"\n{'Category':<40} {'Mean Error':>15} {'Median Error':>18} {'N Samples':>12}")
    print("-" * 90)
    
    for cat, stats in sorted_cats:
        print(f"{cat:<40} {stats['mean_error']:>15.6f} {stats['median_error']:>18.6f} {stats['n_samples']:>12}")
    
    # Generate mitigation recommendations
    recommendations = []
    
    high_error_cats = [cat for cat, stats in sorted_cats if stats['mean_error'] > 0.1]
    if high_error_cats:
        recommendations.append(f"High-error categories: {', '.join([c.replace('class_', '').replace('size_', '').replace('complexity_', '') for c in high_error_cats[:5]])}")
        recommendations.append("Mitigation: Increase training data diversity for these categories")
    
    # Generate plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Chemistry class errors
    ax = axes[0, 0]
    chem_classes = [(c.replace('class_', ''), s) for c, s in sorted_cats if c.startswith('class_')]
    if chem_classes:
        classes = [c[0] for c in chem_classes[:10]]
        errors = [c[1]['mean_error'] for c in chem_classes[:10]]
        ax.barh(classes, errors, color='steelblue', alpha=0.7)
        ax.set_xlabel('Mean Error (eV/atom)', fontsize=12)
        ax.set_title('Error by Chemistry Class', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Size errors
    ax = axes[0, 1]
    size_cats = [(c.replace('size_', ''), s) for c, s in sorted_cats if c.startswith('size_')]
    if size_cats:
        sizes = [c[0] for c in sorted(size_cats, key=lambda x: ['small', 'medium', 'large'].index(x[0]) if x[0] in ['small', 'medium', 'large'] else 99)]
        errors = [dict(size_cats)[s]['mean_error'] for s in sizes]
        ax.bar(sizes, errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax.set_ylabel('Mean Error (eV/atom)', fontsize=12)
        ax.set_title('Error by System Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Complexity errors
    ax = axes[1, 0]
    complexity_cats = [(c.replace('complexity_', ''), s) for c, s in sorted_cats if c.startswith('complexity_')]
    if complexity_cats:
        complexities = [c[0] for c in sorted(complexity_cats, key=lambda x: ['pure', 'binary', 'ternary', 'multinary'].index(x[0]) if x[0] in ['pure', 'binary', 'ternary', 'multinary'] else 99)]
        errors = [dict(complexity_cats)[c]['mean_error'] for c in complexities]
        ax.bar(complexities, errors, color='coral', alpha=0.7)
        ax.set_ylabel('Mean Error (eV/atom)', fontsize=12)
        ax.set_title('Error by Composition Complexity', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Flag-based comparison
    ax = axes[1, 1]
    flag_cats = {k: v for k, v in category_stats.items() if k in ['has_tm', 'is_oxide', 'is_organic', 'is_pure_metal']}
    if flag_cats:
        flags = list(flag_cats.keys())
        errors = [flag_cats[f]['mean_error'] for f in flags]
        ax.bar(flags, errors, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'], alpha=0.7)
        ax.set_ylabel('Mean Error (eV/atom)', fontsize=12)
        ax.set_title('Error by Chemistry Flags', fontsize=14, fontweight='bold')
        ax.set_xticklabels([f.replace('is_', '').replace('has_', '') for f in flags], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'error_taxonomy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✓ Saved taxonomy plot to {output_path / 'error_taxonomy.png'}")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Error Taxonomy Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--output-dir', type=str, default='artifacts/error_taxonomy',
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
    print("  ERROR TAXONOMY ANALYSIS")
    print("="*80)
    
    # Load model and data
    print(f"\n🔧 Loading model...")
    model = load_model(Path(args.model_path), device=device)
    
    print(f"\n📥 Loading test data...")
    test_samples = load_test_data(Path(args.test_data), max_samples=args.max_samples)
    print(f"   Loaded {len(test_samples)} test samples")
    
    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    predictions = []
    targets = []
    
    for i in tqdm(range(0, len(test_samples), args.batch_size), desc="Predicting"):
        batch_samples = test_samples[i:i+args.batch_size]
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
    
    print(f"   Generated {len(predictions)} predictions")
    
    # Analyze errors by category
    print(f"\n📊 Analyzing errors by category...")
    category_stats = analyze_errors_by_category(test_samples, predictions, targets)
    
    # Generate report
    recommendations = generate_taxonomy_report(category_stats, output_dir)
    
    # Save results
    results = {
        'category_stats': category_stats,
        'overall_mae': float(np.mean(np.abs(predictions - targets))),
        'recommendations': recommendations
    }
    
    with open(output_dir / 'error_taxonomy.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Error taxonomy analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()



