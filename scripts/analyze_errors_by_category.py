#!/usr/bin/env python3
"""
Error Analysis by Category

Analyzes model prediction errors grouped by:
1. Crystal system (cubic, hexagonal, etc.)
2. Composition complexity (binary, ternary, quaternary+)
3. Formation energy range
4. Number of atoms
5. Element types

Creates visualizations and summary tables.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import logging
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_gemnet", 
    Path(__file__).parent.parent / "src" / "gnn" / "model_gemnet.py"
)
model_gemnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_gemnet)
GemNetWrapper = model_gemnet.GemNetWrapper

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_domain_id(domain_str):
    """Map domain string to ID."""
    domain_map = {
        'jarvis_dft': 0,
        'jarvis_elastic': 1,
        'oc20_s2ef': 2,
        'oc22_s2ef': 3,
        'ani1x': 4
    }
    return domain_map.get(domain_str.lower(), 0)


def identify_crystal_system(structure_dict):
    """
    Identify crystal system from structure.
    
    Uses cell parameters to determine crystal system:
    - Cubic: a=b=c, α=β=γ=90°
    - Tetragonal: a=b≠c, α=β=γ=90°
    - Orthorhombic: a≠b≠c, α=β=γ=90°
    - Hexagonal: a=b≠c, α=β=90°, γ=120°
    - Rhombohedral: a=b=c, α=β=γ≠90°
    - Monoclinic: a≠b≠c, α=γ=90°≠β
    - Triclinic: a≠b≠c, α≠β≠γ
    
    Args:
        structure_dict: Dict with 'cell' or 'lattice' key
        
    Returns:
        str: Crystal system name
    """
    if 'cell' in structure_dict and structure_dict['cell'] is not None:
        cell = np.array(structure_dict['cell'])
    elif 'lattice' in structure_dict and structure_dict['lattice'] is not None:
        cell = np.array(structure_dict['lattice'])
    else:
        return 'Unknown'
    
    if cell.shape != (3, 3):
        return 'Unknown'
    
    # Extract lattice parameters
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    
    # Calculate angles
    cos_alpha = np.dot(cell[1], cell[2]) / (b * c)
    cos_beta = np.dot(cell[0], cell[2]) / (a * c)
    cos_gamma = np.dot(cell[0], cell[1]) / (a * b)
    
    alpha = np.arccos(np.clip(cos_alpha, -1, 1)) * 180 / np.pi
    beta = np.arccos(np.clip(cos_beta, -1, 1)) * 180 / np.pi
    gamma = np.arccos(np.clip(cos_gamma, -1, 1)) * 180 / np.pi
    
    # Tolerance for equality
    tol = 1e-3
    angle_tol = 0.1  # degrees
    
    # Determine crystal system
    if (abs(a - b) < tol and abs(b - c) < tol and 
        abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
        return 'Cubic'
    elif (abs(a - b) < tol and abs(a - c) > tol and
          abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
        return 'Tetragonal'
    elif (abs(a - b) > tol and abs(b - c) > tol and abs(a - c) > tol and
          abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 90) < angle_tol):
        return 'Orthorhombic'
    elif (abs(a - b) < tol and abs(a - c) > tol and
          abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol and abs(gamma - 120) < angle_tol):
        return 'Hexagonal'
    elif (abs(a - b) < tol and abs(b - c) < tol and
          abs(alpha - beta) < angle_tol and abs(beta - gamma) < angle_tol and abs(alpha - 90) > angle_tol):
        return 'Rhombohedral'
    elif (abs(alpha - 90) < angle_tol and abs(gamma - 90) < angle_tol and abs(beta - 90) > angle_tol):
        return 'Monoclinic'
    else:
        return 'Triclinic'


def categorize_composition(atomic_numbers):
    """Categorize by number of unique elements."""
    unique_elements = len(set(atomic_numbers))
    if unique_elements == 1:
        return 'Elemental'
    elif unique_elements == 2:
        return 'Binary'
    elif unique_elements == 3:
        return 'Ternary'
    elif unique_elements == 4:
        return 'Quaternary'
    else:
        return f'{unique_elements}-element'


def categorize_energy_range(formation_energy):
    """Categorize by formation energy range."""
    if formation_energy < -2:
        return '< -2 eV/atom'
    elif formation_energy < -1:
        return '-2 to -1 eV/atom'
    elif formation_energy < 0:
        return '-1 to 0 eV/atom'
    elif formation_energy < 1:
        return '0 to 1 eV/atom'
    elif formation_energy < 2:
        return '1 to 2 eV/atom'
    else:
        return '> 2 eV/atom'


def categorize_size(n_atoms):
    """Categorize by number of atoms."""
    if n_atoms < 10:
        return '< 10 atoms'
    elif n_atoms < 20:
        return '10-20 atoms'
    elif n_atoms < 50:
        return '20-50 atoms'
    elif n_atoms < 100:
        return '50-100 atoms'
    else:
        return '> 100 atoms'


def sample_to_pyg_data(sample, norm_mean=None, norm_std=None):
    """Convert sample dict to PyG Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Create edge connectivity
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # Get target
    if 'formation_energy_per_atom' in sample:
        per_atom_energy = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', 0.0)
        n_atoms = len(positions)
        per_atom_energy = energy / n_atoms if abs(energy) > 50 and n_atoms > 0 else energy
    
    # Normalize
    if norm_mean is not None and norm_std is not None:
        per_atom_energy_normalized = (per_atom_energy - norm_mean) / norm_std
    else:
        per_atom_energy_normalized = per_atom_energy
    
    domain_str = sample.get('domain', 'jarvis_dft')
    domain_id = get_domain_id(domain_str)
    n_atoms = len(positions)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=torch.tensor([per_atom_energy_normalized], dtype=torch.float32),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),
        domain_id=torch.tensor([domain_id], dtype=torch.long),
        energy_target_original=torch.tensor([per_atom_energy], dtype=torch.float32),
        # Store original structure info for analysis
        structure_dict=sample  # Store for crystal system identification
    )
    
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


def analyze_errors(model_path: str, test_data_path: str, output_dir: str = 'artifacts/error_analysis'):
    """Perform comprehensive error analysis."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model, norm_stats = load_trained_model(model_path, device)
    logger.info(f"Model loaded (normalization: mean={norm_stats.get('mean', 0):.6f}, std={norm_stats.get('std', 1):.6f})")
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_samples = json.load(f)
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    # Prepare data and get predictions
    logger.info("Computing predictions...")
    predictions = []
    targets = []
    sample_metadata = []
    
    norm_mean = norm_stats.get('mean', 0.0)
    norm_std = norm_stats.get('std', 1.0)
    
    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Evaluating"):
            try:
                pyg_data = sample_to_pyg_data(sample, norm_mean, norm_std)
                pyg_data = pyg_data.to(device)
                batch = Batch.from_data_list([pyg_data])
                
                # Predict
                domain_id = torch.tensor([pyg_data.domain_id.item()], dtype=torch.long, device=device)
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
                
                # Convert to per-atom
                n_atoms = pyg_data.n_atoms.item()
                energy_per_atom_norm = energies_total[0] / n_atoms
                
                # Denormalize
                pred = energy_per_atom_norm.item() * norm_std + norm_mean
                target = pyg_data.energy_target_original.item()
                
                predictions.append(pred)
                targets.append(target)
                
                # Extract metadata for categorization
                atomic_numbers = sample['atomic_numbers']
                n_atoms = len(atomic_numbers)
                
                metadata = {
                    'structure_dict': sample,
                    'atomic_numbers': atomic_numbers,
                    'n_atoms': n_atoms,
                    'formation_energy': target,
                    'prediction': pred,
                    'error': abs(pred - target),
                    'squared_error': (pred - target) ** 2
                }
                sample_metadata.append(metadata)
                
            except Exception as e:
                logger.debug(f"Error processing sample: {e}")
                continue
    
    logger.info(f"Evaluated {len(predictions)} samples")
    
    # Categorize errors
    logger.info("Categorizing errors...")
    
    categories = {
        'crystal_system': defaultdict(list),
        'composition': defaultdict(list),
        'energy_range': defaultdict(list),
        'size': defaultdict(list)
    }
    
    for metadata in tqdm(sample_metadata, desc="Categorizing"):
        sample = metadata['structure_dict']
        error = metadata['error']
        sq_error = metadata['squared_error']
        
        # Crystal system
        crystal_system = identify_crystal_system(sample)
        categories['crystal_system'][crystal_system].append({'error': error, 'sq_error': sq_error})
        
        # Composition
        composition = categorize_composition(metadata['atomic_numbers'])
        categories['composition'][composition].append({'error': error, 'sq_error': sq_error})
        
        # Energy range
        energy_range = categorize_energy_range(metadata['formation_energy'])
        categories['energy_range'][energy_range].append({'error': error, 'sq_error': sq_error})
        
        # Size
        size = categorize_size(metadata['n_atoms'])
        categories['size'][size].append({'error': error, 'sq_error': sq_error})
    
    # Compute statistics for each category
    logger.info("Computing statistics...")
    results = {}
    
    for category_name, category_data in categories.items():
        category_stats = {}
        for subcategory, errors in category_data.items():
            if len(errors) > 0:
                errors_array = np.array([e['error'] for e in errors])
                sq_errors_array = np.array([e['sq_error'] for e in errors])
                
                category_stats[subcategory] = {
                    'count': len(errors),
                    'mae': np.mean(errors_array),
                    'rmse': np.sqrt(np.mean(sq_errors_array)),
                    'median_error': np.median(errors_array),
                    'std_error': np.std(errors_array),
                    'min_error': np.min(errors_array),
                    'max_error': np.max(errors_array),
                    'q25': np.percentile(errors_array, 25),
                    'q75': np.percentile(errors_array, 75),
                    'q90': np.percentile(errors_array, 90),
                    'q95': np.percentile(errors_array, 95)
                }
        
        results[category_name] = category_stats
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    results_json = {}
    for cat_name, cat_stats in results.items():
        results_json[cat_name] = {
            k: {m: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                for m, v in stat.items()} 
            for k, stat in cat_stats.items()
        }
    
    with open(output_path / 'error_analysis_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Results saved to {output_path / 'error_analysis_results.json'}")
    
    # Create summary tables
    logger.info("Creating summary tables...")
    create_summary_tables(results, output_path)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(results, predictions, targets, output_path)
    
    logger.info("✅ Error analysis complete!")
    return results


def create_summary_tables(results, output_path):
    """Create summary tables for each category."""
    
    for category_name, category_stats in results.items():
        # Create DataFrame
        rows = []
        for subcategory, stats in sorted(category_stats.items(), key=lambda x: x[1]['mae']):
            rows.append({
                'Category': subcategory,
                'Count': stats['count'],
                'MAE (eV/atom)': f"{stats['mae']:.6f}",
                'RMSE (eV/atom)': f"{stats['rmse']:.6f}",
                'Median Error': f"{stats['median_error']:.6f}",
                'Std Error': f"{stats['std_error']:.6f}",
                'Q25': f"{stats['q25']:.6f}",
                'Q75': f"{stats['q75']:.6f}",
                'Q95': f"{stats['q95']:.6f}",
                'Max Error': f"{stats['max_error']:.6f}"
            })
        
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_path = output_path / f'error_analysis_{category_name}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved table: {csv_path}")
        
        # Print summary
        print(f"\n{category_name.upper().replace('_', ' ')} ERROR ANALYSIS:")
        print("=" * 100)
        print(df.to_string(index=False))
        print()


def create_visualizations(results, predictions, targets, output_path):
    """Create visualization figures."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    figsize = (10, 6)
    
    # 1. Error distribution histogram
    fig, ax = plt.subplots(figsize=figsize)
    errors = np.abs(np.array(predictions) - np.array(targets))
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Error (eV/atom)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    ax.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path / 'error_distribution.png'}")
    
    # 2. MAE by crystal system
    if 'crystal_system' in results:
        fig, ax = plt.subplots(figsize=(12, 6))
        crystal_stats = results['crystal_system']
        categories = sorted(crystal_stats.keys(), key=lambda x: crystal_stats[x]['mae'])
        mae_values = [crystal_stats[c]['mae'] for c in categories]
        counts = [crystal_stats[c]['count'] for c in categories]
        
        bars = ax.bar(categories, mae_values, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Crystal System')
        ax.set_ylabel('MAE (eV/atom)')
        ax.set_title('MAE by Crystal System')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({mae_values[i]:.4f})',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'mae_by_crystal_system.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_path / 'mae_by_crystal_system.png'}")
    
    # 3. MAE by composition
    if 'composition' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        comp_stats = results['composition']
        categories = sorted(comp_stats.keys(), key=lambda x: comp_stats[x]['mae'])
        mae_values = [comp_stats[c]['mae'] for c in categories]
        counts = [comp_stats[c]['count'] for c in categories]
        
        bars = ax.bar(categories, mae_values, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Composition Type')
        ax.set_ylabel('MAE (eV/atom)')
        ax.set_title('MAE by Composition Complexity')
        ax.tick_params(axis='x', rotation=45)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({mae_values[i]:.4f})',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'mae_by_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_path / 'mae_by_composition.png'}")
    
    # 4. MAE by energy range
    if 'energy_range' in results:
        fig, ax = plt.subplots(figsize=(12, 6))
        energy_stats = results['energy_range']
        # Sort by energy range (custom order)
        energy_order = ['< -2 eV/atom', '-2 to -1 eV/atom', '-1 to 0 eV/atom', 
                       '0 to 1 eV/atom', '1 to 2 eV/atom', '> 2 eV/atom']
        categories = [c for c in energy_order if c in energy_stats]
        mae_values = [energy_stats[c]['mae'] for c in categories]
        counts = [energy_stats[c]['count'] for c in categories]
        
        bars = ax.bar(categories, mae_values, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Formation Energy Range')
        ax.set_ylabel('MAE (eV/atom)')
        ax.set_title('MAE by Formation Energy Range')
        ax.tick_params(axis='x', rotation=45)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({mae_values[i]:.4f})',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'mae_by_energy_range.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved: {output_path / 'mae_by_energy_range.png'}")
    
    # 5. Prediction vs Target scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Target Formation Energy (eV/atom)')
    ax.set_ylabel('Predicted Formation Energy (eV/atom)')
    ax.set_title('Prediction vs Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R²
    r2 = 1 - np.sum((np.array(targets) - np.array(predictions))**2) / np.sum((np.array(targets) - np.mean(targets))**2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f} eV/atom', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / 'prediction_vs_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {output_path / 'prediction_vs_target.png'}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Error analysis by category')
    parser.add_argument('--model-path', type=str,
                       default='models/gemnet_baseline/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--test-data', type=str,
                       default='data/preprocessed_full_unified/test_data.json',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str,
                       default='artifacts/error_analysis',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    analyze_errors(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()


