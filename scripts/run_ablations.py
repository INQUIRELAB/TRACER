#!/usr/bin/env python3
"""
Ablation Studies for GemNet Pipeline
Tests contribution of each component: FiLM, domain embeddings, gate-hard components.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm

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
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    use_film: bool = True
    use_domain_embedding: bool = True
    film_on_readout: bool = True
    domain_embedding_dim: int = 16


def load_test_data(test_data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test data for evaluation."""
    logger.info(f"Loading test data from {test_data_path}")
    
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"Loaded {len(data)} test samples")
    return data


def sample_to_pyg_data(sample: Dict, norm_stats: Dict) -> Data:
    """Convert sample to PyTorch Geometric Data object."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    # Get energy target
    if 'formation_energy_per_atom' in sample:
        energy_per_atom = sample['formation_energy_per_atom']
    else:
        energy = sample.get('energy', 0.0)
        n_atoms = len(atomic_numbers)
        # Heuristic: if energy is large (>50 eV), assume total; otherwise per-atom
        if abs(energy) > 50 and n_atoms > 0:
            energy_per_atom = energy / n_atoms
        else:
            energy_per_atom = energy  # Already per-atom
    
    # Normalize energy
    norm_mean = norm_stats.get('mean', 0.0)
    norm_std = norm_stats.get('std', 1.0)
    energy_normalized = (energy_per_atom - norm_mean) / norm_std
    
    # Create graph using distance-based edges (fallback for torch-cluster)
    cutoff = 10.0
    distances_matrix = torch.cdist(positions, positions)
    edge_mask = (distances_matrix < cutoff) & (distances_matrix > 1e-8)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    # Get domain ID (default to JARVIS-DFT = 0)
    domain_id = sample.get('domain_id', 0)
    n_atoms = len(atomic_numbers)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        energy_target=torch.tensor([energy_normalized], dtype=torch.float32),
        energy_target_original=torch.tensor([energy_per_atom], dtype=torch.float32),
        domain_id=torch.tensor([domain_id], dtype=torch.long),
        n_atoms=torch.tensor([n_atoms], dtype=torch.long)
    )
    
    # Add cell if available
    if 'cell' in sample and sample['cell'] is not None:
        data.cell = torch.tensor(sample['cell'], dtype=torch.float32)
    
    return data


def evaluate_model(model: GemNetWrapper, test_data: List[Dict], 
                  norm_stats: Dict, device: torch.device, 
                  config: AblationConfig) -> Dict[str, float]:
    """Evaluate model on test data."""
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc=f"Evaluating {config.name}"):
            try:
                data = sample_to_pyg_data(sample, norm_stats)
                data = data.to(device)
                
                # Create batch
                batch = Batch.from_data_list([data])
                
                # Extract domain_id properly
                domain_id = None
                if hasattr(batch, 'domain_id'):
                    domain_id_tensor = batch.domain_id
                    if domain_id_tensor is not None and len(domain_id_tensor) > 0:
                        domain_id = domain_id_tensor
                    else:
                        domain_id = torch.tensor([0], dtype=torch.long, device=device)
                else:
                    domain_id = torch.tensor([0], dtype=torch.long, device=device)
                
                # Predict
                energies_total, _, _ = model(batch, compute_forces=False, domain_id=domain_id)
                
                # Model outputs total energy, convert to per-atom
                n_atoms = len(data.atomic_numbers)
                energy_per_atom_normalized = energies_total.item() / n_atoms
                
                # Denormalize
                norm_mean = norm_stats.get('mean', 0.0)
                norm_std = norm_stats.get('std', 1.0)
                energy_per_atom_pred = energy_per_atom_normalized * norm_std + norm_mean
                
                # Get target
                target = data.energy_target_original.item()
                
                predictions.append(energy_per_atom_pred)
                targets.append(target)
                
            except Exception as e:
                logger.debug(f"Error evaluating sample: {e}")
                continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(predictions)
    }


def load_trained_model(model_path: str, device: torch.device) -> Tuple[GemNetWrapper, Dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Get model config from checkpoint or use defaults
    model_config = checkpoint.get('model_config', {})
    norm_stats = checkpoint.get('normalization', {})
    
    # Get model architecture parameters
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
    
    # Load weights
    try:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"  ✓ Loaded model weights successfully")
    except Exception as e:
        logger.error(f"  ✗ Could not load state dict: {e}")
        raise
    
    model.eval()
    return model, norm_stats


def run_ablations():
    """Run ablation studies."""
    logger.info("=" * 80)
    logger.info("  ABLATION STUDIES")
    logger.info("=" * 80)
    logger.info("")
    
    # Configuration - paths to trained models
    # NOTE: Using 'fixed' version of full model which has correct normalization stats
    model_paths = {
        "Baseline (No FiLM, No Domain Embedding)": "models/gemnet_baseline/best_model.pt",
        "Domain Embedding Only (No FiLM)": "models/gemnet_domain_only/best_model.pt",
        "Full Model (FiLM + Domain Embedding)": "models/gemnet_per_atom_fixed/best_model.pt"
    }
    
    test_data_path = "data/preprocessed_full_unified/test_data.json"
    output_dir = Path("artifacts/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("")
    
    # Load test data (use full test set for proper evaluation)
    test_data = load_test_data(test_data_path, max_samples=None)
    logger.info("")
    
    # Define ablation configurations with model paths
    ablations = [
        {
            "name": "Baseline (No FiLM, No Domain Embedding)",
            "description": "GemNet without FiLM or domain embeddings",
            "model_path": model_paths["Baseline (No FiLM, No Domain Embedding)"]
        },
        {
            "name": "Domain Embedding Only (No FiLM)",
            "description": "GemNet with domain embeddings but no FiLM modulation",
            "model_path": model_paths["Domain Embedding Only (No FiLM)"]
        },
        {
            "name": "Full Model (FiLM + Domain Embedding)",
            "description": "Complete model with FiLM and domain embeddings",
            "model_path": model_paths["Full Model (FiLM + Domain Embedding)"]
        },
    ]
    
    results = {}
    
    for ablation in ablations:
        logger.info("=" * 80)
        logger.info(f"Evaluating: {ablation['name']}")
        logger.info(f"  Description: {ablation['description']}")
        logger.info(f"  Model path: {ablation['model_path']}")
        
        if not Path(ablation['model_path']).exists():
            logger.error(f"  ✗ Model checkpoint not found: {ablation['model_path']}")
            results[ablation['name']] = {'error': f"Model checkpoint not found: {ablation['model_path']}"}
            continue
        
        try:
            # Load trained model
            model, norm_stats = load_trained_model(ablation['model_path'], device)
            logger.info(f"  ✓ Model loaded (mean={norm_stats.get('mean', 0):.6f}, std={norm_stats.get('std', 1):.6f})")
            
            # Create config object for evaluation
            config = AblationConfig(
                name=ablation['name'],
                description=ablation['description'],
                use_film="FiLM" in ablation['name'],
                use_domain_embedding="Domain" in ablation['name'] or "Full" in ablation['name']
            )
            
            # Evaluate
            logger.info(f"  Evaluating on {len(test_data)} test samples...")
            metrics = evaluate_model(model, test_data, norm_stats, device, config)
            
            results[ablation['name']] = {
                'config': config.__dict__,
                'metrics': metrics,
                'model_path': ablation['model_path']
            }
            
            logger.info(f"")
            logger.info(f"  Results:")
            logger.info(f"    MAE: {metrics['mae']:.6f} eV/atom")
            logger.info(f"    RMSE: {metrics['rmse']:.6f} eV/atom")
            logger.info(f"    R²: {metrics['r2']:.6f}")
            logger.info(f"    N samples: {metrics['n_samples']}")
            
        except Exception as e:
            logger.error(f"  ✗ Error evaluating {ablation['name']}: {e}")
            import traceback
            traceback.print_exc()
            results[ablation['name']] = {'error': str(e)}
    
    # Save results
    output_file = output_dir / "ablation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("")
    logger.info(f"✅ Results saved to {output_file}")
    
    # Generate summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("  SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    
    for name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info(f"{name}:")
            logger.info(f"  MAE: {metrics['mae']:.6f} eV/atom")
            logger.info(f"  R²: {metrics['r2']:.6f}")
            logger.info("")


if __name__ == "__main__":
    run_ablations()

