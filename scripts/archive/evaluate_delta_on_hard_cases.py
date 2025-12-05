#!/usr/bin/env python3
"""
Evaluate Delta Head on Hard Cases Only
Tests if quantum corrections help on the gate-hard selected samples
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_norm_stats():
    """Load normalization statistics."""
    train_file = Path("data/preprocessed_full_unified/train_data.json")
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
            return {'mean': np.mean(all_energies), 'std': np.std(all_energies)}
    return {'mean': 0.0, 'std': 1.0}


def load_gemnet_model(norm_stats: dict):
    """Load trained GemNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load("models/gemnet_per_atom/best_model.pt", map_location='cpu')
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
    
    logger.info(f"✅ Loaded GemNet model on {device}")
    return model, device


def load_delta_head(device):
    """Load trained delta head."""
    checkpoint_path = Path("artifacts/delta_head_final/best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config')
    if config is None:
        config = DeltaHeadConfig()
    
    model = DeltaHead(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✅ Loaded Delta Head model")
    return model, config


def load_gate_hard_samples():
    """Load gate-hard selected samples."""
    gate_hard_file = Path("artifacts/gate_hard_full/topK_all.jsonl")
    samples = []
    
    with open(gate_hard_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    logger.info(f"✅ Loaded {len(samples)} gate-hard samples")
    return samples


def find_matching_preprocessed_sample(gate_sample: Dict, preprocessed_samples: Dict[str, Dict]) -> Dict:
    """Find matching preprocessed sample by sample_id."""
    sample_id = gate_sample.get('sample_id', '')
    
    # Try exact match first
    if sample_id in preprocessed_samples:
        return preprocessed_samples[sample_id]
    
    # Try domain matching
    domain = gate_sample.get('domain', '')
    for key, sample in preprocessed_samples.items():
        if sample.get('domain') == domain and abs(sample.get('energy', 0) - gate_sample.get('energy_target', 0)) < 1.0:
            return sample
    
    return None


def load_preprocessed_samples():
    """Load all preprocessed samples indexed by sample_id."""
    data_path = Path("data/preprocessed_full_unified")
    samples = {}
    
    for split in ['train', 'val', 'test']:
        split_file = data_path / f'{split}_data.json'
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_samples = json.load(f)
                for sample in split_samples:
                    sample_id = sample.get('sample_id', f"{sample.get('domain', 'unknown')}_{len(samples)}")
                    samples[sample_id] = sample
    
    logger.info(f"✅ Loaded {len(samples)} preprocessed samples")
    return samples


def extract_schnet_features(gnn_model, sample: Dict, device):
    """Extract SchNet features from GemNet model."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        x=atomic_numbers.unsqueeze(1).float()
    ).to(device)
    
    # Extract embeddings
    if hasattr(gnn_model, 'model') and hasattr(gnn_model.model, 'embedding'):
        embedding_layer = gnn_model.model.embedding
    elif hasattr(gnn_model, 'embedding'):
        embedding_layer = gnn_model.embedding
    else:
        raise AttributeError("GNN model does not have accessible embedding layer")
    
    with torch.no_grad():
        embeddings = embedding_layer(data.atomic_numbers)
        pooled_features = embeddings.sum(dim=0, keepdim=True)
        
        # Resize to 256
        target_dim = 256
        if pooled_features.size(1) > target_dim:
            pooled_features = pooled_features[:, :target_dim]
        elif pooled_features.size(1) < target_dim:
            padding = torch.zeros(1, target_dim - pooled_features.size(1), device=device)
            pooled_features = torch.cat([pooled_features, padding], dim=1)
    
    return pooled_features


def predict_with_gemnet(gnn_model, sample: Dict, device, norm_stats):
    """Predict energy with GemNet."""
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import radius_graph
    
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    edge_index = radius_graph(positions, r=10.0, max_num_neighbors=32)
    
    data = PyGData(
        atomic_numbers=atomic_numbers,
        pos=positions,
        edge_index=edge_index,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long)
    ).to(device)
    
    with torch.no_grad():
        energy_total, _, _ = gnn_model(data, compute_forces=False)
        n_atoms = len(atomic_numbers)
        energy_per_atom = energy_total / n_atoms
        
        # Denormalize
        if gnn_model.mean is not None and gnn_model.std is not None:
            energy_per_atom = energy_per_atom * gnn_model.std + gnn_model.mean
        
        return energy_per_atom.item() * n_atoms  # Return total energy


def get_domain_id(domain: str) -> int:
    """Convert domain to ID."""
    domain_map = {
        'jarvis_dft': 0,
        'jarvis_elastic': 1,
        'oc20_s2ef': 2,
        'oc22_s2ef': 3,
        'ani1x': 4
    }
    return domain_map.get(domain, 0)


def evaluate_on_hard_cases():
    """Evaluate baseline vs delta-corrected on hard cases."""
    logger.info("🚀 EVALUATING DELTA HEAD ON HARD CASES")
    logger.info("="*80)
    
    # Load data
    norm_stats = load_norm_stats()
    gemnet_model, device = load_gemnet_model(norm_stats)
    delta_head, delta_config = load_delta_head(device)
    gate_hard_samples = load_gate_hard_samples()
    preprocessed_samples = load_preprocessed_samples()
    
    # Evaluate
    baseline_results = []
    delta_results = []
    
    matched_count = 0
    failed_count = 0
    
    logger.info("🔄 Evaluating on hard cases...")
    
    for gate_sample in tqdm(gate_hard_samples, desc="Processing"):
        sample_id = gate_sample.get('sample_id', '')
        domain = gate_sample.get('domain', 'jarvis_dft')
        
        # Find matching preprocessed sample
        matched_sample = find_matching_preprocessed_sample(gate_sample, preprocessed_samples)
        
        if matched_sample is None:
            failed_count += 1
            continue
        
        if 'atomic_numbers' not in matched_sample:
            failed_count += 1
            continue
        
        try:
            n_atoms = len(matched_sample['atomic_numbers'])
            if n_atoms == 0:
                failed_count += 1
                continue
            
            # Get target energy (from gate-hard sample or matched sample)
            energy_target_total = gate_sample.get('energy_target', matched_sample.get('energy', 0.0))
            energy_target_per_atom = energy_target_total / n_atoms
            
            # Baseline prediction
            energy_baseline_total = predict_with_gemnet(gemnet_model, matched_sample, device, norm_stats)
            energy_baseline_per_atom = energy_baseline_total / n_atoms
            baseline_error_per_atom = abs(energy_baseline_per_atom - energy_target_per_atom)
            
            # Apply delta correction
            schnet_features = extract_schnet_features(gemnet_model, matched_sample, device)
            domain_id = torch.tensor([get_domain_id(domain)], dtype=torch.long).to(device)
            
            delta_output = delta_head(schnet_features, domain_id)
            if isinstance(delta_output, dict):
                delta_per_atom = delta_output.get('delta_energy', 0.0)
                if isinstance(delta_per_atom, torch.Tensor):
                    delta_per_atom = delta_per_atom.squeeze().item()
                else:
                    delta_per_atom = float(delta_per_atom)
            elif isinstance(delta_output, torch.Tensor):
                delta_per_atom = delta_output.squeeze().item()
            else:
                delta_per_atom = float(delta_output)
            
            # Apply conservative gating (same as in pipeline)
            uncertainty_per_atom = np.sqrt(gate_sample.get('energy_variance', 0.0)) / n_atoms if gate_sample.get('energy_variance', 0.0) > 0 else 0.005
            
            if uncertainty_per_atom < 0.005:
                delta_per_atom = delta_per_atom * 0.1
            elif uncertainty_per_atom < 0.01:
                delta_per_atom = delta_per_atom * 0.3
            else:
                delta_per_atom = delta_per_atom * 0.5
            
            max_correction = 0.02
            delta_per_atom = np.clip(delta_per_atom, -max_correction, max_correction)
            
            if abs(delta_per_atom) < 0.0001:
                delta_per_atom = 0.0
            
            # Delta-corrected prediction
            energy_delta_total = energy_baseline_total + (delta_per_atom * n_atoms)
            energy_delta_per_atom = energy_delta_total / n_atoms
            delta_error_per_atom = abs(energy_delta_per_atom - energy_target_per_atom)
            
            baseline_results.append({
                'sample_id': sample_id,
                'domain': domain,
                'n_atoms': n_atoms,
                'energy_target_per_atom': energy_target_per_atom,
                'energy_baseline_per_atom': energy_baseline_per_atom,
                'baseline_error_per_atom': baseline_error_per_atom,
                'uncertainty_per_atom': uncertainty_per_atom
            })
            
            delta_results.append({
                'sample_id': sample_id,
                'domain': domain,
                'n_atoms': n_atoms,
                'energy_target_per_atom': energy_target_per_atom,
                'energy_delta_per_atom': energy_delta_per_atom,
                'delta_error_per_atom': delta_error_per_atom,
                'delta_correction_per_atom': delta_per_atom,
                'uncertainty_per_atom': uncertainty_per_atom
            })
            
            matched_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {sample_id}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"\n✅ Evaluated {matched_count} samples ({failed_count} failed)")
    
    if matched_count == 0:
        logger.error("No samples matched! Cannot evaluate.")
        return
    
    # Compute metrics
    baseline_errors = [r['baseline_error_per_atom'] for r in baseline_results]
    delta_errors = [r['delta_error_per_atom'] for r in delta_results]
    
    baseline_mae = np.mean(baseline_errors)
    delta_mae = np.mean(delta_errors)
    
    baseline_rmse = np.sqrt(np.mean([e**2 for e in baseline_errors]))
    delta_rmse = np.sqrt(np.mean([e**2 for e in delta_errors]))
    
    # Improvement calculation
    improvement_pct = ((baseline_mae - delta_mae) / baseline_mae) * 100 if baseline_mae > 0 else 0
    
    # Print results
    print("\n" + "="*80)
    print("  DELTA HEAD EVALUATION ON HARD CASES ONLY")
    print("="*80)
    print(f"\n📊 Results on {matched_count} hard cases:")
    print()
    print(f"{'Metric':<40} {'Baseline':>15} {'Delta-Corrected':>15} {'Improvement':>15}")
    print("-" * 85)
    print(f"{'MAE (eV/atom)':<40} {baseline_mae:>15.6f} {delta_mae:>15.6f} {improvement_pct:>14.1f}%")
    print(f"{'RMSE (eV/atom)':<40} {baseline_rmse:>15.6f} {np.sqrt(np.mean([e**2 for e in delta_errors])):>15.6f} {((baseline_rmse - np.sqrt(np.mean([e**2 for e in delta_errors]))) / baseline_rmse * 100):>14.1f}%")
    print()
    
    # Per-domain breakdown
    domain_baseline = {}
    domain_delta = {}
    
    for r in baseline_results:
        domain = r['domain']
        if domain not in domain_baseline:
            domain_baseline[domain] = []
        domain_baseline[domain].append(r['baseline_error_per_atom'])
    
    for r in delta_results:
        domain = r['domain']
        if domain not in domain_delta:
            domain_delta[domain] = []
        domain_delta[domain].append(r['delta_error_per_atom'])
    
    print("📊 Per-Domain Results:")
    print()
    print(f"{'Domain':<20} {'Baseline MAE':>15} {'Delta MAE':>15} {'Improvement':>15}")
    print("-" * 65)
    
    for domain in sorted(set(list(domain_baseline.keys()) + list(domain_delta.keys()))):
        if domain in domain_baseline and domain in domain_delta:
            bl_mae = np.mean(domain_baseline[domain])
            dt_mae = np.mean(domain_delta[domain])
            imp = ((bl_mae - dt_mae) / bl_mae * 100) if bl_mae > 0 else 0
            print(f"{domain:<20} {bl_mae:>15.6f} {dt_mae:>15.6f} {imp:>14.1f}%")
    
    print()
    print("="*80)
    
    if improvement_pct > 0:
        print("✅ Delta head IMPROVES predictions on hard cases!")
    elif improvement_pct > -5:
        print("⚠️  Delta head maintains similar performance on hard cases")
    else:
        print("❌ Delta head degrades predictions on hard cases")
    
    print("="*80)


if __name__ == "__main__":
    evaluate_on_hard_cases()



