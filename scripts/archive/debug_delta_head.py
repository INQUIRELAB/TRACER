#!/usr/bin/env python3
"""
Debug script to check delta head predictions on a few samples.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig

def sample_to_pyg(sample):
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

def extract_features(gnn_model, sample, device):
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        x=atomic_numbers.unsqueeze(1).float()
    ).to(device)
    
    if hasattr(gnn_model, 'model') and hasattr(gnn_model.model, 'embedding'):
        embedding_layer = gnn_model.model.embedding
    elif hasattr(gnn_model, 'embedding'):
        embedding_layer = gnn_model.embedding
    else:
        raise AttributeError("No embedding layer")
    
    embeddings = embedding_layer(data.atomic_numbers)
    pooled_features = embeddings.sum(dim=0, keepdim=True)
    
    target_dim = 256
    if pooled_features.size(1) > target_dim:
        pooled_features = pooled_features[:, :target_dim]
    elif pooled_features.size(1) < target_dim:
        padding = torch.zeros(1, target_dim - pooled_features.size(1), device=device)
        pooled_features = torch.cat([pooled_features, padding], dim=1)
    
    return pooled_features

device = torch.device('cuda')

# Load models
print("Loading models...")
gemnet_checkpoint = torch.load("models/gemnet_per_atom/best_model.pt", map_location='cpu')
gemnet_config = gemnet_checkpoint.get('model_config', {})

# Compute normalization from training data
train_file = Path("data/preprocessed_full_unified/train_data.json")
with open(train_file, 'r') as f:
    train_samples = json.load(f)
all_energies = []
for s in train_samples:
    energy = s.get('energy', s.get('energy_target', 0))
    n_atoms = len(s.get('atomic_numbers', [1]))
    if n_atoms > 0:
        all_energies.append(energy / n_atoms)
norm_mean = np.mean(all_energies)
norm_std = np.std(all_energies)

gemnet_model = GemNetWrapper(
    num_atoms=gemnet_config.get('num_atoms', 120),
    hidden_dim=gemnet_config.get('hidden_dim', 256),
    num_filters=gemnet_config.get('num_filters', 256),
    num_interactions=gemnet_config.get('num_interactions', 6),
    cutoff=gemnet_config.get('cutoff', 10.0),
    readout="sum",
    mean=norm_mean,
    std=norm_std,
).to(device)
gemnet_model.load_state_dict(gemnet_checkpoint['model_state_dict'])
gemnet_model.eval()

delta_checkpoint = torch.load("artifacts/delta_head_real_270_per_atom/best_model.pt", map_location=device)
config_dict = delta_checkpoint.get('config', {})
if isinstance(config_dict, DeltaHeadConfig):
    delta_config = config_dict
else:
    delta_config = DeltaHeadConfig(
        schnet_feature_dim=config_dict.get('schnet_feature_dim', 256),
        domain_embedding_dim=config_dict.get('domain_embedding_dim', 16),
        hidden_dim=config_dict.get('hidden_dim', 128),
        num_layers=config_dict.get('num_layers', 3),
        dropout=config_dict.get('dropout', 0.1),
        num_domains=config_dict.get('num_domains', 5)
    )
delta_head = DeltaHead(delta_config).to(device)
delta_head.load_state_dict(delta_checkpoint['model_state_dict'])
delta_head.eval()

# Load test data
with open("data/preprocessed_full_unified/test_data.json", 'r') as f:
    test_data = json.load(f)

# Check first 5 samples
print("\n" + "="*80)
print("DELTA HEAD PREDICTIONS ON FIRST 5 SAMPLES:")
print("="*80)

domain_map = {'jarvis_dft': 0, 'jarvis_elastic': 1, 'oc20_s2ef': 2, 'oc22_s2ef': 3, 'ani1x': 4}

for i in range(min(5, len(test_data))):
    sample = test_data[i]
    n_atoms = len(sample['atomic_numbers'])
    target_total = sample.get('energy', sample.get('energy_target', 0.0))
    target_per_atom = target_total / n_atoms
    
    # GemNet prediction
    pyg_data = sample_to_pyg(sample)
    pyg_data = pyg_data.to(device)
    
    with torch.no_grad():
        energy_total_pred, _, _ = gemnet_model(pyg_data, compute_forces=False)
        if hasattr(pyg_data, 'batch'):
            n_atoms_tensor = torch.tensor([n_atoms], dtype=energy_total_pred.dtype, device=device)
        else:
            n_atoms_tensor = torch.tensor([n_atoms], dtype=energy_total_pred.dtype, device=device)
        energy_per_atom_pred = energy_total_pred / n_atoms_tensor
        if gemnet_model.mean is not None and gemnet_model.std is not None:
            energy_per_atom_pred = energy_per_atom_pred * gemnet_model.std + gemnet_model.mean
        energy_per_atom_pred = energy_per_atom_pred.item()
        energy_total_pred = energy_per_atom_pred * n_atoms
    
    # Delta head prediction
    features = extract_features(gemnet_model, sample, device)
    domain_id = torch.tensor([domain_map.get(sample.get('domain', 'jarvis_dft'), 0)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        delta_output = delta_head(features, domain_id)
        if isinstance(delta_output, dict):
            delta_per_atom = delta_output.get('delta_energy', 0.0)
            if isinstance(delta_per_atom, torch.Tensor):
                delta_per_atom = delta_per_atom.squeeze().item()
        elif isinstance(delta_output, torch.Tensor):
            delta_per_atom = delta_output.squeeze().item()
        else:
            delta_per_atom = float(delta_output)
    
    delta_total = delta_per_atom * n_atoms
    corrected_per_atom = energy_per_atom_pred + delta_per_atom
    corrected_total = energy_total_pred + delta_total
    
    print(f"\nSample {i+1}: {sample.get('sample_id', 'unknown')}")
    print(f"  Domain: {sample.get('domain', 'unknown')}")
    print(f"  N atoms: {n_atoms}")
    print(f"  Target (per-atom): {target_per_atom:.6f} eV/atom")
    print(f"  Baseline (per-atom): {energy_per_atom_pred:.6f} eV/atom")
    print(f"  Delta correction (per-atom): {delta_per_atom:.6f} eV/atom")
    print(f"  Corrected (per-atom): {corrected_per_atom:.6f} eV/atom")
    print(f"  Baseline error: {abs(energy_per_atom_pred - target_per_atom):.6f} eV/atom")
    print(f"  Corrected error: {abs(corrected_per_atom - target_per_atom):.6f} eV/atom")
    print(f"  Delta magnitude: {abs(delta_per_atom):.6f} eV/atom")
    
print("\n" + "="*80)


