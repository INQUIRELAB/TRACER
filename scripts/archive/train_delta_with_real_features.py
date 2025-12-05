#!/usr/bin/env python3
"""
Train delta head using REAL GemNet features extracted from gate-hard samples.
This fixes the critical issue where the previous training used random features.
"""

import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig, DeltaHeadTrainer
from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data

console = Console()


def load_gemnet_model_for_features(norm_stats: dict):
    """Load GemNet model for feature extraction."""
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
    
    return model, device


def extract_real_features(gnn_model, sample, device):
    """Extract REAL SchNet/GemNet features from the model."""
    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
    positions = torch.tensor(sample['positions'], dtype=torch.float32)
    
    data = Data(
        atomic_numbers=atomic_numbers,
        pos=positions,
        batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
        x=atomic_numbers.unsqueeze(1).float()
    ).to(device)
    
    # Extract embeddings from GemNet
    if hasattr(gnn_model, 'model') and hasattr(gnn_model.model, 'embedding'):
        embedding_layer = gnn_model.model.embedding
    elif hasattr(gnn_model, 'embedding'):
        embedding_layer = gnn_model.embedding
    else:
        raise AttributeError("GNN model does not have accessible embedding layer")
    
    with torch.no_grad():
        embeddings = embedding_layer(data.atomic_numbers)
        
        # Pool features (sum over atoms)
        pooled_features = embeddings.sum(dim=0, keepdim=True)
        
        # Resize to match delta head input dim (256)
        target_dim = 256
        if pooled_features.size(1) > target_dim:
            pooled_features = pooled_features[:, :target_dim]
        elif pooled_features.size(1) < target_dim:
            padding = torch.zeros(1, target_dim - pooled_features.size(1), device=device)
            pooled_features = torch.cat([pooled_features, padding], dim=1)
    
    return pooled_features


def load_preprocessed_dataset():
    """Load all preprocessed splits (train, val, test) to find matching samples."""
    data_path = Path("data/preprocessed_full_unified")
    
    all_samples = []
    
    for split in ['train', 'val', 'test']:
        split_file = data_path / f'{split}_data.json'
        if split_file.exists():
            with open(split_file, 'r') as f:
                samples = json.load(f)
                # Add split info and construct sample_id if missing
                for i, sample in enumerate(samples):
                    if 'sample_id' not in sample:
                        domain = sample.get('domain', 'unknown')
                        # Construct sample_id: domain_sample_XXXXXX
                        sample_id = f"{domain}_sample_{i:06d}"
                        sample['sample_id'] = sample_id
                    sample['_split'] = split
                    sample['_index'] = i
                    all_samples.append(sample)
    
    console.print(f"📁 Loaded {len(all_samples)} samples from preprocessed dataset")
    return all_samples


def create_sample_mapping(preprocessed_samples: List[Dict]):
    """Create mapping from (domain, energy_target) to sample for fast lookup."""
    mapping = {}
    energy_tolerance = 0.001  # eV
    
    for sample in preprocessed_samples:
        domain = sample.get('domain', 'unknown')
        energy_target = sample.get('energy', sample.get('energy_target', 0.0))
        
        # Create key with rounded energy for tolerance matching
        energy_key = round(energy_target / energy_tolerance) * energy_tolerance
        key = (domain, energy_key)
        
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(sample)
    
    return mapping


def create_domain_to_samples_mapping(preprocessed_samples: List[Dict]):
    """Create mapping from domain to list of samples (indexed by domain-specific index)."""
    domain_to_samples = {}
    
    for sample in preprocessed_samples:
        domain = sample.get('domain', 'unknown')
        if domain not in domain_to_samples:
            domain_to_samples[domain] = []
        domain_to_samples[domain].append(sample)
    
    return domain_to_samples


def parse_sample_id(sample_id: str):
    """Parse sample ID like 'jarvis_dft_sample_000259' to extract domain and index."""
    if '_sample_' not in sample_id:
        return None, None
    
    parts = sample_id.split('_sample_')
    if len(parts) != 2:
        return None, None
    
    # Domain is everything before '_sample_', but may contain underscores
    # Try common patterns: jarvis_dft, jarvis_elastic, oc20_s2ef, oc22_s2ef, ani1x
    domain_prefix = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        return None, None
    
    # Map domain prefixes
    domain_map = {
        'jarvis_dft': 'jarvis_dft',
        'jarvis_elastic': 'jarvis_elastic',
        'oc20_s2ef': 'oc20_s2ef',
        'oc22_s2ef': 'oc22_s2ef',
        'ani1x': 'ani1x'
    }
    
    # Try exact match first
    domain = domain_map.get(domain_prefix, domain_prefix)
    
    return domain, idx


def find_matching_sample(gate_sample: Dict, sample_mapping: Dict, preprocessed_samples: List[Dict], domain_to_samples: Dict) -> Dict:
    """Find matching preprocessed sample for a gate-hard sample using multiple strategies."""
    domain = gate_sample.get('domain', 'unknown')
    energy_target = gate_sample.get('energy_target', 0.0)
    sample_id = gate_sample.get('sample_id', '')
    
    # Strategy 1: Parse sample ID and match by domain + index
    if sample_id:
        parsed_domain, parsed_idx = parse_sample_id(sample_id)
        if parsed_domain and parsed_idx is not None:
            # Get samples for this domain
            domain_samples = domain_to_samples.get(parsed_domain, [])
            if parsed_idx < len(domain_samples):
                candidate = domain_samples[parsed_idx]
                # Verify it's the same domain
                if candidate.get('domain') == parsed_domain:
                    return candidate
    
    # Strategy 2: Direct sample_id match (if preprocessed data has sample_id)
    if sample_id:
        for sample in preprocessed_samples:
            if sample.get('sample_id') == sample_id:
                return sample
    
    # Strategy 3: Match by domain + energy_target with tight tolerance (0.001 eV)
    energy_tolerance_tight = 0.001
    energy_key = round(energy_target / energy_tolerance_tight) * energy_tolerance_tight
    key = (domain, energy_key)
    
    if key in sample_mapping:
        candidates = sample_mapping[key]
        # Find closest energy match
        best_match = None
        best_diff = float('inf')
        for candidate in candidates:
            cand_energy = candidate.get('energy', candidate.get('energy_target', 0.0))
            diff = abs(cand_energy - energy_target)
            if diff < best_diff and diff < 0.01:  # Within 0.01 eV
                best_diff = diff
                best_match = candidate
        
        if best_match:
            return best_match
    
    # Strategy 4: Match by domain + energy with moderate tolerance (0.01 eV)
    domain_samples = domain_to_samples.get(domain, [])
    best_match = None
    best_diff = float('inf')
    
    for sample in domain_samples:
        cand_energy = sample.get('energy', sample.get('energy_target', 0.0))
        diff = abs(cand_energy - energy_target)
        if diff < best_diff and diff < 0.05:  # Within 0.05 eV
            best_diff = diff
            best_match = sample
    
    if best_match:
        return best_match
    
    # Strategy 5: For jarvis_dft, try matching across all splits with relaxed tolerance
    if domain == 'jarvis_dft' and len(domain_samples) > 0:
        # Sort by energy difference and try closest matches
        candidates_with_diff = []
        for sample in domain_samples:
            cand_energy = sample.get('energy', sample.get('energy_target', 0.0))
            diff = abs(cand_energy - energy_target)
            candidates_with_diff.append((sample, diff))
        
        # Sort by energy difference
        candidates_with_diff.sort(key=lambda x: x[1])
        
        # Try increasingly relaxed tolerances
        for tolerance in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            for sample, diff in candidates_with_diff:
                if diff < tolerance:
                    return sample
    
    # Strategy 6: Fallback - closest energy match within domain (up to 2.0 eV for final fallback)
    for sample in domain_samples:
        cand_energy = sample.get('energy', sample.get('energy_target', 0.0))
        diff = abs(cand_energy - energy_target)
        if diff < best_diff:
            best_diff = diff
            best_match = sample
    
    if best_match and best_diff < 2.0:  # Within 2.0 eV (relaxed for edge cases)
        return best_match
    
    return None


def load_gate_hard_samples(gate_hard_dir: str) -> List[Dict[str, Any]]:
    """Load gate-hard selected samples."""
    gate_hard_path = Path(gate_hard_dir)
    samples = []
    
    topk_all_file = gate_hard_path / "topK_all.jsonl"
    if topk_all_file.exists():
        console.print(f"📁 Loading gate-hard samples from: {topk_all_file}")
        with open(topk_all_file, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
    
    console.print(f"✅ Loaded {len(samples)} gate-hard samples")
    return samples


def load_qnn_labels(qnn_labels_path: str) -> pd.DataFrame:
    """Load QNN labels."""
    console.print(f"📁 Loading QNN labels from: {qnn_labels_path}")
    df = pd.read_csv(qnn_labels_path)
    console.print(f"✅ Loaded {len(df)} QNN labels")
    return df


class GateHardDeltaDataset(Dataset):
    """Dataset for delta head training using REAL GemNet features."""
    
    def __init__(self, gate_hard_samples: List[Dict], qnn_labels: pd.DataFrame, 
                 preprocessed_samples: List[Dict], sample_mapping: Dict,
                 domain_to_samples: Dict, gnn_model, device):
        self.samples = []
        self.gnn_model = gnn_model
        self.device = device
        
        # Create mapping from sample_id to QNN labels
        qnn_dict = {}
        for _, row in qnn_labels.iterrows():
            qnn_dict[row['sample_id']] = row
        
        console.print("🔄 Matching gate-hard samples with preprocessed data and extracting REAL features...")
        
        matched_count = 0
        failed_count = 0
        domain_stats = {}  # Track matching by domain
        
        # Match gate-hard samples with preprocessed data and extract REAL features
        for i, gate_sample in enumerate(gate_hard_samples):
            sample_id = gate_sample['sample_id']
            domain = gate_sample.get('domain', 'unknown')
            
            # Initialize domain stats
            if domain not in domain_stats:
                domain_stats[domain] = {'matched': 0, 'failed': 0}
            
            if sample_id not in qnn_dict:
                domain_stats[domain]['failed'] += 1
                failed_count += 1
                continue
            
            qnn_data = qnn_dict[sample_id]
            
            # Find matching preprocessed sample
            matched_sample = find_matching_sample(gate_sample, sample_mapping, preprocessed_samples, domain_to_samples)
            
            if matched_sample is None:
                domain_stats[domain]['failed'] += 1
                failed_count += 1
                if failed_count <= 10:  # Print first 10 failures
                    console.print(f"   ⚠️  Could not find matching sample for {sample_id} (domain: {domain})")
                continue
            
            # Check that matched sample has required fields
            if 'atomic_numbers' not in matched_sample or 'positions' not in matched_sample:
                failed_count += 1
                if failed_count <= 5:
                    console.print(f"   ⚠️  Matched sample {sample_id} missing atomic structure data")
                continue
            
            # Extract REAL SchNet features from GemNet model using matched sample
            try:
                schnet_features = extract_real_features(gnn_model, matched_sample, device)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    console.print(f"   ⚠️  Failed to extract features for {sample_id}: {e}")
                continue
            
            # Get domain ID
            domain_map = {
                'jarvis_dft': 0,
                'jarvis_elastic': 1,
                'oc20_s2ef': 2,
                'oc22_s2ef': 3,
                'ani1x': 4
            }
            domain_id = domain_map.get(gate_sample['domain'], 0)
            
            # Calculate per-atom delta target
            n_atoms = max(1, len(matched_sample.get('atomic_numbers', [])))
            if 'delta_energy_per_atom' in qnn_data:
                delta_target = float(qnn_data['delta_energy_per_atom'])
            else:
                gnn_energy = float(qnn_data.get('gnn_energy', gate_sample['energy_pred']))
                qnn_energy = float(qnn_data.get('qnn_energy', gnn_energy))
                delta_target = (qnn_energy - gnn_energy) / n_atoms
            
            self.samples.append({
                'schnet_features': schnet_features.cpu(),  # Move to CPU for dataset
                'domain_ids': torch.tensor([domain_id], dtype=torch.long),
                'delta_targets': torch.tensor([delta_target], dtype=torch.float32),
                'sample_id': sample_id
            })
            matched_count += 1
            domain_stats[domain]['matched'] += 1
            
            if (matched_count + failed_count) % 50 == 0:
                console.print(f"   Processed {matched_count + failed_count}/{len(gate_hard_samples)} samples... ({matched_count} matched, {failed_count} failed)")
        
        console.print(f"✅ Created dataset with {len(self.samples)} samples (with real features)")
        console.print(f"   Matched: {matched_count}, Failed: {failed_count}")
        console.print(f"\n📊 Matching statistics by domain:")
        for domain, stats in sorted(domain_stats.items()):
            total = stats['matched'] + stats['failed']
            console.print(f"   {domain}: {stats['matched']}/{total} matched ({stats['matched']/total*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_data_loaders(dataset: GateHardDeltaDataset, batch_size: int = 32, train_split: float = 0.8):
    """Create train and validation data loaders."""
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    console.print(f"📊 Data split: {train_size} train, {val_size} validation")
    return train_loader, val_loader


def train_delta_head_with_real_features(
    gate_hard_dir: str = "artifacts/gate_hard_full",
    qnn_labels_path: str = "artifacts/quantum_labels_gate_hard.csv",
    output_dir: str = "artifacts/delta_head_real_features",
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train delta head using REAL GemNet features."""
    
    console.print("🚀 Training Delta Head with REAL GemNet Features")
    console.print(f"📁 Gate-hard directory: {gate_hard_dir}")
    console.print(f"📁 QNN labels: {qnn_labels_path}")
    console.print(f"💾 Output directory: {output_dir}")
    console.print(f"🔧 Device: {device}")
    
    device = torch.device(device)
    
    # Compute normalization stats for GemNet
    train_file = Path("data/preprocessed_full_unified/train_data.json")
    norm_stats = {}
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
            norm_stats['mean'] = np.mean(all_energies)
            norm_stats['std'] = np.std(all_energies)
    
    # Load GemNet model for feature extraction
    gnn_model, gnn_device = load_gemnet_model_for_features(norm_stats)
    
    # Load data
    gate_hard_samples = load_gate_hard_samples(gate_hard_dir)
    qnn_labels = load_qnn_labels(qnn_labels_path)
    
    if len(gate_hard_samples) == 0:
        console.print("❌ No gate-hard samples found!")
        return
    
    if len(qnn_labels) == 0:
        console.print("❌ No QNN labels found!")
        return
    
    # Load preprocessed dataset for matching
    console.print("\n🔄 Loading preprocessed dataset for sample matching...")
    preprocessed_samples = load_preprocessed_dataset()
    sample_mapping = create_sample_mapping(preprocessed_samples)
    domain_to_samples = create_domain_to_samples_mapping(preprocessed_samples)
    
    # Print domain statistics
    console.print(f"📊 Domain distribution in preprocessed data:")
    for domain in sorted(domain_to_samples.keys()):
        console.print(f"   {domain}: {len(domain_to_samples[domain])} samples")
    
    # Create dataset with REAL features
    dataset = GateHardDeltaDataset(
        gate_hard_samples, 
        qnn_labels, 
        preprocessed_samples,
        sample_mapping,
        domain_to_samples,
        gnn_model, 
        gnn_device
    )
    
    if len(dataset) == 0:
        console.print("❌ No matching samples found between gate-hard and QNN labels!")
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, batch_size)
    
    # Create model
    config = DeltaHeadConfig(
        schnet_feature_dim=256,
        domain_embedding_dim=16,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
        learning_rate=learning_rate,
        weight_decay=1e-4
    )
    
    model = DeltaHead(config).to(device)
    trainer = DeltaHeadTrainer(model, config, device)
    
    console.print(f"🏗️ Created delta head model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    console.print(f"🎯 Starting training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        for epoch in range(num_epochs):
            # Train
            train_task = progress.add_task(f"Epoch {epoch+1}/{num_epochs} - Training", total=len(train_loader))
            train_loss = trainer.train_epoch(train_loader)
            progress.update(train_task, completed=len(train_loader))
            
            # Validate
            val_task = progress.add_task(f"Epoch {epoch+1}/{num_epochs} - Validation", total=len(val_loader))
            val_loss, domain_maes = trainer.validate_epoch(val_loader)
            progress.update(val_task, completed=len(val_loader))
            
            # Update history
            train_history.append(train_loss)
            val_history.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, output_path / "best_model.pt")
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                console.print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if epoch > 20 and val_loss > best_val_loss * 1.1:
                console.print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Save final model and history
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_history': train_history,
        'val_history': val_history
    }, output_path / "final_model.pt")
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_history) + 1),
        'train_loss': train_history,
        'val_loss': val_history
    })
    history_df.to_csv(output_path / "training_history.csv", index=False)
    
    console.print(f"\n✅ Training completed!")
    console.print(f"📊 Final Results:")
    console.print(f"   Best validation loss: {best_val_loss:.6f}")
    console.print(f"   Final train loss: {train_loss:.6f}")
    console.print(f"   Final val loss: {val_loss:.6f}")
    console.print(f"   Total epochs: {epoch + 1}")
    console.print(f"   Model saved to: {output_path}")
    console.print(f"   ⚠️  This model uses REAL GemNet features (not random!)")
    
    return output_path


def main(
    gate_hard_dir: str = typer.Option("artifacts/gate_hard_full", help="Directory containing gate-hard selected samples"),
    qnn_labels_path: str = typer.Option("artifacts/quantum_labels_gate_hard.csv", help="Path to QNN labels CSV file"),
    output_dir: str = typer.Option("artifacts/delta_head_real_features", help="Output directory for trained model"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_epochs: int = typer.Option(100, help="Number of training epochs"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
):
    """Train delta head using REAL GemNet features."""
    train_delta_head_with_real_features(
        gate_hard_dir=gate_hard_dir,
        qnn_labels_path=qnn_labels_path,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )


if __name__ == "__main__":
    typer.run(main)

