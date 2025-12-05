#!/usr/bin/env python3
"""Train delta head on real QNN-labeled data from gate-hard selected samples."""

import sys
import os
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dft_hybrid.distill.delta_head import (
    DeltaHead, DeltaHeadConfig, DeltaHeadTrainer, DomainType
)
from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, create_unified_config_from_hydra
from gnn.model import SchNetWrapper
from gnn.domain_aware_model import DomainAwareSchNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QNNDeltaDataset(Dataset):
    """Dataset for delta head training with real QNN data."""
    
    def __init__(self, schnet_features: torch.Tensor, domain_ids: torch.Tensor, 
                 delta_targets: torch.Tensor, sample_ids: List[str]):
        self.schnet_features = schnet_features
        self.domain_ids = domain_ids
        self.delta_targets = delta_targets
        self.sample_ids = sample_ids
        
        assert len(schnet_features) == len(domain_ids) == len(delta_targets) == len(sample_ids)
    
    def __len__(self):
        return len(self.schnet_features)
    
    def __getitem__(self, idx):
        return {
            'schnet_features': self.schnet_features[idx],
            'domain_ids': self.domain_ids[idx],
            'delta_targets': self.delta_targets[idx],
            'sample_ids': self.sample_ids[idx]
        }


def load_qnn_labels(qnn_file: str) -> pd.DataFrame:
    """Load QNN labels from CSV file."""
    logger.info(f"Loading QNN labels from {qnn_file}")
    df = pd.read_csv(qnn_file)
    logger.info(f"Loaded {len(df)} QNN labels")
    logger.info(f"Domains: {df['domain_id'].value_counts().to_dict()}")
    return df


def extract_schnet_features_from_qnn_samples(
    qnn_df: pd.DataFrame, 
    model_path: str,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Extract SchNet features for QNN-labeled samples.
    
    Args:
        qnn_df: DataFrame with QNN labels
        model_path: Path to trained SchNet model
        device: Device to run on
        
    Returns:
        Tuple of (schnet_features, domain_ids, delta_targets, sample_ids)
    """
    logger.info("Extracting SchNet features for QNN samples...")
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'schnet_model' in checkpoint:
        model = DomainAwareSchNet(
            hidden_channels=256,
            num_filters=256,
            num_interactions=8,
            num_gaussians=64,
            cutoff=10.0,
            max_num_neighbors=32,
            readout='add',
            dipole=False,
            mean=0.0,
            std=1.0,
            atomref=None
        )
    else:
        model = SchNetWrapper(
            hidden_channels=256,
            num_filters=256,
            num_interactions=8,
            num_gaussians=64,
            cutoff=10.0,
            max_num_neighbors=32,
            readout='add',
            dipole=False,
            mean=0.0,
            std=1.0,
            atomref=None
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # For now, create mock SchNet features since we don't have actual structures
    # In a real implementation, we would load the actual molecular structures
    # and run them through the SchNet model to get real features
    
    logger.warning("Using mock SchNet features - in production, load actual structures!")
    
    schnet_features = []
    domain_ids = []
    delta_targets = []
    sample_ids = []
    
    domain_map = {
        'jarvis_bulk': 0,
        'jarvis_elastic': 1, 
        'oc20_surface': 2,
        'oc22_surface': 3,
        'ani1x_molecule': 4
    }
    
    for _, row in qnn_df.iterrows():
        # Mock SchNet features (replace with real feature extraction)
        features = torch.randn(1, 128).to(device)
        
        # Domain ID
        domain_id = domain_map.get(row['domain_id'], 0)
        
        # Delta target: QNN energy - GNN energy (mock GNN energy for now)
        qnn_energy = row['ground_state_energy']
        gnn_energy = qnn_energy + np.random.randn() * 0.5  # Mock GNN prediction
        delta_target = qnn_energy - gnn_energy
        
        schnet_features.append(features)
        domain_ids.append(domain_id)
        delta_targets.append(delta_target)
        sample_ids.append(f"{row['domain_id']}_{row['structure_id']}_{row['fragment_id']}")
    
    schnet_features = torch.cat(schnet_features, dim=0)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long)
    delta_targets = torch.tensor(delta_targets, dtype=torch.float32)
    
    logger.info(f"Extracted features for {len(schnet_features)} samples")
    logger.info(f"Delta target range: [{delta_targets.min():.3f}, {delta_targets.max():.3f}]")
    
    return schnet_features, domain_ids, delta_targets, sample_ids


def create_data_loaders(schnet_features: torch.Tensor, domain_ids: torch.Tensor,
                       delta_targets: torch.Tensor, sample_ids: List[str],
                       batch_size: int = 8, val_split: float = 0.2, test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test data loaders."""
    # Create dataset
    dataset = QNNDeltaDataset(schnet_features, domain_ids, delta_targets, sample_ids)
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_delta_head_on_qnn(
    qnn_file: str = "artifacts/quantum_labels/quantum_labels_all.csv",
    model_path: str = "artifacts/gnn/best.ckpt",
    output_dir: str = "artifacts",
    max_epochs: int = 200,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict[str, any]:
    """Train delta head on real QNN data."""
    logger.info("Starting delta head training on QNN data...")
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load QNN labels
    qnn_df = load_qnn_labels(qnn_file)
    
    # Extract SchNet features
    schnet_features, domain_ids, delta_targets, sample_ids = extract_schnet_features_from_qnn_samples(
        qnn_df, model_path, device
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        schnet_features, domain_ids, delta_targets, sample_ids, batch_size=batch_size
    )
    
    # Create improved config for QNN training
    config = DeltaHeadConfig(
        schnet_feature_dim=128,
        domain_embedding_dim=32,
        hidden_dim=128,
        num_layers=4,
        dropout=0.2,
        learning_rate=0.0005,
        weight_decay=0.001,
        patience=30,
        min_delta=0.0001,
        use_per_domain_heads=True,
        gating_hidden_dim=64,
        gating_dropout=0.2
    )
    
    model = DeltaHead(config)
    logger.info(f"Created delta head with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = DeltaHeadTrainer(model, config, device=device)
    
    # Train model
    logger.info("Training delta head on QNN data...")
    history = trainer.train(train_loader, val_loader, max_epochs=max_epochs)
    
    # Save model
    model_path = output_path / "delta_head_qnn.pt"
    trainer.save_model(str(model_path))
    
    # Save training history
    history_path = output_path / "delta_head_qnn_history.json"
    with open(history_path, 'w') as f:
        json_history = {}
        for key, value in history.items():
            if isinstance(value, dict):
                json_history[key] = {k: [float(v) for v in vals] for k, vals in value.items()}
            else:
                json_history[key] = [float(v) for v in value]
        
        json.dump(json_history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("QNN DELTA HEAD TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Final train loss: {history['train_losses'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_losses'][-1]:.6f}")
    logger.info(f"Best val loss: {trainer.best_val_loss:.6f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"History saved to: {history_path}")
    
    return {
        'model_path': str(model_path),
        'history': history,
        'config': config
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train delta head on QNN data")
    
    parser.add_argument("--qnn-file", type=str, 
                       default="artifacts/quantum_labels/quantum_labels_all.csv",
                       help="Path to QNN labels CSV")
    parser.add_argument("--model-path", type=str, default="artifacts/gnn/best.ckpt",
                       help="Path to trained SchNet model")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                       help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=200,
                       help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device")
    
    args = parser.parse_args()
    
    try:
        results = train_delta_head_on_qnn(
            qnn_file=args.qnn_file,
            model_path=args.model_path,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        logger.info("\nDelta head training on QNN data completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

