#!/usr/bin/env python3
"""
QM9 Evaluation Script
Tests the pipeline on QM9 dataset (not in training data)
"""

import sys
import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from pathlib import Path
import logging
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.run import HybridPipeline
from gnn.model import SchNetWrapper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def evaluate_qm9(
    max_samples: int = 1000,
    device: str = "cuda"
):
    """Evaluate model on QM9 dataset"""
    
    logger.info("🚀 QM9 EVALUATION")
    logger.info("="*80)
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Device: {device}")
    
    # 1. Load QM9 dataset
    logger.info("\n📥 Loading QM9 dataset...")
    try:
        dataset = QM9(root="data/QM9")
        logger.info(f"✅ Loaded QM9: {len(dataset)} molecules")
        logger.info(f"   First molecule: {dataset[0]}")
    except Exception as e:
        logger.error(f"❌ Failed to load QM9: {e}")
        logger.info("📥 Downloading QM9... (this may take a while)")
        dataset = QM9(root="data/QM9", download=True)
        logger.info(f"✅ Loaded QM9: {len(dataset)} molecules")
    
    # 2. Load pipeline and models
    logger.info("\n📥 Loading models...")
    config = DictConfig({'pipeline': {'device': device}})
    pipeline = HybridPipeline(config)
    
    gnn_model_path = "models/gnn_training_enhanced/ensemble/ckpt_0.pt"
    gnn_model = pipeline.load_model(gnn_model_path)
    logger.info("✅ Loaded GNN model")
    
    # Load delta head if available
    try:
        delta_head_path = "artifacts/delta_head.pt"
        delta_head = pipeline.load_delta_head(delta_head_path)
        logger.info("✅ Loaded delta head")
        use_delta = True
    except:
        logger.info("⚠️  Delta head not found, using GNN only")
        use_delta = False
    
    # 3. Prepare samples
    logger.info("\n🔬 Preparing samples...")
    samples = []
    for i in range(min(max_samples, len(dataset))):
        data = dataset[i]
        
        # QM9 uses 'y' for target - it's shape [1, 19]
        # QM9 provides 19 properties, index 10 is U0 (formation energy at 0K)
        # Index 15 is U0_atom (formation energy per atom)
        # We'll use U0 (index 10) which is the internal energy at 0K in eV
        energy_target = data.y[0][10].item()  # U0 formation energy (eV)
        energy_per_atom_target = data.y[0][15].item()  # U0 per atom
        
        sample = {
            'sample_id': f"qm9_{i}",
            'domain': 'ani1x',  # Similar to ANI1x
            'num_atoms': data.pos.shape[0],
            'atomic_numbers': data.z.tolist() if hasattr(data, 'z') else [6]*data.pos.shape[0],
            'positions': data.pos.tolist(),
            'energy_target': energy_target,
            'energy_per_atom_target': energy_per_atom_target,
            'forces_target': [[0.0, 0.0, 0.0]] * data.pos.shape[0]
        }
        samples.append(sample)
    
    logger.info(f"✅ Prepared {len(samples)} samples")
    
    # 4. Run predictions
    logger.info("\n🔬 Running predictions...")
    gnn_predictions = pipeline.predict_with_model(gnn_model, samples)
    
    # 5. Apply delta head if available
    if use_delta:
        logger.info("\n🔬 Applying quantum corrections...")
        corrected_predictions = pipeline.apply_delta_head(delta_head, gnn_predictions, gnn_model=gnn_model)
        predictions = corrected_predictions
    else:
        predictions = gnn_predictions
    
    # 6. Calculate metrics (focus on per-atom for proper comparison)
    logger.info("\n📊 Calculating metrics...")
    errors = []
    per_atom_errors = []
    
    for pred in predictions:
        # Total energy error
        error = abs(pred['energy_pred'] - pred['energy_target'])
        errors.append(error)
        
        # Per-atom energy error
        per_atom_error = abs(pred['energy_per_atom_pred'] - pred['energy_per_atom_target'])
        per_atom_errors.append(per_atom_error)
    
    mae_total = sum(errors) / len(errors) if len(errors) > 0 else 0
    mae_per_atom = sum(per_atom_errors) / len(per_atom_errors) if len(per_atom_errors) > 0 else 0
    
    rmse_total = (sum(e**2 for e in errors) / len(errors))**0.5 if len(errors) > 0 else 0
    rmse_per_atom = (sum(e**2 for e in per_atom_errors) / len(per_atom_errors))**0.5 if len(per_atom_errors) > 0 else 0
    
    # 7. Print results
    logger.info("\n" + "="*80)
    logger.info("📊 QM9 EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"   Samples: {len(samples)}")
    logger.info(f"   MAE (total): {mae_total:.6f} eV")
    logger.info(f"   MAE (per-atom): {mae_per_atom:.6f} eV/atom")
    logger.info(f"   RMSE (total): {rmse_total:.6f} eV")
    logger.info(f"   RMSE (per-atom): {rmse_per_atom:.6f} eV/atom")
    logger.info("\n✅ Evaluation complete!")
    logger.info("="*80)

if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def test(
        max_samples: int = typer.Option(1000, help="Maximum number of samples to evaluate"),
        device: str = typer.Option("cuda", help="Device to use (cuda or cpu)")
    ):
        evaluate_qm9(max_samples=max_samples, device=device)
    
    app()
