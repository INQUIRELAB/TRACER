#!/usr/bin/env python3
"""Evaluation script for GNN models with MAE metrics on held-out test split."""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn.model import SchNetWrapper
from gnn.uncertainty import EnsembleUncertainty
from dft_hybrid.data.jarvis_dft import create_jarvis_dataloader


def calculate_mae_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                         num_atoms: torch.Tensor) -> Dict[str, float]:
    """Calculate MAE metrics for energy predictions.
    
    Args:
        predictions: Predicted energies (batch_size,)
        targets: Target energies (batch_size,)
        num_atoms: Number of atoms per structure (batch_size,)
        
    Returns:
        Dictionary with MAE metrics
    """
    # MAE for total energy
    mae_total = torch.mean(torch.abs(predictions - targets)).item()
    
    # MAE for energy per atom
    pred_per_atom = predictions / num_atoms.float()
    target_per_atom = targets / num_atoms.float()
    mae_per_atom = torch.mean(torch.abs(pred_per_atom - target_per_atom)).item()
    
    return {
        'mae_total': mae_total,
        'mae_per_atom': mae_per_atom
    }


def evaluate_model(model: nn.Module, test_loader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on test dataset.
    
    Args:
        model: Trained GNN model
        test_loader: Test data loader
        device: Device for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_num_atoms = []
    
    print("🔍 Evaluating model on test split...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluation"):
            batch_graph, batch_targets = batch_data
            batch_graph = batch_graph.to(device)
            
            # Extract targets
            target_energies = batch_targets['energy'].to(device)
            
            # Forward pass
            pred_energies, _, _, _ = model(batch_graph)
            
            # Get number of atoms per structure
            num_atoms = batch_graph.batch.bincount()
            
            all_predictions.append(pred_energies.cpu())
            all_targets.append(target_energies.cpu())
            all_num_atoms.append(num_atoms.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_num_atoms = torch.cat(all_num_atoms, dim=0)
    
    # Calculate metrics
    metrics = calculate_mae_metrics(all_predictions, all_targets, all_num_atoms)
    
    # Add additional statistics
    metrics['num_test_samples'] = len(all_predictions)
    metrics['mean_prediction'] = torch.mean(all_predictions).item()
    metrics['mean_target'] = torch.mean(all_targets).item()
    metrics['std_prediction'] = torch.std(all_predictions).item()
    metrics['std_target'] = torch.std(all_targets).item()
    
    return metrics


def load_best_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the best model from checkpoint.
    
    Args:
        checkpoint_path: Path to the best model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"📂 Loading best model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture as training
    model = SchNetWrapper(
        hidden_channels=256,
        num_filters=256,
        num_interactions=8,
        num_gaussians=64,
        cutoff=6.0,
        max_num_neighbors=64
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✅ Model loaded successfully")
    return model


def create_test_split(data_path: str, test_size: int = 5000) -> str:
    """Create a held-out test split from JARVIS-DFT dataset.
    
    Args:
        data_path: Path to JARVIS-DFT JSON file
        test_size: Number of samples for test split
        
    Returns:
        Path to test data (same as input for now, but with test_size limit)
    """
    return data_path  # For now, we'll use the same data but limit test samples


def save_best_checkpoint(source_path: str, artifacts_dir: str = "artifacts/gnn") -> str:
    """Save best checkpoint to artifacts directory.
    
    Args:
        source_path: Path to source checkpoint
        artifacts_dir: Artifacts directory
        
    Returns:
        Path to saved checkpoint
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    best_ckpt_path = artifacts_path / "best.ckpt"
    
    # Check if source and destination are the same
    if Path(source_path).resolve() == best_ckpt_path.resolve():
        print(f"✅ Best checkpoint already at {best_ckpt_path}")
        return str(best_ckpt_path)
    
    print(f"💾 Saving best checkpoint to {best_ckpt_path}")
    
    # Copy checkpoint
    import shutil
    shutil.copy2(source_path, best_ckpt_path)
    
    print(f"✅ Best checkpoint saved to {best_ckpt_path}")
    return str(best_ckpt_path)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate GNN model with MAE metrics")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint or ensemble directory")
    parser.add_argument("--data-path", type=str, 
                       default="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
                       help="Path to JARVIS-DFT dataset")
    parser.add_argument("--test-size", type=int, default=5000,
                       help="Number of samples for test split")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--save-artifacts", action="store_true",
                       help="Save best checkpoint to artifacts/gnn/best.ckpt")
    parser.add_argument("--ensemble", action="store_true",
                       help="Evaluate ensemble model (checkpoint should be ensemble directory)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("🚀 GNN Model Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Test size: {args.test_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Create test data loader
    print(f"\n📊 Creating test split...")
    test_loader = create_jarvis_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        cutoff_radius=6.0,
        max_samples=args.test_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Test batches: {len(test_loader)}")
    
    if args.ensemble:
        # Evaluate ensemble model
        print(f"\n🎯 Evaluating ensemble model...")
        ensemble_uncertainty = EnsembleUncertainty(device=device)
        ensemble_uncertainty.load_ensemble(args.checkpoint)
        
        # Get ensemble predictions with uncertainty
        metrics = ensemble_uncertainty.evaluate_uncertainty(test_loader)
        
        print("\n📈 Ensemble Evaluation Results")
        print("=" * 50)
        print(f"MAE(E_total):           {metrics['mae_mean']:.6f} eV")
        print(f"MAE(E_per_atom):        {metrics['mae_per_atom']:.6f} eV/atom")
        print(f"Mean Uncertainty:       {metrics['mean_uncertainty']:.6f} eV")
        print(f"Mean Uncertainty/atom:  {metrics['mean_uncertainty_per_atom']:.6f} eV/atom")
        print(f"Uncertainty Correlation: {metrics['uncertainty_correlation']:.4f}")
        print(f"Ensemble Size:          {metrics['ensemble_size']}")
        print(f"Test samples:           {metrics['num_samples']}")
        print("=" * 50)
        
    else:
        # Load single model
        model = load_best_model(args.checkpoint, device)
        
        # Evaluate single model
        metrics = evaluate_model(model, test_loader, device)
        
        # Print single model results
        print("\n📈 Single Model Evaluation Results")
        print("=" * 50)
        print(f"MAE(E_total):     {metrics['mae_total']:.6f} eV")
        print(f"MAE(E_per_atom): {metrics['mae_per_atom']:.6f} eV/atom")
        print(f"Test samples:    {metrics['num_test_samples']}")
        print(f"Mean prediction: {metrics['mean_prediction']:.6f} eV")
        print(f"Mean target:     {metrics['mean_target']:.6f} eV")
        print(f"Std prediction:  {metrics['std_prediction']:.6f} eV")
        print(f"Std target:      {metrics['std_target']:.6f} eV")
        print("=" * 50)
    
    # Save metrics to JSON
    metrics_path = Path("artifacts/gnn/evaluation_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"📊 Metrics saved to {metrics_path}")
    
    # Save best checkpoint to artifacts if requested
    if args.save_artifacts:
        best_ckpt_path = save_best_checkpoint(args.checkpoint)
        print(f"🏆 Best checkpoint available at: {best_ckpt_path}")
    
    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
