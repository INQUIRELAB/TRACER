#!/usr/bin/env python3
"""Script to evaluate uncertainty using ensemble models."""

import typer
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn.uncertainty import EnsembleUncertainty, EnsemblePrediction, compute_uncertainty_threshold, save_uncertainty_analysis
from gnn.model import MACEWrapper
from graphs.periodic_graph import PeriodicGraph, GraphBatch

app = typer.Typer()


@app.command()
def main(checkpoint_dir: str = typer.Option(..., help="Directory containing ensemble checkpoints"),
         data_path: str = typer.Option(..., help="Path to test data"),
         output_path: str = typer.Option("uncertainty_analysis.json", help="Output path for uncertainty analysis"),
         pattern: str = typer.Option("final_model.pt", help="Pattern for checkpoint files"),
         device: str = typer.Option("auto", help="Device to use (auto, cpu, cuda)"),
         batch_size: int = typer.Option(1, help="Batch size for inference")) -> None:
    """Evaluate uncertainty using ensemble models.
    
    Args:
        checkpoint_dir: Directory containing ensemble checkpoint files
        data_path: Path to test data
        output_path: Output path for uncertainty analysis results
        pattern: Pattern for checkpoint files (e.g., "final_model.pt")
        device: Device to use for inference
        batch_size: Batch size for inference
    """
    typer.echo(f"Evaluating uncertainty using checkpoints from {checkpoint_dir}")
    typer.echo(f"Test data: {data_path}")
    typer.echo(f"Output: {output_path}")
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    typer.echo(f"Using device: {device}")
    
    # Initialize ensemble uncertainty estimator
    ensemble = EnsembleUncertainty(
        model_class=MACEWrapper,
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls="RealAgnosticResidualInteractionBlock",
        interaction_cls_first="RealAgnosticResidualInteractionBlock",
        num_interactions=6,
        num_elements=100,
        hidden_irreps="128x0e + 128x1o + 128x2e",
        MLP_irreps="16x0e",
        correlation=3,
        compute_stress=True,
    )
    
    # Load checkpoints
    try:
        ensemble.load_checkpoints(checkpoint_dir, pattern=pattern)
        typer.echo(f"Loaded {ensemble.get_ensemble_size()} models")
    except Exception as e:
        typer.echo(f"Error loading checkpoints: {e}")
        raise typer.Exit(1)
    
    # TODO: Load test data
    # For now, we'll create dummy test data
    typer.echo("Loading test data...")
    
    # Create dummy test structures
    test_structures = []
    structure_ids = []
    
    # This would normally load from your data format
    # For demonstration, we'll create some dummy structures
    from ase import Atoms
    from pymatgen.core import Structure
    
    # Create a simple test structure (e.g., water molecule)
    water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0.48, 0.93, 0]], 
                  cell=[10, 10, 10], pbc=True)
    test_structures.append(water)
    structure_ids.append("water_molecule")
    
    # Create periodic graph builder
    graph_builder = PeriodicGraph(cutoff_radius=5.0)
    
    # Build graph batch
    try:
        graph_batch = graph_builder.build_batch(test_structures, node_feature_dim=64)
        typer.echo(f"Created graph batch with {graph_batch.positions.shape[0]} atoms")
    except Exception as e:
        typer.echo(f"Error creating graph batch: {e}")
        raise typer.Exit(1)
    
    # Make ensemble predictions
    typer.echo("Making ensemble predictions...")
    try:
        prediction = ensemble.predict_ensemble(graph_batch, device=device)
        typer.echo("Ensemble predictions completed")
    except Exception as e:
        typer.echo(f"Error making predictions: {e}")
        raise typer.Exit(1)
    
    # Compute uncertainty metrics
    typer.echo("Computing uncertainty metrics...")
    
    # Get single structure uncertainty
    uncertainty_metrics = ensemble.predict_single_structure(graph_batch, device=device)
    
    # Print results
    typer.echo("\n=== Uncertainty Analysis Results ===")
    typer.echo(f"Energy uncertainty: {uncertainty_metrics['energy_uncertainty']:.6f} eV")
    typer.echo(f"Force uncertainty (RMS): {uncertainty_metrics['force_uncertainty_rms']:.6f} eV/Å")
    typer.echo(f"Stress uncertainty (Frobenius): {uncertainty_metrics['stress_uncertainty_frobenius']:.6f} GPa")
    typer.echo(f"Mean energy: {uncertainty_metrics['mean_energy']:.6f} eV")
    typer.echo(f"Mean force magnitude: {uncertainty_metrics['mean_force_magnitude']:.6f} eV/Å")
    
    # Save detailed results
    results = {
        'ensemble_size': ensemble.get_ensemble_size(),
        'model_info': ensemble.get_model_info(),
        'uncertainty_metrics': uncertainty_metrics,
        'predictions': {
            'energies': prediction.energies.tolist(),
            'energy_uncertainty': prediction.energy_uncertainty.tolist(),
            'energy_variance': prediction.energy_variance.tolist(),
            'force_uncertainty_rms': torch.sqrt(torch.mean(prediction.force_variance)).item(),
            'stress_uncertainty_frobenius': torch.norm(prediction.stress_uncertainty).item(),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"\nDetailed results saved to {output_path}")
    typer.echo("Uncertainty evaluation completed!")


@app.command()
def compute_thresholds(checkpoint_dir: str = typer.Option(..., help="Directory containing ensemble checkpoints"),
                      data_path: str = typer.Option(..., help="Path to validation data"),
                      output_path: str = typer.Option("uncertainty_thresholds.json", help="Output path for thresholds"),
                      percentile: float = typer.Option(95.0, help="Percentile for threshold computation"),
                      pattern: str = typer.Option("final_model.pt", help="Pattern for checkpoint files")) -> None:
    """Compute uncertainty thresholds from validation data.
    
    Args:
        checkpoint_dir: Directory containing ensemble checkpoint files
        data_path: Path to validation data
        output_path: Output path for uncertainty thresholds
        percentile: Percentile for threshold computation
        pattern: Pattern for checkpoint files
    """
    typer.echo(f"Computing uncertainty thresholds from {data_path}")
    typer.echo(f"Checkpoint directory: {checkpoint_dir}")
    typer.echo(f"Percentile: {percentile}%")
    
    # Initialize ensemble uncertainty estimator
    ensemble = EnsembleUncertainty(
        model_class=MACEWrapper,
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls="RealAgnosticResidualInteractionBlock",
        interaction_cls_first="RealAgnosticResidualInteractionBlock",
        num_interactions=6,
        num_elements=100,
        hidden_irreps="128x0e + 128x1o + 128x2e",
        MLP_irreps="16x0e",
        correlation=3,
        compute_stress=True,
    )
    
    # Load checkpoints
    try:
        ensemble.load_checkpoints(checkpoint_dir, pattern=pattern)
        typer.echo(f"Loaded {ensemble.get_ensemble_size()} models")
    except Exception as e:
        typer.echo(f"Error loading checkpoints: {e}")
        raise typer.Exit(1)
    
    # TODO: Load validation data and compute thresholds
    # This would iterate through validation data and compute uncertainty thresholds
    
    typer.echo("Computing uncertainty thresholds...")
    
    # For demonstration, create dummy thresholds
    dummy_thresholds = {
        'energy_uncertainty_threshold': 0.1,  # eV
        'force_uncertainty_threshold': 0.05,  # eV/Å
        'stress_uncertainty_threshold': 0.02,  # GPa
        'percentile': percentile,
        'ensemble_size': ensemble.get_ensemble_size(),
    }
    
    # Save thresholds
    with open(output_path, 'w') as f:
        json.dump(dummy_thresholds, f, indent=2)
    
    typer.echo(f"Uncertainty thresholds saved to {output_path}")
    typer.echo("Threshold computation completed!")


if __name__ == "__main__":
    app()



