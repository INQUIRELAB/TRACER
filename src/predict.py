#!/usr/bin/env python3
"""Prediction utility for trained GNN models."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from gnn.model import MACEWrapper
from gnn.train import GNNTrainer
from graphs.periodic_graph import PeriodicGraph, GraphBatch


class GNNPredictor:
    """Predictor for trained GNN models."""
    
    def __init__(self, model_path: Union[str, Path], device: str = "auto") -> None:
        """Initialize GNN predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        self.graph_builder = PeriodicGraph(cutoff_radius=5.0)
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device.
        
        Args:
            device: Device specification
            
        Returns:
            PyTorch device
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(self) -> None:
        """Load the trained model from checkpoint."""
        console = Console()
        console.print(f"[blue]Loading model from {self.model_path}[/blue]")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model (you may need to adjust parameters based on your training)
        self.model = MACEWrapper(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=3,
            num_interactions=6,
            num_elements=100,
            hidden_irreps="128x0e + 128x1o + 128x2e",
            MLP_irreps="16x0e",
            compute_stress=True
        )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        console.print(f"[green]Model loaded successfully![/green]")
        console.print(f"Epoch: {epoch}")
        console.print(f"Best validation loss: {best_val_loss}")
    
    def predict_structures(self, structures: List[Union[Atoms, Structure]], 
                          batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Predict energies, forces, and stress for a list of structures.
        
        Args:
            structures: List of ASE Atoms or pymatgen Structure objects
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            self.load_model()
        
        console = Console()
        console.print(f"[blue]Predicting for {len(structures)} structures...[/blue]")
        
        all_energies = []
        all_forces = []
        all_stress = []
        all_features = []
        
        # Process in batches
        with Progress() as progress:
            task = progress.add_task("Processing structures...", total=len(structures))
            
            for i in range(0, len(structures), batch_size):
                batch_structures = structures[i:i + batch_size]
                
                # Build graph batch
                batch = self.graph_builder.build_batch(batch_structures)
                batch = batch.to(self.device)
                
                # Predict
                with torch.no_grad():
                    energies, forces, stress, features = self.model(batch)
                
                # Convert to numpy and store
                all_energies.append(energies.cpu().numpy())
                all_forces.append(forces.cpu().numpy())
                all_stress.append(stress.cpu().numpy())
                all_features.append(features.cpu().numpy())
                
                progress.update(task, advance=len(batch_structures))
        
        # Concatenate results
        results = {
            'energies': np.concatenate(all_energies),
            'forces': np.concatenate(all_forces),
            'stress': np.concatenate(all_stress),
            'features': np.concatenate(all_features),
        }
        
        console.print("[green]Prediction completed![/green]")
        return results
    
    def predict_single_structure(self, structure: Union[Atoms, Structure]) -> Dict[str, np.ndarray]:
        """Predict for a single structure.
        
        Args:
            structure: ASE Atoms or pymatgen Structure object
            
        Returns:
            Dictionary with predictions
        """
        return self.predict_structures([structure], batch_size=1)
    
    def save_predictions(self, results: Dict[str, np.ndarray], 
                        output_path: Union[str, Path]) -> None:
        """Save predictions to file.
        
        Args:
            results: Prediction results
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.npz':
            np.savez(output_path, **results)
        elif output_path.suffix == '.npy':
            # Save as separate files
            for key, value in results.items():
                np.save(output_path.parent / f"{output_path.stem}_{key}.npy", value)
        else:
            raise ValueError("Output file must have .npz or .npy extension")
        
        console = Console()
        console.print(f"[green]Predictions saved to {output_path}[/green]")


def main():
    """Main prediction script."""
    parser = argparse.ArgumentParser(description="Predict with trained GNN model")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", required=True, help="Input structure file(s)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--format", default="auto", 
                       help="Input file format (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = GNNPredictor(args.model, device=args.device)
    
    # Load structures
    console = Console()
    console.print(f"[blue]Loading structures from {args.input}[/blue]")
    
    structures = []
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        if args.format == "auto":
            if input_path.suffix.lower() in ['.xyz', '.pdb', '.cif']:
                from ase.io import read
                structures = read(input_path, ':')
            elif input_path.suffix.lower() == '.json':
                from pymatgen.io.ase import AseAtomsAdaptor
                import json
                with open(input_path) as f:
                    data = json.load(f)
                for item in data:
                    structure = Structure.from_dict(item)
                    structures.append(AseAtomsAdaptor.get_atoms(structure))
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
        else:
            # Use specified format
            from ase.io import read
            structures = read(input_path, format=args.format, ':')
    else:
        # Directory of files
        for file_path in input_path.glob("*"):
            if file_path.suffix.lower() in ['.xyz', '.pdb', '.cif']:
                from ase.io import read
                structures.extend(read(file_path, ':'))
    
    if not structures:
        console.print("[red]No structures found![/red]")
        return
    
    console.print(f"[green]Loaded {len(structures)} structures[/green]")
    
    # Predict
    results = predictor.predict_structures(structures, batch_size=args.batch_size)
    
    # Display summary
    table = Table(title="Prediction Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Shape", style="magenta")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    
    for key, value in results.items():
        if value.ndim > 0:
            table.add_row(
                key,
                str(value.shape),
                f"{np.mean(value):.6f}",
                f"{np.std(value):.6f}"
            )
    
    console.print(table)
    
    # Save results
    predictor.save_predictions(results, args.output)


if __name__ == "__main__":
    main()



