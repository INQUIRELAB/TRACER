#!/usr/bin/env python3
"""
VASP File Processing and Pipeline Testing Script

This script processes VASP files (POSCAR/CONTCAR) and their corresponding
formation energies to test our DFT→GNN→QNN pipeline.

Usage:
    python scripts/test_vasp_pipeline.py --vasp-dir /path/to/vasp/files --results-file results.json
"""

import sys
import os
sys.path.append('src')

import typer
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from ase import Atoms
from ase.io import read
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="VASP Pipeline Testing Tool")

def parse_vasp_file(vasp_path: str) -> Optional[Dict[str, Any]]:
    """Parse VASP file and extract atomic structure."""
    try:
        # Read VASP file using ASE
        atoms = read(vasp_path)
        
        # Extract structure information
        structure_data = {
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions().tolist(),
            'cell': atoms.get_cell().array.tolist(),
            'pbc': atoms.get_pbc().tolist(),
            'n_atoms': len(atoms),
            'chemical_formula': atoms.get_chemical_formula(),
            'volume': atoms.get_volume(),
            'density': atoms.get_masses().sum() / atoms.get_volume()
        }
        
        return structure_data
        
    except Exception as e:
        logger.error(f"Failed to parse {vasp_path}: {e}")
        return None

def create_pytorch_geometric_data(structure_data: Dict[str, Any]) -> torch.Tensor:
    """Convert structure data to PyTorch Geometric format."""
    try:
        from torch_geometric.data import Data
        import torch
        
        # Extract atomic numbers and positions
        symbols = structure_data['symbols']
        positions = torch.tensor(structure_data['positions'], dtype=torch.float32)
        
        # Convert symbols to atomic numbers
        atomic_numbers = []
        for symbol in symbols:
            atomic_numbers.append(get_atomic_number(symbol))
        
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        batch = torch.zeros(len(z), dtype=torch.long)  # Single molecule
        
        # Create PyTorch Geometric Data object
        data = Data(z=z, pos=positions, batch=batch)
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to create PyTorch Geometric data: {e}")
        return None

def get_atomic_number(symbol: str) -> int:
    """Get atomic number from chemical symbol."""
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86
    }
    return periodic_table.get(symbol, 1)  # Default to H if not found

def load_formation_energies(energy_file: str) -> Dict[str, float]:
    """Load formation energies from file."""
    try:
        energies = {}
        
        if energy_file.endswith('.csv'):
            df = pd.read_csv(energy_file)
            # Assume first column is structure ID, second is formation energy
            for _, row in df.iterrows():
                structure_id = str(row.iloc[0])
                formation_energy = float(row.iloc[1])
                energies[structure_id] = formation_energy
                
        elif energy_file.endswith('.json'):
            with open(energy_file, 'r') as f:
                data = json.load(f)
                energies = data
                
        elif energy_file.endswith('.txt'):
            with open(energy_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        structure_id = parts[0]
                        formation_energy = float(parts[1])
                        energies[structure_id] = formation_energy
        
        return energies
        
    except Exception as e:
        logger.error(f"Failed to load formation energies: {e}")
        return {}

@app.command()
def test_pipeline(
    vasp_dir: str = typer.Option(..., "--vasp-dir", "-d", help="Directory containing VASP files"),
    energy_file: str = typer.Option(..., "--energy-file", "-e", help="File containing formation energies"),
    output_file: str = typer.Option("vasp_test_results.json", "--output", "-o", help="Output results file"),
    model_path: str = typer.Option("models/gnn_training_enhanced/ensemble/", "--model-path", "-m", help="Path to trained models"),
    use_delta_head: bool = typer.Option(True, "--use-delta-head", help="Use delta head for corrections")
):
    """Test pipeline on VASP files with known formation energies."""
    try:
        from pipeline.run import HybridPipeline
        import omegaconf
        
        # Initialize pipeline
        config = omegaconf.DictConfig({
            'data': {'max_samples': 1000},
            'gnn': {'model_type': 'schnet', 'hidden_dim': 256},
            'quantum': {'backend': 'simulator', 'max_steps': 100},
            'pipeline': {'use_delta_head': use_delta_head, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        })
        
        pipeline = HybridPipeline(config)
        logger.info("Pipeline initialized for VASP testing")
        
        # Load formation energies
        formation_energies = load_formation_energies(energy_file)
        logger.info(f"Loaded {len(formation_energies)} formation energies")
        
        # Find VASP files
        vasp_dir_path = Path(vasp_dir)
        vasp_files = []
        
        for pattern in ['*.vasp', 'POSCAR*', 'CONTCAR*']:
            vasp_files.extend(vasp_dir_path.glob(pattern))
        
        logger.info(f"Found {len(vasp_files)} VASP files")
        
        # Process each VASP file
        results = []
        successful_predictions = 0
        
        for i, vasp_file in enumerate(vasp_files):
            try:
                # Parse VASP file
                structure_data = parse_vasp_file(str(vasp_file))
                if structure_data is None:
                    continue
                
                # Create PyTorch Geometric data
                data = create_pytorch_geometric_data(structure_data)
                if data is None:
                    continue
                
                # Get structure ID (use filename)
                structure_id = vasp_file.stem
                
                # Get ground truth formation energy
                ground_truth = formation_energies.get(structure_id, None)
                if ground_truth is None:
                    logger.warning(f"No formation energy found for {structure_id}")
                    continue
                
                # Run prediction
                prediction = pipeline.predict_with_model(data)
                
                # Extract predicted energy
                predicted_energy = prediction.get('energy', 0.0)
                uncertainty = prediction.get('uncertainty', 0.0)
                
                # Calculate error
                error = abs(predicted_energy - ground_truth)
                relative_error = error / abs(ground_truth) * 100 if ground_truth != 0 else 0
                
                result = {
                    'structure_id': structure_id,
                    'vasp_file': str(vasp_file),
                    'n_atoms': structure_data['n_atoms'],
                    'chemical_formula': structure_data['chemical_formula'],
                    'ground_truth_energy': ground_truth,
                    'predicted_energy': predicted_energy,
                    'uncertainty': uncertainty,
                    'absolute_error': error,
                    'relative_error_percent': relative_error,
                    'use_delta_head': use_delta_head
                }
                
                results.append(result)
                successful_predictions += 1
                
                logger.info(f"Processed {structure_id}: Error = {error:.4f} eV ({relative_error:.2f}%)")
                
            except Exception as e:
                logger.error(f"Failed to process {vasp_file}: {e}")
                continue
        
        # Calculate summary statistics
        if results:
            errors = [r['absolute_error'] for r in results]
            relative_errors = [r['relative_error_percent'] for r in results]
            
            summary = {
                'total_structures': len(results),
                'successful_predictions': successful_predictions,
                'mae_eV': np.mean(errors),
                'rmse_eV': np.sqrt(np.mean([e**2 for e in errors])),
                'max_error_eV': np.max(errors),
                'min_error_eV': np.min(errors),
                'mean_relative_error_percent': np.mean(relative_errors),
                'max_relative_error_percent': np.max(relative_errors),
                'model_path': model_path,
                'use_delta_head': use_delta_head
            }
        else:
            summary = {'error': 'No successful predictions'}
        
        # Save results
        output_data = {
            'summary': summary,
            'detailed_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print summary
        typer.echo(f"\\n🎯 VASP Pipeline Test Results:")
        typer.echo(f"   Structures processed: {successful_predictions}")
        typer.echo(f"   MAE: {summary.get('mae_eV', 0):.4f} eV")
        typer.echo(f"   RMSE: {summary.get('rmse_eV', 0):.4f} eV")
        typer.echo(f"   Mean relative error: {summary.get('mean_relative_error_percent', 0):.2f}%")
        typer.echo(f"   Results saved to: {output_file}")
        
        if summary.get('mae_eV', 0) < 0.1:
            typer.echo(f"   🎉 EXCELLENT accuracy!")
        elif summary.get('mae_eV', 0) < 0.5:
            typer.echo(f"   ✅ GOOD accuracy!")
        elif summary.get('mae_eV', 0) < 1.0:
            typer.echo(f"   ⚠️  MODERATE accuracy - consider delta head corrections")
        else:
            typer.echo(f"   ❌ POOR accuracy - pipeline needs improvement")
        
    except Exception as e:
        logger.error(f"VASP testing failed: {e}")
        typer.echo(f"❌ VASP testing failed: {e}")
        raise typer.Exit(1)

@app.command()
def create_sample_energy_file(
    vasp_dir: str = typer.Option(..., "--vasp-dir", "-d", help="Directory containing VASP files"),
    output_file: str = typer.Option("sample_energies.txt", "--output", "-o", help="Output energy file")
):
    """Create a sample energy file template for your VASP files."""
    try:
        # Find VASP files
        vasp_dir_path = Path(vasp_dir)
        vasp_files = []
        
        for pattern in ['*.vasp', 'POSCAR*', 'CONTCAR*']:
            vasp_files.extend(vasp_dir_path.glob(pattern))
        
        logger.info(f"Found {len(vasp_files)} VASP files")
        
        # Create sample energy file
        with open(output_file, 'w') as f:
            f.write("# Sample formation energy file\\n")
            f.write("# Format: structure_id formation_energy_eV\\n")
            f.write("# Replace the placeholder values with your actual VASP formation energies\\n\\n")
            
            for vasp_file in vasp_files:
                structure_id = vasp_file.stem
                # Parse to get chemical formula
                try:
                    atoms = read(str(vasp_file))
                    formula = atoms.get_chemical_formula()
                    f.write(f"{structure_id} 0.0  # {formula} - Replace with actual formation energy\\n")
                except:
                    f.write(f"{structure_id} 0.0  # Replace with actual formation energy\\n")
        
        typer.echo(f"✅ Sample energy file created: {output_file}")
        typer.echo(f"   Found {len(vasp_files)} VASP files")
        typer.echo(f"   Please edit the file and replace 0.0 with your actual formation energies")
        
    except Exception as e:
        logger.error(f"Failed to create sample energy file: {e}")
        typer.echo(f"❌ Failed to create sample energy file: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()

