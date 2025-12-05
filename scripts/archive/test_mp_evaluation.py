#!/usr/bin/env python3
"""
Materials Project Evaluation Script
Tests the pipeline on MP dataset (proper test - same PBE DFT level)
"""

import sys
import os
import torch
from pathlib import Path
import logging
from omegaconf import DictConfig
import json
from ase.io import read

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.run import HybridPipeline
from gnn.model import SchNetWrapper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_mp_data(mp_data_path: str, max_samples: int = 1000):
    """Load Materials Project data from JSON"""
    logger.info(f"📥 Loading MP data from {mp_data_path}")
    
    with open(mp_data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"✅ Loaded {len(data)} entries")
    
    samples = []
    for i, entry in enumerate(data[:max_samples]):
        # Parse MP data structure
        mp_id = entry.get('material_id', f'mp_{i}')
        structure = entry.get('structure', {})
        
        # Extract atomic information
        if 'sites' in structure:
            sites = structure['sites']
            atomic_numbers = []
            positions = []
            
            for site in sites:
                species = site.get('species', [])
                if len(species) > 0:
                    species_name = species[0].get('element')
                    if species_name:
                        from ase.data import chemical_symbols
                        if species_name in chemical_symbols:
                            atomic_numbers.append(chemical_symbols.index(species_name))
                coords = site.get('xyz', [0, 0, 0])
                positions.append(coords)
        else:
            logger.warning(f"Invalid structure for {mp_id}, skipping")
            continue
        
        # Get total energy
        total_energy = entry.get('formation_energy_per_atom', 0) * len(atomic_numbers)
        
        # Create sample
        sample = {
            'sample_id': mp_id,
            'domain': 'jarvis',  # Use jarvis as base domain
            'num_atoms': len(atomic_numbers),
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'energy_target': total_energy,  # Total energy in eV
            'energy_per_atom_target': entry.get('formation_energy_per_atom', 0),
            'forces_target': [[0.0, 0.0, 0.0]] * len(atomic_numbers)
        }
        samples.append(sample)
    
    logger.info(f"✅ Prepared {len(samples)} samples")
    return samples

def evaluate_mp(
    mp_data_path: str,
    max_samples: int = 1000,
    device: str = "cuda"
):
    """Evaluate model on Materials Project dataset"""
    
    logger.info("🚀 MATERIALS PROJECT EVALUATION")
    logger.info("="*80)
    logger.info(f"   Data: {mp_data_path}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Device: {device}")
    
    # 1. Load MP data
    samples = load_mp_data(mp_data_path, max_samples)
    
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
    
    # 3. Run predictions
    logger.info("\n🔬 Running predictions...")
    gnn_predictions = pipeline.predict_with_model(gnn_model, samples)
    
    # 4. Apply delta head if available
    if use_delta:
        logger.info("\n🔬 Applying quantum corrections...")
        corrected_predictions = pipeline.apply_delta_head(delta_head, gnn_predictions, gnn_model=gnn_model)
        predictions = corrected_predictions
    else:
        predictions = gnn_predictions
    
    # 5. Calculate metrics
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
    
    # 6. Print results
    logger.info("\n" + "="*80)
    logger.info("📊 MATERIALS PROJECT EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"   Samples: {len(samples)}")
    logger.info(f"   MAE (total): {mae_total:.6f} eV")
    logger.info(f"   MAE (per-atom): {mae_per_atom:.6f} eV/atom")
    logger.info(f"   RMSE (total): {rmse_total:.6f} eV")
    logger.info(f"   RMSE (per-atom): {rmse_per_atom:.6f} eV/atom")
    logger.info("\n✅ Evaluation complete!")
    logger.info("="*80)
    
    return {
        'mae_total': mae_total,
        'mae_per_atom': mae_per_atom,
        'rmse_total': rmse_total,
        'rmse_per_atom': rmse_per_atom,
    }

if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def test(
        mp_data_path: str = typer.Option("data/mp/materials_data.json", help="Path to MP data"),
        max_samples: int = typer.Option(1000, help="Maximum number of samples to evaluate"),
        device: str = typer.Option("cuda", help="Device to use (cuda or cpu)")
    ):
        evaluate_mp(mp_data_path=mp_data_path, max_samples=max_samples, device=device)
    
    app()


