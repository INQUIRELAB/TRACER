#!/usr/bin/env python3
"""
Materials Project Trajectory (MPtrj) Evaluation Script
Tests the pipeline on MPtrj dataset (proper test - same PBE DFT level)
"""

import sys
import os
import torch
from pathlib import Path
import logging
from omegaconf import DictConfig
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.run import HybridPipeline
from gnn.model import SchNetWrapper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_mptrj_data(mptrj_data_path: str, max_samples: int = 1000):
    """Load MPtrj data from JSON"""
    logger.info(f"📥 Loading MPtrj data from {mptrj_data_path}")
    
    with open(mptrj_data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"✅ Loaded {len(data)} entries")
    
    samples = []
    sample_count = 0
    
    for mp_id, material_data in data.items():
        if sample_count >= max_samples:
            break
        
        # MPtrj has nested structure: mp_id -> trajectory_id -> structure
        for traj_id, trajectory_data in material_data.items():
            if sample_count >= max_samples:
                break
                
            # Extract structure from PyMatGen format
            structure = trajectory_data.get('structure', {})
            sites = structure.get('sites', [])
            
            atomic_numbers = []
            positions = []
            
            for site in sites:
                species = site.get('species', [])
                if len(species) > 0:
                    element = species[0].get('element', '')
                    if element:
                        from ase.data import chemical_symbols
                        if element in chemical_symbols:
                            atomic_numbers.append(chemical_symbols.index(element))
                
                # Get Cartesian positions
                xyz = site.get('xyz', [0, 0, 0])
                positions.append(xyz)
            
            if len(atomic_numbers) == 0 or len(positions) == 0:
                continue
            
            # Try to get energy from various possible locations
            # MPtrj uses 'corrected_total_energy' (corrected for vdW and other corrections)
            energy_total = trajectory_data.get('corrected_total_energy', 0.0)
            if energy_total == 0:
                energy_total = trajectory_data.get('uncorrected_total_energy', 0.0)
            if energy_total == 0:
                energy_total = trajectory_data.get('output', {}).get('final_energy', 0.0)
            if energy_total == 0:
                energy_total = trajectory_data.get('energy', 0.0)
            
            if len(atomic_numbers) == 0 or len(positions) == 0:
                continue
            
            # Model predicts FORMATION energy per atom, not total energy!
            # Use formation energy if available, otherwise convert from total
            if 'formation_energy_per_atom' in trajectory_data:
                # Use formation energy directly
                formation_energy_pa = trajectory_data['formation_energy_per_atom']
                energy_target = formation_energy_pa * len(atomic_numbers)  # Total formation
            else:
                # Estimate formation energy from total
                # This is approximate - would need atomic energies for exact calculation
                # For now, use energy_per_atom as proxy for formation energy
                energy_per_atom = energy_total / len(atomic_numbers)
                formation_energy_pa = energy_per_atom  # Approximation
                energy_target = formation_energy_pa * len(atomic_numbers)
            
            # Create sample
            sample = {
                'sample_id': f"{mp_id}_{traj_id}",
                'domain': 'jarvis',  # Use jarvis as base domain
                'num_atoms': len(atomic_numbers),
                'atomic_numbers': atomic_numbers,
                'positions': positions,
                'energy_target': energy_target,  # Total formation energy
                'energy_per_atom_target': energy_target / len(atomic_numbers) if len(atomic_numbers) > 0 else 0,
                'forces_target': [[0.0, 0.0, 0.0]] * len(atomic_numbers)
            }
            samples.append(sample)
            sample_count += 1
    
    logger.info(f"✅ Prepared {len(samples)} samples")
    return samples

def evaluate_mptrj(
    mptrj_data_path: str,
    max_samples: int = 1000,
    device: str = "cuda"
):
    """Evaluate model on MPtrj dataset"""
    
    logger.info("🚀 MATERIALS PROJECT TRAJECTORY (MPTRJ) EVALUATION")
    logger.info("="*80)
    logger.info(f"   Data: {mptrj_data_path}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Device: {device}")
    
    # 1. Load MPtrj data
    samples = load_mptrj_data(mptrj_data_path, max_samples)
    
    if len(samples) == 0:
        logger.error("❌ No samples loaded. Cannot evaluate.")
        return
    
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
        # Total energy error (this is what pipeline predicts)
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
    logger.info("📊 MPTRJ EVALUATION RESULTS")
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
        mptrj_data_path: str = typer.Option("data/mptrj/mptrj_sample.json", help="Path to MPtrj data"),
        max_samples: int = typer.Option(100, help="Maximum number of samples to evaluate"),
        device: str = typer.Option("cuda", help="Device to use (cuda or cpu)")
    ):
        evaluate_mptrj(mptrj_data_path=mptrj_data_path, max_samples=max_samples, device=device)
    
    app()

