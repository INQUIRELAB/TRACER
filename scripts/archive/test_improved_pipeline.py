#!/usr/bin/env python3
"""
Test the improved pipeline with the new delta head on VASP structures.
"""

import sys
sys.path.append('src')

import torch
from ase.io import read
from pathlib import Path
import logging
from omegaconf import DictConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_structure(poscar_path: str, structure_name: str):
    """Test a single VASP structure."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔬 Testing {structure_name}")
    logger.info(f"{'='*70}")
    
    # Load structure
    atoms = read(poscar_path)
    logger.info(f"✅ Loaded: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    
    # Create test sample
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    positions = atoms.get_positions().tolist()
    
    sample = {
        'sample_id': structure_name,
        'domain': 'jarvis',
        'num_atoms': len(atoms),
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'energy_target': 0.0,
        'forces_target': [[0.0, 0.0, 0.0]] * len(atoms)
    }
    
    # Initialize pipeline
    config = DictConfig({"pipeline": {"device": "cpu"}})
    from pipeline.run import HybridPipeline
    pipeline = HybridPipeline(config)
    
    # Load GNN model
    gnn_model_path = 'models/gnn_training_enhanced/ensemble/ckpt_0.pt'
    gnn_model = pipeline.load_model(gnn_model_path)
    logger.info("✅ Loaded GNN model")
    
    # Run GNN prediction
    gnn_predictions = pipeline.predict_with_model(gnn_model, [sample])
    gnn_pred = gnn_predictions[0]
    
    logger.info(f"\n📊 GNN BASELINE:")
    logger.info(f"   Energy: {gnn_pred['energy_pred']:.6f} eV")
    logger.info(f"   Per-atom: {gnn_pred['energy_per_atom_pred']:.6f} eV/atom")
    
    # Load IMPROVED delta head
    delta_head_path = 'artifacts/delta_head_improved/best_model.pt'
    delta_head = pipeline.load_delta_head(delta_head_path)
    logger.info("✅ Loaded IMPROVED delta head")
    
    # Apply delta head correction
    corrected_predictions = pipeline.apply_delta_head(
        delta_head, gnn_predictions, gnn_model=gnn_model
    )
    corrected_pred = corrected_predictions[0]
    
    logger.info(f"\n🎯 IMPROVED PREDICTION:")
    logger.info(f"   Original: {gnn_pred['energy_pred']:.6f} eV")
    logger.info(f"   Delta correction: {corrected_pred['delta_correction']:.6f} eV")
    logger.info(f"   Corrected: {corrected_pred['energy_pred']:.6f} eV")
    logger.info(f"   Corrected per-atom: {corrected_pred['energy_per_atom_pred']:.6f} eV/atom")
    
    return {
        'name': structure_name,
        'formula': atoms.get_chemical_formula(),
        'atoms': len(atoms),
        'gnn_energy': gnn_pred['energy_pred'],
        'gnn_per_atom': gnn_pred['energy_per_atom_pred'],
        'delta_correction': corrected_pred['delta_correction'],
        'final_energy': corrected_pred['energy_pred'],
        'final_per_atom': corrected_pred['energy_per_atom_pred']
    }


def main():
    """Run pipeline on all VASP structures."""
    
    logger.info("🚀 TESTING IMPROVED PIPELINE ON VASP STRUCTURES")
    logger.info("="*70)
    
    # List of VASP structures
    vasp_structures = [
        ('test_data/Mn58_POSCAR', 'Mn58'),
        ('test_data/O16_POSCAR', 'O16'),
        ('test_data/Li9_POSCAR', 'Li9'),
        ('test_data/V2_POSCAR', 'V2'),
        ('test_data/Cr5O10_POSCAR', 'Cr5O10'),
        ('test_data/V3O8_POSCAR', 'V3O8'),
        ('test_data/Li4Co4O8_POSCAR', 'Li4Co4O8'),
        ('test_data/Li1Mn3O8_POSCAR', 'Li1Mn3O8'),
        ('test_data/V2O4_POSCAR', 'V2O4'),
        ('test_data/V5O11_POSCAR', 'V5O11'),
    ]
    
    results = []
    
    for poscar_path, structure_name in vasp_structures:
        try:
            if Path(poscar_path).exists():
                result = test_structure(poscar_path, structure_name)
                results.append(result)
            else:
                logger.warning(f"⚠️  File not found: {poscar_path}")
        except Exception as e:
            logger.error(f"❌ Failed to process {structure_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("📊 RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Structure':<15} {'Formula':<12} {'Atoms':<6} {'GNN/atom':<12} {'Delta':<10} {'Final/atom':<12}")
    logger.info(f"{'-'*70}")
    
    for result in results:
        logger.info(f"{result['name']:<15} {result['formula']:<12} {result['atoms']:<6} "
                   f"{result['gnn_per_atom']:<12.6f} {result['delta_correction']:<10.6f} "
                   f"{result['final_per_atom']:<12.6f}")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ Pipeline testing completed!")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()

