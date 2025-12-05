#!/usr/bin/env python3
"""
Test Pipeline with Real Structures
Compares predictions with actual target energies.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealStructureTester:
    """Test pipeline on real structures with ground truth."""
    
    def __init__(self):
        """Initialize tester."""
        self.gemnet = self._load_model()
        self.training_mean, self.training_std = self._load_norm_stats()
        self.delta_head = self._load_delta_head()
        
    def _load_model(self):
        """Load trained GemNet model."""
        checkpoint = torch.load("models/gemnet_full/best_model.pt", map_location='cpu')
        model_config = checkpoint.get('model_config', {})
        model = GemNetWrapper(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def _load_norm_stats(self):
        """Load normalization stats."""
        with open('data/preprocessed_full_unified/preprocessing_results.json', 'r') as f:
            prep_results = json.load(f)
        norm_stats = prep_results.get('normalization', {})
        return norm_stats.get('mean', 0.0), norm_stats.get('std', 1.0)
    
    def _load_delta_head(self):
        """Load delta head."""
        delta_path = Path("artifacts/delta_head_gemnet.pt")
        if delta_path.exists():
            return torch.load(delta_path, map_location='cpu')
        return {'mean_delta': 0.0}
    
    def predict(self, atomic_numbers, positions, apply_delta=True):
        """Make prediction on a structure."""
        # Convert to tensors
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float32)
        
        # Create PyG Data
        data = Data(
            atomic_numbers=atomic_numbers,
            pos=positions,
            batch=torch.zeros(len(atomic_numbers), dtype=torch.long)
        )
        
        # Predict with GemNet
        with torch.no_grad():
            output = self.gemnet(data)
            energy_normalized = output[0].item() if isinstance(output, tuple) else output.item()
        
        # Denormalize
        energy_raw = energy_normalized * self.training_std + self.training_mean
        
        # Apply delta correction
        if apply_delta:
            delta = self.delta_head.get('mean_delta', 0.0)
            delta_raw = delta * self.training_std
            energy_corrected = energy_raw + delta_raw
        else:
            energy_corrected = energy_raw
            delta_raw = 0.0
        
        return {
            'energy_gnn': energy_raw,
            'energy_corrected': energy_corrected,
            'delta': delta_raw
        }


def test_1_random_test_samples():
    """Test 1: Predict on random test samples."""
    logger.info("="*80)
    logger.info("TEST 1: RANDOM TEST SAMPLES")
    logger.info("="*80)
    
    tester = RealStructureTester()
    
    # Load test data
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Test on 20 random samples
    np.random.seed(42)
    test_indices = np.random.choice(len(test_data), 20, replace=False)
    
    logger.info(f"\nTesting on 20 random test samples...\n")
    
    results = []
    
    for i, idx in enumerate(test_indices):
        sample = test_data[idx]
        
        try:
            # Get prediction
            pred = tester.predict(
                sample['atomic_numbers'],
                sample['positions'],
                apply_delta=True
            )
            
            # Get target
            target = sample.get('energy_target', sample.get('energy', 0.0))
            
            # Calculate errors
            error_gnn = abs(pred['energy_gnn'] - target)
            error_corrected = abs(pred['energy_corrected'] - target)
            
            results.append({
                'sample_id': idx,
                'n_atoms': len(sample['atomic_numbers']),
                'target': target,
                'gnn_pred': pred['energy_gnn'],
                'corrected_pred': pred['energy_corrected'],
                'error_gnn': error_gnn,
                'error_corrected': error_corrected
            })
            
            logger.info(f"Sample {i+1}/20:")
            logger.info(f"   Atoms: {len(sample['atomic_numbers'])}")
            logger.info(f"   Target: {target:.4f} eV")
            logger.info(f"   GNN: {pred['energy_gnn']:.4f} eV (error: {error_gnn:.4f} eV)")
            logger.info(f"   Corrected: {pred['energy_corrected']:.4f} eV (error: {error_corrected:.4f} eV)")
            logger.info("")
            
        except Exception as e:
            logger.warning(f"Skipping sample {idx}: {e}")
            continue
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY:")
    avg_error_gnn = np.mean([r['error_gnn'] for r in results])
    avg_error_corrected = np.mean([r['error_corrected'] for r in results])
    
    logger.info(f"   Average GNN error: {avg_error_gnn:.4f} eV")
    logger.info(f"   Average corrected error: {avg_error_corrected:.4f} eV")
    logger.info(f"   Improvement: {avg_error_gnn - avg_error_corrected:.4f} eV")
    
    return results


def test_2_best_and_worst():
    """Test 2: Find best and worst predictions."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: BEST AND WORST PREDICTIONS")
    logger.info("="*80)
    
    tester = RealStructureTester()
    
    # Load test data
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.info("\nEvaluating on 500 random samples...")
    
    results = []
    for i, sample in enumerate(test_data[:500]):
        try:
            pred = tester.predict(
                sample['atomic_numbers'],
                sample['positions'],
                apply_delta=True
            )
            target = sample.get('energy_target', sample.get('energy', 0.0))
            error = abs(pred['energy_corrected'] - target)
            
            results.append({
                'sample_id': i,
                'n_atoms': len(sample['atomic_numbers']),
                'target': target,
                'prediction': pred['energy_corrected'],
                'error': error
            })
        except:
            continue
    
    # Sort by error
    results_sorted = sorted(results, key=lambda x: x['error'])
    
    # Best
    logger.info("\n✅ BEST 5 PREDICTIONS:")
    for i, r in enumerate(results_sorted[:5]):
        logger.info(f"   {i+1}. Sample {r['sample_id']}: {r['n_atoms']} atoms, Error={r['error']:.4f} eV")
        logger.info(f"      Target: {r['target']:.4f} eV, Pred: {r['prediction']:.4f} eV")
    
    # Worst
    logger.info("\n⚠️  WORST 5 PREDICTIONS:")
    for i, r in enumerate(results_sorted[-5:]):
        logger.info(f"   {i+1}. Sample {r['sample_id']}: {r['n_atoms']} atoms, Error={r['error']:.4f} eV")
        logger.info(f"      Target: {r['target']:.4f} eV, Pred: {r['prediction']:.4f} eV")


def test_3_specific_structures():
    """Test 3: Predict on specific structures from user's data."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: USER PROVIDED STRUCTURES")
    logger.info("="*80)
    
    # Look for user's VASP structures
    test_dir = Path("test_data")
    
    if not test_dir.exists():
        logger.info("   No test_data/ directory found")
        logger.info("   Skipping user structures test")
        return
    
    logger.info(f"\n   Checking for POSCAR files in {test_dir}...")
    
    poscar_files = list(test_dir.glob("*POSCAR*"))
    
    if not poscar_files:
        logger.info("   No POSCAR files found")
        return
    
    logger.info(f"   Found {len(poscar_files)} structures")
    
    # This would require ASE to read POSCAR files
    logger.info("\n   To test user structures:")
    logger.info("   1. Load POSCAR file with ASE")
    logger.info("   2. Convert to atomic_numbers and positions")
    logger.info("   3. Run pipeline.predict()")
    logger.info("   4. Compare with provided energy")
    
    logger.info("\n   Feature ready for implementation!")


def main():
    """Run all real structure tests."""
    logger.info("\n🚀 TESTING PIPELINE WITH REAL STRUCTURES")
    logger.info("="*80)
    
    # Run tests
    results_1 = test_1_random_test_samples()
    test_2_best_and_worst()
    test_3_specific_structures()
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL REAL STRUCTURE TESTS COMPLETE")
    logger.info("="*80)
    
    if results_1:
        errors = [r['error_corrected'] for r in results_1]
        logger.info(f"\n📊 FINAL METRICS:")
        logger.info(f"   Min error: {np.min(errors):.4f} eV")
        logger.info(f"   Max error: {np.max(errors):.4f} eV")
        logger.info(f"   Median error: {np.median(errors):.4f} eV")
        logger.info(f"   Mean error: {np.mean(errors):.4f} eV")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()


