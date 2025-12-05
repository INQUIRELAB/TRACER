#!/usr/bin/env python3
"""
Final Comprehensive Validation of the Hybrid GNN Pipeline
Tests accuracy, generalization, and publication-readiness metrics.
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model_gemnet import GemNetWrapper
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FinalValidator:
    """Comprehensive validation of the pipeline."""
    
    def __init__(self):
        """Initialize validator."""
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
        """Load normalization statistics."""
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
    
    def predict(self, atomic_numbers, positions):
        """Make prediction with denormalization."""
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float32)
        data = Data(
            atomic_numbers=atomic_numbers,
            pos=positions,
            batch=torch.zeros(len(atomic_numbers), dtype=torch.long)
        )
        
        with torch.no_grad():
            output = self.gemnet(data)
            energy_normalized = output[0].item() if isinstance(output, tuple) else output.item()
        
        # Denormalize
        energy_raw = energy_normalized * self.training_std + self.training_mean
        
        return energy_raw


def test_1_accuracy_metrics():
    """Test 1: Compute standard accuracy metrics."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: ACCURACY METRICS")
    logger.info("="*80)
    
    validator = FinalValidator()
    
    # Load test data
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Evaluating on {len(test_data)} test samples...")
    
    predictions = []
    targets = []
    
    for sample in test_data[:1000]:  # Evaluate on 1000 samples
        try:
            pred = validator.predict(sample['atomic_numbers'], sample['positions'])
            target = sample.get('energy_target', sample.get('energy', 0.0))
            
            predictions.append(pred)
            targets.append(target)
        except:
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    mape = np.mean(np.abs((predictions - targets) / targets)) * 100
    r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - np.mean(targets))**2)
    
    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   MAE:  {mae:.4f} eV")
    logger.info(f"   RMSE: {rmse:.4f} eV")
    logger.info(f"   MAPE: {mape:.2f}%")
    logger.info(f"   R²:   {r2:.4f}")
    
    # Per-atom metrics
    per_atom_predictions = predictions / np.array([len(s['atomic_numbers']) for s in test_data[:1000] if 'atomic_numbers' in s][:len(predictions)])
    per_atom_targets = targets / np.array([len(s['atomic_numbers']) for s in test_data[:1000] if 'atomic_numbers' in s][:len(targets)])
    
    mae_per_atom = np.mean(np.abs(per_atom_predictions - per_atom_targets))
    logger.info(f"\n   MAE (per atom): {mae_per_atom:.4f} eV/atom")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2, 'mae_per_atom': mae_per_atom}


def test_2_size_distribution():
    """Test 2: Performance by structure size."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: SIZE-DEPENDENT PERFORMANCE")
    logger.info("="*80)
    
    validator = FinalValidator()
    
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    size_groups = {'small (1-10)': [], 'medium (11-30)': [], 'large (31-96)': []}
    
    for sample in test_data[:1000]:
        n_atoms = len(sample['atomic_numbers'])
        pred = validator.predict(sample['atomic_numbers'], sample['positions'])
        target = sample.get('energy_target', sample.get('energy', 0.0))
        error = abs(pred - target)
        
        if n_atoms <= 10:
            size_groups['small (1-10)'].append(error)
        elif n_atoms <= 30:
            size_groups['medium (11-30)'].append(error)
        else:
            size_groups['large (31-96)'].append(error)
    
    logger.info(f"\n📊 RESULTS BY STRUCTURE SIZE:")
    for size_range, errors in size_groups.items():
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            logger.info(f"   {size_range}: {mean_error:.4f} ± {std_error:.4f} eV ({len(errors)} samples)")


def test_3_energy_range():
    """Test 3: Performance across energy ranges."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: ENERGY-RANGE PERFORMANCE")
    logger.info("="*80)
    
    validator = FinalValidator()
    
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    predictions = []
    targets = []
    
    for sample in test_data[:1000]:
        try:
            pred = validator.predict(sample['atomic_numbers'], sample['positions'])
            target = sample.get('energy_target', sample.get('energy', 0.0))
            predictions.append(pred)
            targets.append(target)
        except:
            continue
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Energy ranges
    ranges = {
        'very negative (<-50)': [],
        'negative (-50 to 0)': [],
        'positive (>0)': []
    }
    
    for pred, target in zip(predictions, targets):
        error = abs(pred - target)
        if target < -50:
            ranges['very negative (<-50)'].append(error)
        elif target < 0:
            ranges['negative (-50 to 0)'].append(error)
        else:
            ranges['positive (>0)'].append(error)
    
    logger.info(f"\n📊 RESULTS BY ENERGY RANGE:")
    for energy_range, errors in ranges.items():
        if errors:
            mean_error = np.mean(errors)
            logger.info(f"   {energy_range}: {mean_error:.4f} eV ({len(errors)} samples)")


def test_4_convergence_check():
    """Test 4: Check if model converged properly."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: CONVERGENCE CHECK")
    logger.info("="*80)
    
    checkpoint = torch.load("models/gemnet_full/best_model.pt", map_location='cpu')
    
    logger.info(f"\n📊 TRAINING METRICS:")
    logger.info(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    logger.info(f"   Final epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"   Final train loss: {checkpoint.get('train_loss', 'N/A')}")
    
    # Check if loss is reasonable
    val_loss = checkpoint.get('best_val_loss', 999)
    if val_loss < 0.1:
        logger.info(f"   ✓ Validation loss is excellent (< 0.1)")
    elif val_loss < 1.0:
        logger.info(f"   ✓ Validation loss is good (< 1.0)")
    else:
        logger.info(f"   ⚠ Validation loss is high (> 1.0)")


def test_5_publication_metrics():
    """Test 5: Publication-readiness metrics."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: PUBLICATION-READINESS ASSESSMENT")
    logger.info("="*80)
    
    validator = FinalValidator()
    
    with open('data/preprocessed_full_unified/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    results = []
    for sample in test_data[:500]:
        try:
            pred = validator.predict(sample['atomic_numbers'], sample['positions'])
            target = sample.get('energy_target', sample.get('energy', 0.0))
            error = abs(pred - target)
            rel_error = abs(error / target) * 100 if target != 0 else 0
            results.append({
                'error': error,
                'rel_error': rel_error,
                'n_atoms': len(sample['atomic_numbers'])
            })
        except:
            continue
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"\n📊 PUBLICATION METRICS:")
    logger.info(f"   Median MAE: {results_df['error'].median():.4f} eV")
    logger.info(f"   95th percentile MAE: {results_df['error'].quantile(0.95):.4f} eV")
    logger.info(f"   Median relative error: {results_df['rel_error'].median():.2f}%")
    logger.info(f"   Samples within 5% error: {(results_df['rel_error'] < 5).sum() / len(results_df) * 100:.1f}%")
    logger.info(f"   Samples within 10% error: {(results_df['rel_error'] < 10).sum() / len(results_df) * 100:.1f}%")


def main():
    """Run all validation tests."""
    logger.info("🚀 STARTING COMPREHENSIVE PIPELINE VALIDATION")
    logger.info("="*80)
    
    # Run all tests
    results_1 = test_1_accuracy_metrics()
    test_2_size_distribution()
    test_3_energy_range()
    test_4_convergence_check()
    test_5_publication_metrics()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info("\n📊 KEY METRICS:")
    logger.info(f"   MAE: {results_1['mae']:.4f} eV")
    logger.info(f"   RMSE: {results_1['rmse']:.4f} eV")
    logger.info(f"   R²: {results_1['r2']:.4f}")
    logger.info(f"   MAE per atom: {results_1['mae_per_atom']:.4f} eV/atom")
    
    logger.info("\n🎯 PUBLICATION STATUS:")
    if results_1['mae'] < 1.0 and results_1['r2'] > 0.9:
        logger.info("   ✓ EXCELLENT: Ready for publication!")
    elif results_1['mae'] < 5.0 and results_1['r2'] > 0.7:
        logger.info("   ✓ GOOD: Suitable for publication")
    else:
        logger.info("   ⚠ NEEDS IMPROVEMENT")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()



