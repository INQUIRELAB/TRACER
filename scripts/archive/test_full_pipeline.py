#!/usr/bin/env python3
"""
Test the Full Updated Pipeline with GemNet
Demonstrates the complete hybrid GNN + Quantum pipeline.
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FullPipeline:
    """Complete pipeline with GemNet + Delta Head."""
    
    def __init__(self):
        """Initialize the full pipeline."""
        logger.info("🚀 INITIALIZING FULL PIPELINE")
        logger.info("="*80)
        
        # Load GemNet model
        logger.info("📥 Loading GemNet model...")
        self.gemnet = self._load_gemnet_model()
        logger.info("   ✓ GemNet model loaded")
        
        # Load Delta Head
        logger.info("📥 Loading Delta Head...")
        self.delta_head = self._load_delta_head()
        logger.info("   ✓ Delta Head loaded")
        
        # Load denormalization stats
        logger.info("📥 Loading denormalization stats...")
        self.training_mean, self.training_std = self._load_norm_stats()
        logger.info(f"   Mean: {self.training_mean:.4f}, Std: {self.training_std:.4f}")
        
        logger.info("\n✅ Pipeline initialized successfully!")
    
    def _load_gemnet_model(self):
        """Load trained GemNet model."""
        model_path = Path("models/gemnet_full/best_model.pt")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint.get('model_config', {
            'num_atoms': 95,
            'hidden_dim': 256,
            'num_interactions': 6,
            'cutoff': 10.0
        })
        
        model = GemNetWrapper(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _load_delta_head(self):
        """Load delta head model."""
        delta_path = Path("artifacts/delta_head_gemnet.pt")
        
        if not delta_path.exists():
            logger.warning("Delta head not found, using default")
            return {'mean_delta': 0.0, 'std_delta': 1.0}
        
        return torch.load(delta_path, map_location='cpu')
    
    def _load_norm_stats(self):
        """Load normalization statistics."""
        # From preprocessing results
        prep_path = Path("data/preprocessed_full_unified/preprocessing_results.json")
        
        with open(prep_path, 'r') as f:
            prep_results = json.load(f)
        
        norm_stats = prep_results.get('normalization', {})
        mean = norm_stats.get('mean', 0.0)
        std = norm_stats.get('std', 1.0)
        
        return mean, std
    
    def predict(self, atomic_numbers, positions):
        """Run full pipeline prediction.
        
        Args:
            atomic_numbers: List of atomic numbers
            positions: List of (x, y, z) positions
            
        Returns:
            Dictionary with energy_pred, energy_corrected, delta
        """
        # Convert to torch tensors
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            atomic_numbers=atomic_numbers,
            pos=positions,
            batch=torch.zeros(len(atomic_numbers), dtype=torch.long)
        )
        
        # GNN Prediction
        with torch.no_grad():
            output = self.gemnet(data)
            
            if isinstance(output, tuple):
                energy_pred = output[0].item()  # Extract energy from tuple
            else:
                energy_pred = output.item()
        
        # Denormalize energy prediction
        # Model outputs normalized energies (mean=0, std=1)
        # Need to convert back to raw energy scale
        norm_mean = 0.0  # Model was trained with mean=0
        norm_std = 1.0   # Model was trained with std=1
        
        # Denormalize: raw = (normalized * std) + mean
        energy_pred_denorm = energy_pred * self.training_std + self.training_mean
        
        # Delta Head Correction (also denormalized)
        delta = self.delta_head.get('mean_delta', 0.0)
        delta_denorm = delta * self.training_std  # Delta was in normalized space
        energy_corrected = energy_pred_denorm + delta_denorm
        
        return {
            'energy_gnn_normalized': energy_pred,
            'energy_gnn_denormalized': energy_pred_denorm,
            'delta_correction': delta_denorm,
            'energy_corrected': energy_corrected,
            'num_atoms': len(atomic_numbers)
        }


def test_with_samples():
    """Test the pipeline with sample structures."""
    logger.info("\n🔬 TESTING PIPELINE WITH SAMPLE STRUCTURES")
    logger.info("="*80)
    
    # Initialize pipeline
    pipeline = FullPipeline()
    
    # Load test data
    test_path = Path("data/preprocessed_full_unified/test_data.json")
    with open(test_path, 'r') as f:
        test_samples = json.load(f)
    
    logger.info(f"\n   Loaded {len(test_samples)} test samples")
    logger.info("   Testing on first 10 samples...\n")
    
    results = []
    
    for i, sample in enumerate(test_samples[:10]):
        logger.info(f"\n📊 Sample {i+1}/10:")
        logger.info(f"   Formula: {len(sample['atomic_numbers'])} atoms")
        logger.info(f"   Domain: {sample.get('domain', 'unknown')}")
        
        # Get prediction
        pred = pipeline.predict(
            sample['atomic_numbers'],
            sample['positions']
        )
        
        # Get target
        energy_target = sample.get('energy_target', sample.get('energy', 0.0))
        
        # Calculate errors
        error_gnn = abs(pred['energy_gnn_denormalized'] - energy_target)
        error_corrected = abs(pred['energy_corrected'] - energy_target)
        improvement = error_gnn - error_corrected
        
        logger.info(f"   GNN (denorm): {pred['energy_gnn_denormalized']:.4f} eV")
        logger.info(f"   Target: {energy_target:.4f} eV")
        logger.info(f"   Corrected: {pred['energy_corrected']:.4f} eV")
        logger.info(f"   Error (GNN): {error_gnn:.4f} eV")
        logger.info(f"   Error (Corrected): {error_corrected:.4f} eV")
        
        if improvement > 0:
            logger.info(f"   ✓ Improvement: {improvement:.4f} eV")
        else:
            logger.info(f"   - Degradation: {abs(improvement):.4f} eV")
        
        results.append({
            'sample_id': i,
            'gnn_error': error_gnn,
            'corrected_error': error_corrected,
            'improvement': improvement
        })
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("📊 PIPELINE TEST SUMMARY")
    logger.info("="*80)
    
    avg_error_gnn = np.mean([r['gnn_error'] for r in results])
    avg_error_corrected = np.mean([r['corrected_error'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    
    logger.info(f"   Average GNN error: {avg_error_gnn:.4f} eV")
    logger.info(f"   Average corrected error: {avg_error_corrected:.4f} eV")
    logger.info(f"   Average improvement: {avg_improvement:.4f} eV")
    logger.info(f"   Improvement rate: {avg_improvement/avg_error_gnn*100:.1f}%")
    
    logger.info("\n✅ Pipeline test complete!")
    logger.info("="*80)


def main():
    """Main test execution."""
    test_with_samples()


if __name__ == "__main__":
    main()

