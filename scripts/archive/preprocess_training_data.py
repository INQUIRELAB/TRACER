#!/usr/bin/env python3
"""
Preprocess and Clean Training Data
Applies data quality checks, outlier detection, and normalization.
"""

import sys
import os
from pathlib import Path
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_unified_dataset():
    """Load the unified training dataset."""
    logger.info("📥 Loading unified training dataset...")
    
    # For now, we'll load from existing data
    # In production, this would use UnifiedDatasetRegistry
    samples = []
    
    try:
        from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig, DatasetDomain
        
        config = UnifiedDatasetConfig(
            datasets=[
                DatasetConfig(
                    domain_id=DatasetDomain.JARVIS_DFT,
                    name="JARVIS-DFT",
                    path="data/jarvis_dft",
                    weight=1.0,
                    enabled=True
                ),
            ],
            mix_strategy="uniform",
            validation_split=0.1,
            test_split=0.1,
            random_seed=42,
            normalize_energies=False,  # We'll normalize in preprocessing
        )
        
        registry = UnifiedDatasetRegistry(config)
        samples = registry.get_all_samples(max_samples=10000)
        
    except Exception as e:
        logger.error(f"Failed to load unified dataset: {e}")
        logger.info("Will use demo samples for preprocessing demonstration")
        
        # Create demo samples for testing
        samples = [
            {
                'atomic_numbers': [1, 6, 8],
                'positions': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                'energy': -150.0,
                'forces': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                'num_atoms': 3,
            },
            {
                'atomic_numbers': [6] * 10,
                'positions': [[i, 0, 0] for i in range(10)],
                'energy': -300.0,
                'forces': [[0, 0, 0]] * 10,
                'num_atoms': 10,
            },
        ]
    
    logger.info(f"✅ Loaded {len(samples)} samples")
    return samples


def preprocess_data(output_dir: str = "data/preprocessed"):
    """Run preprocessing pipeline on training data."""
    logger.info("🚀 DATA PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    # 1. Load data
    data = load_unified_dataset()
    
    if len(data) == 0:
        logger.error("❌ No data to preprocess!")
        return
    
    # 2. Create preprocessor
    preprocessor = DataPreprocessor(
        energy_outlier_threshold=5.0,
        force_outlier_threshold=5.0,
        max_force_threshold=100.0,
        min_atoms=1,
        max_atoms=500,
        energy_range=(-10000, 10000),
    )
    
    # 3. Run preprocessing
    cleaned_data, results = preprocessor.process_dataset(
        data,
        normalize=True
    )
    
    # 4. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    with open(output_path / 'cleaned_data.json', 'w') as f:
        json.dump(cleaned_data[:100], f, indent=2)  # Save first 100 for demo
    
    # Save preprocessing results
    preprocessor.save_preprocessing_results(
        results,
        output_path / 'preprocessing_results.json'
    )
    
    logger.info(f"\n✅ Preprocessing complete!")
    logger.info(f"   Output directory: {output_path}")
    logger.info(f"   Cleaned samples: {len(cleaned_data)}")
    logger.info("="*80)
    
    return cleaned_data, results


if __name__ == "__main__":
    import typer
    app = typer.Typer()
    
    @app.command()
    def preprocess(
        output_dir: str = typer.Option("data/preprocessed", help="Output directory")
    ):
        preprocess_data(output_dir=output_dir)
    
    app()



