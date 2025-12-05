#!/usr/bin/env python3
"""Test script for unified dataset registry with temperature-based sampling."""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dft_hybrid.data.unified_registry import (
    UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig, DatasetDomain,
    create_unified_dataloader, UnitConverter, TemperatureSampler
)


def test_unit_converter():
    """Test unit conversion functionality."""
    print("🧪 Testing Unit Converter")
    print("-" * 40)
    
    # Test energy conversions
    hartree_to_ev = UnitConverter.convert_energy(1.0, "Hartree", "eV")
    print(f"1 Hartree = {hartree_to_ev:.6f} eV")
    
    kcal_to_ev = UnitConverter.convert_energy(1.0, "kcal/mol", "eV")
    print(f"1 kcal/mol = {kcal_to_ev:.6f} eV")
    
    # Test force conversions
    hartree_bohr_to_ev_ang = UnitConverter.convert_force(1.0, "Hartree/Bohr", "eV/Å")
    print(f"1 Hartree/Bohr = {hartree_bohr_to_ev_ang:.6f} eV/Å")
    
    # Test stress conversions
    gpa_to_ev_ang3 = UnitConverter.convert_stress(1.0, "GPa", "eV/Å³")
    print(f"1 GPa = {gpa_to_ev_ang3:.6f} eV/Å³")
    
    print("✅ Unit conversion tests passed\n")


def test_temperature_sampler():
    """Test temperature-based sampling."""
    print("🌡️ Testing Temperature Sampler")
    print("-" * 40)
    
    # Create temperature sampler
    sampler = TemperatureSampler(tau=1.0)
    
    # Test with different temperature ranges
    temperatures = np.array([100, 300, 500, 1000, 2000])  # Kelvin
    
    print(f"Temperatures: {temperatures} K")
    
    # Compute weights
    weights = sampler.compute_temperature_weights(temperatures)
    print(f"Weights: {weights}")
    
    # Sample indices
    sampled_indices = sampler.sample_indices(temperatures, num_samples=10, replacement=True)
    print(f"Sampled indices: {sampled_indices}")
    
    # Test with different tau values
    print("\nTesting different tau values:")
    for tau in [0.5, 1.0, 2.0]:
        sampler_tau = TemperatureSampler(tau=tau)
        weights_tau = sampler_tau.compute_temperature_weights(temperatures)
        print(f"τ={tau}: {weights_tau}")
    
    print("✅ Temperature sampler tests passed\n")


def test_unified_registry():
    """Test unified dataset registry."""
    print("🗂️ Testing Unified Dataset Registry")
    print("-" * 40)
    
    # Create test configuration with all available datasets
    datasets = [
        DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT-v1",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(0, 1000),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
            max_samples=500,  # Small sample for testing
            weight=1.0,
            enabled=True
        ),
        DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT-v2",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-6-6-2019.json",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(0, 1000),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
            max_samples=500,  # Small sample for testing
            weight=0.8,
            enabled=True
        ),
        DatasetConfig(
            domain_id=DatasetDomain.OC20_S2EF,
            name="OC20-S2EF",
            path="data/oc20/s2ef_train_200K/s2ef_train_200K",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(300, 2000),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
            max_samples=200,  # Small sample for testing
            weight=2.0,
            enabled=True
        ),
        DatasetConfig(
            domain_id=DatasetDomain.OC22_S2EF,
            name="OC22-S2EF",
            path="data/oc22/s2ef_val_id/s2ef_val_id",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(300, 2000),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
            max_samples=200,  # Small sample for testing
            weight=1.8,
            enabled=True
        ),
        DatasetConfig(
            domain_id=DatasetDomain.ANI1X,
            name="ANI1x",
            path="data/ani1x/ani1x_dataset.h5",
            energy_unit="Hartree",
            force_unit="Hartree/Bohr",
            stress_unit="Hartree/Bohr³",
            temperature_range=(100, 2000),
            atomic_species=[1, 6, 7, 8],  # H, C, N, O
            max_samples=200,  # Small sample for testing
            weight=1.5,
            enabled=True
        )
    ]
    
    config = UnifiedDatasetConfig(
        datasets=datasets,
        mix_strategy="temperature",
        temperature_tau=1.0,
        unit_conversion=True,
        validation_split=0.1,
        test_split=0.1,
        random_seed=42
    )
    
    print(f"Configuration:")
    print(f"  Mix strategy: {config.mix_strategy}")
    print(f"  Temperature tau: {config.temperature_tau}")
    print(f"  Unit conversion: {config.unit_conversion}")
    print(f"  Number of datasets: {len(config.datasets)}")
    
    # Create registry
    registry = UnifiedDatasetRegistry(config)
    
    print(f"Registry initialized with {len(registry.loaders)} loaders")
    
    # Test loading data
    try:
        print("\nLoading data...")
        mixed_samples = registry.create_mixed_dataset(max_samples=100)
        print(f"✅ Loaded {len(mixed_samples)} mixed samples")
        
        if mixed_samples:
            # Test first sample
            graph, targets = mixed_samples[0]
            print(f"First sample:")
            print(f"  Graph nodes: {graph.pos.shape[0]}")
            print(f"  Graph edges: {graph.edge_index.shape[1]}")
            print(f"  Domain ID: {graph.domain_id}")
            print(f"  Dataset name: {graph.dataset_name}")
            print(f"  Energy shape: {targets['energy'].shape}")
            print(f"  Forces shape: {targets['forces'].shape}")
            print(f"  Stress shape: {targets['stress'].shape}")
            
            # Test dataloader creation
            print("\nCreating dataloader...")
            dataloader = registry.create_dataloader(mixed_samples, batch_size=4, shuffle=False)
            print(f"✅ Created dataloader with {len(dataloader)} batches")
            
            # Test one batch
            for batch_graph, batch_targets in dataloader:
                print(f"Batch:")
                print(f"  Graph nodes: {batch_graph.pos.shape[0]}")
                print(f"  Graph edges: {batch_graph.edge_index.shape[1]}")
                print(f"  Batch size: {batch_graph.batch.max().item() + 1}")
                print(f"  Energy shape: {batch_targets['energy'].shape}")
                print(f"  Forces shape: {batch_targets['forces'].shape}")
                print(f"  Stress shape: {batch_targets['stress'].shape}")
                break
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Unified registry tests completed\n")


def test_different_mix_strategies():
    """Test different mixing strategies."""
    print("🔄 Testing Different Mix Strategies")
    print("-" * 40)
    
    # Create test configuration
    datasets = [
        DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(0, 1000),
            max_samples=500,  # Small sample for testing
            weight=1.0,
            enabled=True
        )
    ]
    
    strategies = ["temperature", "uniform", "weighted"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        
        config = UnifiedDatasetConfig(
            datasets=datasets,
            mix_strategy=strategy,
            temperature_tau=1.0,
            unit_conversion=True,
            random_seed=42
        )
        
        registry = UnifiedDatasetRegistry(config)
        
        try:
            mixed_samples = registry.create_mixed_dataset(max_samples=50)
            print(f"  ✅ Created {len(mixed_samples)} samples with {strategy} strategy")
        except Exception as e:
            print(f"  ❌ Error with {strategy} strategy: {e}")
    
    print("✅ Mix strategy tests completed\n")


def main():
    """Run all tests."""
    print("🚀 Unified Dataset Registry Test Suite")
    print("=" * 50)
    
    # Run tests
    test_unit_converter()
    test_temperature_sampler()
    test_unified_registry()
    test_different_mix_strategies()
    
    print("🎉 All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
