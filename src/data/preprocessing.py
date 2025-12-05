"""
Data Preprocessing and Cleaning Pipeline
Handles data quality checks, normalization, and outlier detection.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DataQualityStats:
    """Statistics for data quality assessment."""
    total_samples: int
    valid_samples: int
    outlier_samples: int
    energy_range: Tuple[float, float]
    energy_mean: float
    energy_std: float
    force_range: Tuple[float, float]
    max_atoms: int
    min_atoms: int
    atomic_species: List[int]


class DataPreprocessor:
    """Preprocesses and cleans molecular data for training."""
    
    def __init__(
        self,
        energy_outlier_threshold: float = 5.0,  # Standard deviations
        force_outlier_threshold: float = 5.0,
        max_force_threshold: float = 100.0,  # eV/Å
        min_atoms: int = 1,
        max_atoms: int = 500,
        energy_range: Tuple[float, float] = (-10000, 10000),  # eV
    ):
        """Initialize preprocessor with validation thresholds."""
        self.energy_outlier_threshold = energy_outlier_threshold
        self.force_outlier_threshold = force_outlier_threshold
        self.max_force_threshold = max_force_threshold
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.energy_range = energy_range
        
        self.energy_mean = None
        self.energy_std = None
        self.force_mean = None
        self.force_std = None
        
    def compute_statistics(self, data: List[Dict]) -> DataQualityStats:
        """Compute statistics on dataset."""
        logger.info("📊 Computing dataset statistics...")
        
        energies = []
        forces = []
        num_atoms = []
        atomic_species = set()
        
        for sample in data:
            if 'energy' in sample or 'energy_target' in sample:
                energy = sample.get('energy', sample.get('energy_target', 0))
                if isinstance(energy, torch.Tensor):
                    energy = energy.item()
                # Convert to float, skip if not numeric
                try:
                    energy = float(energy)
                except (ValueError, TypeError):
                    continue
                energies.append(energy)
            
            if 'forces' in sample or 'forces_target' in sample:
                force_data = sample.get('forces', sample.get('forces_target', []))
                if isinstance(force_data, list):
                    for force in force_data:
                        if isinstance(force, list) and len(force) == 3:
                            force_mag = np.linalg.norm(force)
                            forces.append(force_mag)
            
            num_atoms.append(sample.get('num_atoms', len(sample.get('atomic_numbers', []))))
            
            for z in sample.get('atomic_numbers', []):
                atomic_species.add(z)
        
        if len(energies) > 0:
            self.energy_mean = np.mean(energies)
            self.energy_std = np.std(energies)
        
        if len(forces) > 0:
            self.force_mean = np.mean(forces)
            self.force_std = np.std(forces)
        
        stats = DataQualityStats(
            total_samples=len(data),
            valid_samples=len(data),  # Will be updated during validation
            outlier_samples=0,
            energy_range=(min(energies) if energies else 0, max(energies) if energies else 0),
            energy_mean=self.energy_mean or 0.0,
            energy_std=self.energy_std or 1.0,
            force_range=(min(forces) if forces else 0, max(forces) if forces else 0),
            max_atoms=max(num_atoms) if num_atoms else 0,
            min_atoms=min(num_atoms) if num_atoms else 0,
            atomic_species=sorted(atomic_species)
        )
        
        logger.info(f"   Total samples: {stats.total_samples}")
        logger.info(f"   Energy range: {stats.energy_range[0]:.2f} to {stats.energy_range[1]:.2f} eV")
        logger.info(f"   Mean energy: {stats.energy_mean:.2f} eV, Std: {stats.energy_std:.2f} eV")
        logger.info(f"   Force range: {stats.force_range[0]:.2f} to {stats.force_range[1]:.2f} eV/Å")
        logger.info(f"   Atoms range: {stats.min_atoms} to {stats.max_atoms}")
        logger.info(f"   Atomic species: {len(stats.atomic_species)} types")
        
        return stats
    
    def validate_sample(self, sample: Dict) -> Tuple[bool, str]:
        """Validate a single sample."""
        issues = []
        
        # Check atomic numbers and positions
        atomic_numbers = sample.get('atomic_numbers', [])
        positions = sample.get('positions', [])
        num_atoms = len(atomic_numbers)
        
        if num_atoms == 0:
            return False, "No atoms"
        
        if num_atoms < self.min_atoms or num_atoms > self.max_atoms:
            return False, f"Invalid number of atoms: {num_atoms}"
        
        if len(positions) != num_atoms:
            return False, f"Position count mismatch: {len(positions)} != {num_atoms}"
        
        # Check energy
        energy = sample.get('energy', sample.get('energy_target', None))
        if energy is not None:
            if isinstance(energy, torch.Tensor):
                energy = energy.item()
            
            # Convert to float, skip if not numeric
            try:
                energy = float(energy)
            except (ValueError, TypeError):
                return False, "Energy not numeric"
            
            # Check energy range
            if energy < self.energy_range[0] or energy > self.energy_range[1]:
                return False, f"Energy out of range: {energy:.2f} eV"
            
            # Check for outliers (if we have statistics)
            if self.energy_mean is not None and self.energy_std is not None:
                if self.energy_std > 0:
                    z_score = abs((energy - self.energy_mean) / self.energy_std)
                    if z_score > self.energy_outlier_threshold:
                        return False, f"Energy outlier: z={z_score:.2f}"
        
        # Check forces
        forces = sample.get('forces', sample.get('forces_target', []))
        if forces and len(forces) == num_atoms:
            for force in forces:
                if isinstance(force, list) and len(force) == 3:
                    force_mag = np.linalg.norm(force)
                    if force_mag > self.max_force_threshold:
                        return False, f"Force too large: {force_mag:.2f} eV/Å"
        
        return True, "Valid"
    
    def clean_dataset(self, data: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
        """Clean dataset by removing invalid samples."""
        logger.info("🧹 Cleaning dataset...")
        
        cleaned_data = []
        removed = {
            'invalid': 0,
            'outliers': 0,
            'size_mismatch': 0,
            'energy_outliers': 0,
            'force_outliers': 0,
        }
        
        for sample in data:
            is_valid, reason = self.validate_sample(sample)
            
            if is_valid:
                cleaned_data.append(sample)
            else:
                removed['invalid'] += 1
                if 'outlier' in reason.lower():
                    removed['outliers'] += 1
                if 'size' in reason.lower() or 'atom' in reason.lower():
                    removed['size_mismatch'] += 1
                if 'energy' in reason.lower():
                    removed['energy_outliers'] += 1
                if 'force' in reason.lower():
                    removed['force_outliers'] += 1
        
        logger.info(f"   Removed {removed['invalid']} invalid samples")
        logger.info(f"   Kept {len(cleaned_data)} valid samples")
        logger.info(f"   Removal breakdown: {removed}")
        
        return cleaned_data, removed
    
    def normalize_energies(self, data: List[Dict]) -> Tuple[List[Dict], Dict[str, float]]:
        """Normalize energies to standard scale."""
        logger.info("📏 Normalizing energies...")
        
        # Compute mean and std
        energies = []
        for sample in data:
            energy = sample.get('energy', sample.get('energy_target', 0))
            if isinstance(energy, torch.Tensor):
                energy = energy.item()
            energies.append(energy)
        
        mean = np.mean(energies)
        std = np.std(energies) if np.std(energies) > 0 else 1.0
        
        # Normalize
        for sample in data:
            energy_key = 'energy' if 'energy' in sample else 'energy_target'
            energy = sample[energy_key]
            if isinstance(energy, torch.Tensor):
                energy = energy.item()
            normalized_energy = (energy - mean) / std
            sample[energy_key] = normalized_energy
        
        norm_stats = {'mean': mean, 'std': std}
        
        logger.info(f"   Energy mean: {mean:.4f} eV")
        logger.info(f"   Energy std: {std:.4f} eV")
        logger.info("   Energies normalized to mean=0, std=1")
        
        return data, norm_stats
    
    def process_dataset(self, data: List[Dict], normalize: bool = True) -> Tuple[List[Dict], Dict]:
        """Complete preprocessing pipeline."""
        logger.info("🚀 STARTING DATA PREPROCESSING")
        logger.info("="*80)
        
        # Step 1: Compute statistics
        stats = self.compute_statistics(data)
        
        # Step 2: Clean dataset
        cleaned_data, removal_stats = self.clean_dataset(data)
        
        # Step 3: Normalize if requested
        if normalize and len(cleaned_data) > 0:
            cleaned_data, norm_stats = self.normalize_energies(cleaned_data)
        else:
            norm_stats = {}
        
        # Step 4: Final statistics
        final_stats = self.compute_statistics(cleaned_data)
        
        results = {
            'original_samples': stats.total_samples,
            'cleaned_samples': len(cleaned_data),
            'removed_samples': removal_stats,
            'statistics': stats,
            'final_statistics': final_stats,
            'normalization': norm_stats,
        }
        
        logger.info("\n✅ DATA PREPROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"   Original: {stats.total_samples} samples")
        logger.info(f"   Cleaned: {len(cleaned_data)} samples")
        logger.info(f"   Removed: {stats.total_samples - len(cleaned_data)} samples")
        logger.info(f"   Retention rate: {len(cleaned_data)/stats.total_samples*100:.1f}%")
        
        return cleaned_data, results
    
    def save_preprocessing_results(self, results: Dict, output_path: Path):
        """Save preprocessing results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass objects to dictionaries for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✅ Saved preprocessing results to {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy types and dataclasses to native Python types."""
        from dataclasses import is_dataclass, asdict
        
        if is_dataclass(obj):
            return {k: self._make_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj

