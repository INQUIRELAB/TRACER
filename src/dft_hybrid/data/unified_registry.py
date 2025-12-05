"""
Unified Dataset Registry for Multi-Domain Molecular Property Prediction.

This module provides a unified interface for loading and managing multiple
molecular datasets (JARVIS-DFT, JARVIS-Elastic, OC20-S2EF, OC22-S2EF, ANI1x).

Data Sources:
- JARVIS-DFT: Choudhary & DeCost (2021), npj Computational Materials
- JARVIS-Elastic: Choudhary et al. (2018), Physical Review Materials  
- OC20-S2EF: Chanussot et al. (2021), ACS Catalysis
- OC22-S2EF: Tran et al. (2023), ACS Catalysis
- ANI1x: Smith et al. (2017), Chemical Science

All datasets are real, peer-reviewed, and publicly available.
For complete data documentation, see src/data_documentation.py
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dft_hybrid.data.jarvis_dft import JARVISDFTDataset, create_jarvis_dataloader


class DatasetDomain(Enum):
    """Dataset domain identifiers."""
    ANI1X = "ani1x"
    OC20_S2EF = "oc20_s2ef"
    OC20_IS2RE = "oc20_is2re"
    OC22_S2EF = "oc22_s2ef"
    OC22_IS2RE = "oc22_is2re"
    JARVIS_DFT = "jarvis_dft"
    JARVIS_ELASTIC = "jarvis_elastic"
    JARVIS_ELECTRONIC = "jarvis_electronic"


@dataclass
class DatasetConfig:
    """Configuration for a dataset in the registry."""
    domain_id: DatasetDomain
    name: str
    path: str
    energy_unit: str = "eV"  # eV, Hartree, kcal/mol
    force_unit: str = "eV/Å"  # eV/Å, Hartree/Bohr, kcal/mol/Å
    stress_unit: str = "eV/Å³"  # eV/Å³, GPa, kbar
    temperature_range: Optional[Tuple[float, float]] = None  # (min_K, max_K)
    atomic_species: List[int] = field(default_factory=list)  # Atomic numbers
    max_samples: Optional[int] = None
    weight: float = 1.0  # Sampling weight
    enabled: bool = True


@dataclass
class UnifiedDatasetConfig:
    """Configuration for the unified dataset registry."""
    datasets: List[DatasetConfig] = field(default_factory=list)
    mix_strategy: str = "temperature"  # temperature, uniform, weighted
    temperature_tau: float = 1.0  # Temperature mixing parameter
    unit_conversion: bool = True  # Convert all to eV units
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    normalize_energies: bool = True  # Add energy normalization
    energy_mean: Optional[float] = None  # Computed mean for normalization
    energy_std: Optional[float] = None  # Computed std for normalization


class UnitConverter:
    """Handles unit conversions between different dataset formats."""
    
    # Conversion factors to eV
    ENERGY_CONVERSIONS = {
        "eV": 1.0,
        "Hartree": 27.211386245988,  # eV per Hartree
        "kcal/mol": 0.0433641153087705,  # eV per kcal/mol
        "kJ/mol": 0.0103642688,  # eV per kJ/mol
    }
    
    FORCE_CONVERSIONS = {
        "eV/Å": 1.0,
        "Hartree/Bohr": 51.42208619083232,  # eV/Å per Hartree/Bohr
        "kcal/mol/Å": 0.0433641153087705,  # eV/Å per kcal/mol/Å
        "kJ/mol/Å": 0.0103642688,  # eV/Å per kJ/mol/Å
    }
    
    STRESS_CONVERSIONS = {
        "eV/Å³": 1.0,
        "GPa": 0.00016021766208,  # eV/Å³ per GPa
        "kbar": 0.0016021766208,  # eV/Å³ per kbar
        "Hartree/Bohr³": 0.000147765,  # eV/Å³ per Hartree/Bohr³
    }
    
    @classmethod
    def convert_energy(cls, value: float, from_unit: str, to_unit: str = "eV") -> float:
        """Convert energy from one unit to another."""
        if from_unit == to_unit:
            return value
        
        from_factor = cls.ENERGY_CONVERSIONS.get(from_unit, 1.0)
        to_factor = cls.ENERGY_CONVERSIONS.get(to_unit, 1.0)
        
        return value * from_factor / to_factor
    
    @classmethod
    def convert_force(cls, value: float, from_unit: str, to_unit: str = "eV/Å") -> float:
        """Convert force from one unit to another."""
        if from_unit == to_unit:
            return value
        
        from_factor = cls.FORCE_CONVERSIONS.get(from_unit, 1.0)
        to_factor = cls.FORCE_CONVERSIONS.get(to_unit, 1.0)
        
        return value * from_factor / to_factor
    
    @classmethod
    def convert_stress(cls, value: float, from_unit: str, to_unit: str = "eV/Å³") -> float:
        """Convert stress from one unit to another."""
        if from_unit == to_unit:
            return value
        
        from_factor = cls.STRESS_CONVERSIONS.get(from_unit, 1.0)
        to_factor = cls.STRESS_CONVERSIONS.get(to_unit, 1.0)
        
        return value * from_factor / to_factor


class TemperatureSampler:
    """Temperature-based sampling for dataset mixing."""
    
    def __init__(self, tau: float = 1.0):
        """Initialize temperature sampler.
        
        Args:
            tau: Temperature parameter for Boltzmann-like sampling
        """
        self.tau = tau
    
    def compute_temperature_weights(self, temperatures: np.ndarray) -> np.ndarray:
        """Compute sampling weights based on temperature.
        
        Args:
            temperatures: Array of temperatures in Kelvin
            
        Returns:
            Array of sampling weights
        """
        if len(temperatures) == 0:
            return np.array([])
        
        # Boltzmann-like weighting: exp(-T/tau)
        weights = np.exp(-temperatures / self.tau)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def sample_indices(self, temperatures: np.ndarray, num_samples: int, 
                      replacement: bool = True) -> np.ndarray:
        """Sample indices based on temperature weights.
        
        Args:
            temperatures: Array of temperatures
            num_samples: Number of samples to draw
            replacement: Whether to sample with replacement
            
        Returns:
            Array of sampled indices
        """
        weights = self.compute_temperature_weights(temperatures)
        
        if replacement:
            # Deterministic weighted sampling with replacement using cumulative distribution
            cumulative_weights = np.cumsum(weights)
            samples = []
            for i in range(num_samples):
                # Use deterministic value based on sample index
                rand_val = (hash((i, len(temperatures))) % 1000000) / 1000000.0
                # Find index using cumulative distribution
                selected_idx = np.searchsorted(cumulative_weights, rand_val)
                samples.append(min(selected_idx, len(temperatures) - 1))
            return np.array(samples)
        else:
            # Weighted sampling without replacement using deterministic selection
            indices = np.arange(len(temperatures))
            selected_indices = []
            remaining_indices = list(indices)
            remaining_weights = weights.copy()
            
            for i in range(min(num_samples, len(indices))):
                # Normalize remaining weights
                remaining_weights = remaining_weights / np.sum(remaining_weights)
                # Use deterministic value based on iteration
                rand_val = (hash((i, len(remaining_indices))) % 1000000) / 1000000.0
                # Find index using cumulative distribution
                cumulative_weights = np.cumsum(remaining_weights)
                selected_idx = np.searchsorted(cumulative_weights, rand_val)
                selected_idx = min(selected_idx, len(remaining_indices) - 1)
                selected_indices.append(remaining_indices[selected_idx])
                remaining_indices.pop(selected_idx)
                remaining_weights = np.delete(remaining_weights, selected_idx)
            
            return np.array(selected_indices)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def load_data(self) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Load data from the dataset.
        
        Returns:
            List of (graph, targets) tuples
        """
        pass
    
    @abstractmethod
    def get_temperature(self, sample: Tuple[Data, Dict[str, torch.Tensor]]) -> float:
        """Extract temperature from a sample.
        
        Args:
            sample: (graph, targets) tuple
            
        Returns:
            Temperature in Kelvin
        """
        pass
    
    def convert_units(self, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert units to standard eV units."""
        # For now, just return targets as-is since JARVIS is already in eV
        return targets


class JARVISDatasetLoader(BaseDatasetLoader):
    """Loader for JARVIS datasets."""
    
    def load_data(self) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Load JARVIS dataset."""
        self.logger.info(f"Loading JARVIS dataset from {self.config.path}")
        
        # Use existing JARVIS loader
        dataset = JARVISDFTDataset(
            data_path=self.config.path,
            cutoff_radius=6.0,
            max_samples=self.config.max_samples
        )
        
        data_samples = []
        for i in range(len(dataset)):
            graph, targets = dataset[i]
            
            # Convert units
            targets = self.convert_units(targets)
            
            # Add domain information
            graph.domain_id = self.config.domain_id.value
            graph.dataset_name = self.config.name
            
            data_samples.append((graph, targets))
        
        self.logger.info(f"Loaded {len(data_samples)} samples from JARVIS")
        return data_samples
    
    def get_temperature(self, sample: Tuple[Data, Dict[str, torch.Tensor]]) -> float:
        """Extract temperature from JARVIS sample."""
        # JARVIS datasets typically don't have explicit temperature
        # Use a default temperature based on the domain
        if self.config.domain_id == DatasetDomain.JARVIS_DFT:
            return 300.0  # Room temperature for DFT calculations
        elif self.config.domain_id == DatasetDomain.JARVIS_ELASTIC:
            return 0.0  # Zero temperature for elastic properties
        else:
            return 300.0  # Default room temperature


class OC20DatasetLoader(BaseDatasetLoader):
    """Loader for OC20/OC22 datasets in extxyz format."""
    
    def load_data(self) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Load OC20/OC22 dataset from extxyz files."""
        self.logger.info(f"Loading OC20/OC22 dataset from {self.config.path}")
        
        import glob
        import lzma
        import re
        
        data_samples = []
        extxyz_files = glob.glob(f"{self.config.path}/*.extxyz.xz")
        
        if not extxyz_files:
            self.logger.warning(f"No extxyz.xz files found in {self.config.path}")
            return []
        
        self.logger.info(f"Found {len(extxyz_files)} extxyz files")
        
        for file_path in extxyz_files[:5]:  # Limit for testing
            try:
                with lzma.open(file_path, 'rt') as f:
                    content = f.read()
                    
                # Parse extxyz format
                samples = self._parse_extxyz(content)
                data_samples.extend(samples)
                
                if len(data_samples) >= (self.config.max_samples or 1000):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(data_samples)} samples from OC20/OC22")
        return data_samples[:self.config.max_samples] if self.config.max_samples else data_samples
    
    def _parse_extxyz(self, content: str) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Parse extxyz content."""
        import re
        from torch_geometric.data import Data
        
        samples = []
        lines = content.strip().split('\n')
        i = 0
        
        while i < len(lines):
            try:
                # Skip empty lines
                while i < len(lines) and not lines[i].strip():
                    i += 1
                
                if i >= len(lines):
                    break
                
                # First line: number of atoms
                try:
                    num_atoms = int(lines[i].strip())
                except ValueError:
                    # If this line can't be parsed as int, skip it
                    i += 1
                    continue
                
                i += 1
                
                if i >= len(lines):
                    break
                
                # Second line: properties and energy
                props_line = lines[i].strip()
                i += 1
                
                # Extract energy from properties line
                energy_match = re.search(r'energy=([-\d.]+)', props_line)
                if not energy_match:
                    # Skip this structure
                    i += num_atoms
                    continue
                
                energy = float(energy_match.group(1))
                
                # Parse atomic coordinates and forces
                positions = []
                atomic_numbers = []
                forces = []
                
                atoms_parsed = 0
                while atoms_parsed < num_atoms and i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue
                    
                    parts = line.split()
                    if len(parts) < 4:
                        i += 1
                        continue
                    
                    try:
                        element = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        
                        # Convert element to atomic number
                        atomic_num = self._element_to_atomic_number(element)
                        
                        positions.append([x, y, z])
                        atomic_numbers.append(atomic_num)
                        
                        # Extract forces if available
                        if len(parts) >= 7:
                            try:
                                fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
                                forces.append([fx, fy, fz])
                            except ValueError:
                                forces.append([0.0, 0.0, 0.0])
                        else:
                            forces.append([0.0, 0.0, 0.0])
                        
                        atoms_parsed += 1
                        i += 1
                        
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        i += 1
                        continue
                
                # Skip remaining lines if we didn't parse all atoms
                if atoms_parsed < num_atoms:
                    i += (num_atoms - atoms_parsed)
                
                if len(positions) < 2:  # Skip single atoms
                    continue
                
                # Create PyTorch Geometric Data object
                positions = torch.tensor(positions, dtype=torch.float32)
                atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
                forces = torch.tensor(forces, dtype=torch.float32)
                
                # Create simple graph (no edges for now)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)  # Empty edge attributes
                
                # Create node features (one-hot encoding of atomic numbers)
                x = torch.zeros(len(atomic_numbers), 83)  # Support up to Bi (83)
                for idx, z in enumerate(atomic_numbers):
                    if z <= 83:
                        x[idx, z-1] = 1.0
                
                graph = Data(
                    x=x,
                    pos=positions,
                    atomic_numbers=atomic_numbers,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=torch.zeros(len(atomic_numbers), dtype=torch.long),
                    domain_id=self.config.domain_id.value,
                    dataset_name=self.config.name
                )
                
                # Create targets
                targets = {
                    'energy': torch.tensor(energy, dtype=torch.float32),
                    'forces': forces,
                    'stress': torch.zeros(3, 3, dtype=torch.float32)
                }
                
                # Convert units
                targets = self.convert_units(targets)
                
                samples.append((graph, targets))
                
            except Exception as e:
                self.logger.error(f"Error parsing extxyz at line {i}: {e}")
                i += 1
                continue
        
        return samples
    
    def _element_to_atomic_number(self, element: str) -> int:
        """Convert element symbol to atomic number."""
        element_map = {
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
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83
        }
        return element_map.get(element, 1)  # Default to H if unknown
    
    def get_temperature(self, sample: Tuple[Data, Dict[str, torch.Tensor]]) -> float:
        """Extract temperature from OC20/OC22 sample."""
        # OC20/OC22 datasets typically represent catalytic reactions at various temperatures
        # Use a default temperature range
        if self.config.domain_id == DatasetDomain.OC20_S2EF:
            return 500.0  # Typical catalytic temperature
        elif self.config.domain_id == DatasetDomain.OC22_S2EF:
            return 600.0  # Slightly higher for OC22
        else:
            return 500.0  # Default catalytic temperature


class ANI1xDatasetLoader(BaseDatasetLoader):
    """Loader for ANI1x datasets in HDF5 format."""
    
    def load_data(self) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Load ANI1x dataset from HDF5 file."""
        self.logger.info(f"Loading ANI1x dataset from {self.config.path}")
        
        import h5py
        
        data_samples = []
        
        try:
            with h5py.File(self.config.path, 'r') as f:
                # Get all molecular formula groups
                formula_groups = list(f.keys())
                self.logger.info(f"Found {len(formula_groups)} molecular formula groups")
                
                # Limit groups for testing
                max_groups = min(50, len(formula_groups))
                
                for formula in formula_groups[:max_groups]:
                    try:
                        group = f[formula]
                        
                        # Extract data
                        if 'coordinates' not in group or 'atomic_numbers' not in group:
                            continue
                        
                        coordinates = group['coordinates'][:]  # (n_confs, n_atoms, 3)
                        atomic_numbers = group['atomic_numbers'][:]  # (n_atoms,)
                        
                        # Get energies (use wb97x_dz.energy if available, otherwise ccsd(t)_cbs.energy)
                        if 'wb97x_dz.energy' in group:
                            energies = group['wb97x_dz.energy'][:]  # (n_confs,)
                        elif 'ccsd(t)_cbs.energy' in group:
                            energies = group['ccsd(t)_cbs.energy'][:]
                        else:
                            continue
                        
                        # Get forces if available
                        if 'wb97x_dz.forces' in group:
                            forces = group['wb97x_dz.forces'][:]  # (n_confs, n_atoms, 3)
                        else:
                            forces = None
                        
                        # Process each conformation
                        n_confs = coordinates.shape[0]
                        max_confs = min(100, n_confs)  # Limit conformations per molecule
                        
                        for i in range(max_confs):
                            try:
                                pos = torch.tensor(coordinates[i], dtype=torch.float32)
                                z = torch.tensor(atomic_numbers, dtype=torch.long)
                                energy = torch.tensor(energies[i], dtype=torch.float32)
                                
                                # Create forces tensor
                                if forces is not None:
                                    force_tensor = torch.tensor(forces[i], dtype=torch.float32)
                                else:
                                    force_tensor = torch.zeros_like(pos)
                                
                                # Create simple graph (no edges for now)
                                edge_index = torch.empty((2, 0), dtype=torch.long)
                                edge_attr = torch.empty((0, 1), dtype=torch.float32)  # Empty edge attributes
                                
                                # Create node features (one-hot encoding of atomic numbers)
                                x = torch.zeros(len(z), 83)  # Support up to Bi (83)
                                for i, atomic_num in enumerate(z):
                                    if atomic_num <= 83:
                                        x[i, atomic_num-1] = 1.0
                                
                                graph = Data(
                                    x=x,
                                    pos=pos,
                                    atomic_numbers=z,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    batch=torch.zeros(len(z), dtype=torch.long),
                                    domain_id=self.config.domain_id.value,
                                    dataset_name=self.config.name
                                )
                                
                                # Create targets
                                targets = {
                                    'energy': energy,
                                    'forces': force_tensor,
                                    'stress': torch.zeros(3, 3, dtype=torch.float32)
                                }
                                
                                # Convert units
                                targets = self.convert_units(targets)
                                
                                data_samples.append((graph, targets))
                                
                                if len(data_samples) >= (self.config.max_samples or 10000):
                                    break
                                    
                            except Exception as e:
                                self.logger.error(f"Error processing conformation {i} of {formula}: {e}")
                                continue
                        
                        if len(data_samples) >= (self.config.max_samples or 10000):
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Error processing molecular formula {formula}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error loading ANI1x dataset: {e}")
            return []
        
        self.logger.info(f"Loaded {len(data_samples)} samples from ANI1x")
        return data_samples[:self.config.max_samples] if self.config.max_samples else data_samples
    
    def get_temperature(self, sample: Tuple[Data, Dict[str, torch.Tensor]]) -> float:
        """Extract temperature from ANI1x sample."""
        # ANI1x datasets represent molecular conformations at various temperatures
        # Use a default temperature range for molecular systems
        return 300.0  # Room temperature for molecular conformations


class UnifiedDatasetRegistry:
    """Unified registry for multiple datasets with domain mixing."""
    
    def __init__(self, config: UnifiedDatasetConfig):
        self.config = config
        self.logger = logging.getLogger("UnifiedDatasetRegistry")
        self.temperature_sampler = TemperatureSampler(config.temperature_tau)
        
        # Initialize dataset loaders
        self.loaders = {}
        for dataset_config in config.datasets:
            if not dataset_config.enabled:
                continue
                
            loader = self._create_loader(dataset_config)
            if loader:
                self.loaders[dataset_config.domain_id] = loader
        
        self.logger.info(f"Initialized registry with {len(self.loaders)} datasets")
    
    def _create_loader(self, config: DatasetConfig) -> Optional[BaseDatasetLoader]:
        """Create appropriate loader for dataset."""
        if config.domain_id in [DatasetDomain.JARVIS_DFT, DatasetDomain.JARVIS_ELASTIC, 
                               DatasetDomain.JARVIS_ELECTRONIC]:
            return JARVISDatasetLoader(config)
        elif config.domain_id in [DatasetDomain.OC20_S2EF, DatasetDomain.OC20_IS2RE, DatasetDomain.OC22_S2EF, DatasetDomain.OC22_IS2RE]:
            return OC20DatasetLoader(config)
        elif config.domain_id == DatasetDomain.ANI1X:
            return ANI1xDatasetLoader(config)
        else:
            self.logger.warning(f"Unknown dataset domain: {config.domain_id}")
            return None
    
    def _normalize_energies(self, samples: List[Tuple[Data, Dict[str, torch.Tensor]]]) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Normalize energy values across all samples."""
        if not samples:
            return samples
        
        # Extract all energy values
        energies = []
        for _, targets in samples:
            if 'energy' in targets:
                energies.append(targets['energy'].item())
        
        if not energies:
            self.logger.warning("No energy values found for normalization")
            return samples
        
        # Compute statistics
        energy_array = np.array(energies)
        mean_energy = np.mean(energy_array)
        std_energy = np.std(energy_array)
        
        # Store normalization parameters
        self.config.energy_mean = mean_energy
        self.config.energy_std = std_energy
        
        self.logger.info(f"Energy normalization: mean={mean_energy:.3f}, std={std_energy:.3f}")
        self.logger.info(f"Energy range before normalization: [{np.min(energy_array):.3f}, {np.max(energy_array):.3f}]")
        
        # Normalize energies
        normalized_samples = []
        for graph, targets in samples:
            normalized_targets = targets.copy()
            if 'energy' in targets:
                normalized_energy = (targets['energy'] - mean_energy) / std_energy
                normalized_targets['energy'] = normalized_energy
            normalized_samples.append((graph, normalized_targets))
        
        # Log normalized range
        normalized_energies = [(targets['energy'] - mean_energy) / std_energy for _, targets in normalized_samples if 'energy' in targets]
        if normalized_energies:
            self.logger.info(f"Energy range after normalization: [{np.min(normalized_energies):.3f}, {np.max(normalized_energies):.3f}]")
        
        return normalized_samples
    
    def load_all_datasets(self) -> Dict[str, List[Tuple[Data, Dict[str, torch.Tensor]]]]:
        """Load all enabled datasets and split into train/val."""
        all_samples = []
        
        for domain_id, loader in self.loaders.items():
            try:
                samples = loader.load_data()
                all_samples.extend(samples)
                self.logger.info(f"Loaded {len(samples)} samples from {domain_id.value}")
            except Exception as e:
                self.logger.error(f"Failed to load {domain_id.value}: {e}")
        
        self.logger.info(f"Total samples loaded: {len(all_samples)}")
        
        # Apply energy normalization if enabled
        if self.config.normalize_energies and all_samples:
            all_samples = self._normalize_energies(all_samples)
        
        # Split into train/val
        if not all_samples:
            return {'train': [], 'val': []}
        
        # Shuffle samples
        import random
        random.seed(self.config.random_seed)
        random.shuffle(all_samples)
        
        # Calculate split sizes
        val_size = int(len(all_samples) * self.config.validation_split)
        train_size = len(all_samples) - val_size
        
        # Split samples
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:train_size + val_size]
        
        self.logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")
        
        return {
            'train': train_samples,
            'val': val_samples
        }
    
    def collate_fn(self, batch):
        """Collate function for batching."""
        graphs, targets = zip(*batch)
        
        # Batch PyTorch Geometric Data objects
        batched_graph = Batch.from_data_list(graphs)
        
        # Stack targets
        energy_batch = torch.stack([t['energy'] for t in targets])
        forces_batch = torch.cat([t['forces'] for t in targets], dim=0)
        stress_batch = torch.stack([t['stress'] for t in targets])
        
        # Create batch targets
        batch_targets = {
            'energy': energy_batch,
            'forces': forces_batch,
            'stress': stress_batch
        }
        
        return batched_graph, batch_targets
    
    def create_mixed_dataset(self, max_samples: Optional[int] = None) -> List[Tuple[Data, Dict[str, torch.Tensor]]]:
        """Create mixed dataset based on configuration strategy."""
        datasets = self.load_all_datasets()
        all_samples = datasets['train'] + datasets['val']  # Combine for mixing
        
        if not all_samples:
            self.logger.warning("No samples loaded")
            return []
        
        # Extract temperatures
        temperatures = np.array([self._get_sample_temperature(sample) for sample in all_samples])
        
        if self.config.mix_strategy == "temperature":
            # Temperature-based sampling
            if max_samples is None:
                max_samples = len(all_samples)
            
            sampled_indices = self.temperature_sampler.sample_indices(
                temperatures, max_samples, replacement=True
            )
            
            mixed_samples = [all_samples[i] for i in sampled_indices]
            
        elif self.config.mix_strategy == "uniform":
            # Uniform random sampling
            if max_samples is None:
                max_samples = len(all_samples)
            
            # Deterministic sampling with replacement using modulo
            indices = []
            for i in range(max_samples):
                # Use deterministic index based on sample number
                selected_idx = (hash((i, len(all_samples))) % len(all_samples))
                indices.append(selected_idx)
            indices = np.array(indices)
            mixed_samples = [all_samples[i] for i in indices]
            
        elif self.config.mix_strategy == "weighted":
            # Weighted sampling based on dataset weights
            weights = []
            for sample in all_samples:
                graph, _ = sample
                domain_id = DatasetDomain(graph.domain_id)
                dataset_config = next((d for d in self.config.datasets if d.domain_id == domain_id), None)
                weight = dataset_config.weight if dataset_config else 1.0
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            if max_samples is None:
                max_samples = len(all_samples)
            
            # Deterministic weighted sampling with replacement using cumulative distribution
            indices = []
            cumulative_weights = np.cumsum(weights)
            for i in range(max_samples):
                # Use deterministic value based on sample number
                rand_val = (hash((i, len(all_samples))) % 1000000) / 1000000.0
                # Find index using cumulative distribution
                selected_idx = np.searchsorted(cumulative_weights, rand_val)
                indices.append(min(selected_idx, len(all_samples) - 1))
            indices = np.array(indices)
            mixed_samples = [all_samples[i] for i in indices]
        
        else:
            self.logger.warning(f"Unknown mix strategy: {self.config.mix_strategy}")
            mixed_samples = all_samples
        
        self.logger.info(f"Created mixed dataset with {len(mixed_samples)} samples")
        return mixed_samples
    
    def _get_sample_temperature(self, sample: Tuple[Data, Dict[str, torch.Tensor]]) -> float:
        """Get temperature for a sample."""
        graph, _ = sample
        domain_id = DatasetDomain(graph.domain_id)
        
        if domain_id in self.loaders:
            return self.loaders[domain_id].get_temperature(sample)
        else:
            return 300.0  # Default temperature
    
    def create_dataloader(self, mixed_samples: List[Tuple[Data, Dict[str, torch.Tensor]]],
                         batch_size: int = 32, shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader from mixed samples."""
        
        def collate_fn(batch):
            """Collate function for batching."""
            graphs, targets = zip(*batch)
            
            # Batch PyTorch Geometric Data objects
            batched_graph = Batch.from_data_list(graphs)
            
            # Stack targets
            energy_batch = torch.stack([t['energy'] for t in targets])
            forces_batch = torch.cat([t['forces'] for t in targets], dim=0)
            stress_batch = torch.stack([t['stress'] for t in targets])
            
            # Create batch targets
            batch_targets = {
                'energy': energy_batch,
                'forces': forces_batch,
                'stress': stress_batch
            }
            
            return batched_graph, batch_targets
        
        return DataLoader(
            mixed_samples,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


def create_unified_dataloader(config: UnifiedDatasetConfig, 
                             batch_size: int = 32,
                             max_samples: Optional[int] = None,
                             shuffle: bool = True,
                             num_workers: int = 0) -> DataLoader:
    """Create unified dataloader from configuration.
    
    Args:
        config: Unified dataset configuration
        batch_size: Batch size for dataloader
        max_samples: Maximum number of samples to use
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader with mixed dataset
    """
    registry = UnifiedDatasetRegistry(config)
    mixed_samples = registry.create_mixed_dataset(max_samples)
    return registry.create_dataloader(mixed_samples, batch_size, shuffle, num_workers)


def create_unified_config_from_hydra(config) -> UnifiedDatasetConfig:
    """Create UnifiedDatasetConfig from Hydra config."""
    datasets = []
    
    # JARVIS-DFT
    if config.dataset.jarvis_dft.enabled:
        datasets.append(DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
            energy_unit=config.dataset.jarvis_dft.energy_unit,
            force_unit=config.dataset.jarvis_dft.force_unit,
            stress_unit=config.dataset.jarvis_dft.stress_unit,
            temperature_range=tuple(config.dataset.jarvis_dft.temperature_range),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
            max_samples=config.dataset.jarvis_dft.max_samples,
            enabled=True
        ))
    
    # JARVIS-Elastic
    if config.dataset.jarvis_elastic.enabled:
        datasets.append(DatasetConfig(
            domain_id=DatasetDomain.JARVIS_ELASTIC,
            name="JARVIS-Elastic",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-6-6-2019.json",
            energy_unit=config.dataset.jarvis_elastic.energy_unit,
            force_unit=config.dataset.jarvis_elastic.force_unit,
            stress_unit=config.dataset.jarvis_elastic.stress_unit,
            temperature_range=tuple(config.dataset.jarvis_elastic.temperature_range),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
            max_samples=config.dataset.jarvis_elastic.max_samples,
            enabled=True
        ))
    
    # OC20-S2EF
    if config.dataset.oc20_s2ef.enabled:
        datasets.append(DatasetConfig(
            domain_id=DatasetDomain.OC20_S2EF,
            name="OC20-S2EF",
            path="data/oc20/s2ef_train_200K/s2ef_train_200K",
            energy_unit=config.dataset.oc20_s2ef.energy_unit,
            force_unit=config.dataset.oc20_s2ef.force_unit,
            stress_unit=config.dataset.oc20_s2ef.stress_unit,
            temperature_range=tuple(config.dataset.oc20_s2ef.temperature_range),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
            max_samples=config.dataset.oc20_s2ef.max_samples,
            enabled=True
        ))
    
    # OC22-S2EF
    if config.dataset.oc22_s2ef.enabled:
        datasets.append(DatasetConfig(
            domain_id=DatasetDomain.OC22_S2EF,
            name="OC22-S2EF",
            path="data/oc22/s2ef_val_id/s2ef_val_id",
            energy_unit=config.dataset.oc22_s2ef.energy_unit,
            force_unit=config.dataset.oc22_s2ef.force_unit,
            stress_unit=config.dataset.oc22_s2ef.stress_unit,
            temperature_range=tuple(config.dataset.oc22_s2ef.temperature_range),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
            max_samples=config.dataset.oc22_s2ef.max_samples,
            enabled=True
        ))
    
    # ANI1x
    if config.dataset.ani1x.enabled:
        datasets.append(DatasetConfig(
            domain_id=DatasetDomain.ANI1X,
            name="ANI1x",
            path="data/ani1x/ani1x_dataset.h5",
            energy_unit=config.dataset.ani1x.energy_unit,
            force_unit=config.dataset.ani1x.force_unit,
            stress_unit=config.dataset.ani1x.stress_unit,
            temperature_range=tuple(config.dataset.ani1x.temperature_range),
            atomic_species=[1, 6, 7, 8, 9],
            max_samples=config.dataset.ani1x.max_samples,
            enabled=True
        ))
    
    return UnifiedDatasetConfig(
        datasets=datasets,
        mix_strategy=config.dataset.mix_strategy,
        temperature_tau=config.dataset.temperature_tau,
        unit_conversion=config.dataset.unit_conversion,
        validation_split=config.dataset.validation_split,
        test_split=config.dataset.test_split,
        random_seed=config.dataset.random_seed,
        normalize_energies=True
    )


# Example configuration
def create_example_config() -> UnifiedDatasetConfig:
    """Create example configuration for unified dataset registry."""
    datasets = [
        DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
            energy_unit="eV",
            force_unit="eV/Å",
            stress_unit="eV/Å³",
            temperature_range=(0, 1000),
            atomic_species=[1, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83],
            max_samples=50000,
            weight=1.0,
            enabled=True
        ),
        # Add more datasets as needed
    ]
    
    return UnifiedDatasetConfig(
        datasets=datasets,
        mix_strategy="temperature",
        temperature_tau=1.0,
        unit_conversion=True,
        validation_split=0.1,
        test_split=0.1,
        random_seed=42
    )
