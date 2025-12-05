#!/usr/bin/env python3
"""JARVIS-DFT dataset loader for GNN training."""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import random

from graphs.periodic_graph import PeriodicGraph


class JARVISDFTDataset(Dataset):
    """JARVIS-DFT dataset loader."""
    
    def __init__(self, 
                 data_path: str,
                 cutoff_radius: float = 6.0,
                 max_samples: int = None,
                 shuffle: bool = True):
        """Initialize JARVIS-DFT dataset.
        
        Args:
            data_path: Path to JARVIS-DFT JSON file
            cutoff_radius: Cutoff radius for graph construction
            max_samples: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle the data
        """
        self.data_path = Path(data_path)
        self.cutoff_radius = cutoff_radius
        self.max_samples = max_samples
        self.shuffle = shuffle
        
        # Initialize graph builder
        self.graph_builder = PeriodicGraph(cutoff_radius=cutoff_radius)
        
        # Load data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} samples from JARVIS-DFT dataset")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load JARVIS-DFT data from JSON file."""
        print(f"Loading JARVIS-DFT data from {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Filter out samples with missing data
        filtered_data = []
        for sample in data:
            if self._is_valid_sample(sample):
                filtered_data.append(sample)
        
        print(f"Filtered to {len(filtered_data)} valid samples")
        
        # Limit samples if requested
        if self.max_samples is not None:
            filtered_data = filtered_data[:self.max_samples]
            print(f"Limited to {len(filtered_data)} samples")
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(filtered_data)
        
        return filtered_data
    
    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if sample has valid data."""
        try:
            # Check for structure data - handle both JARVIS formats
            has_atoms_format = 'atoms' in sample
            has_final_str_format = 'final_str' in sample
            
            if not (has_atoms_format or has_final_str_format):
                return False
            
            if has_atoms_format:
                atoms = sample['atoms']
                if 'lattice_mat' not in atoms or 'coords' not in atoms or 'elements' not in atoms:
                    return False
                # Check that we have at least 2 atoms
                if len(atoms['elements']) < 2:
                    return False
            else:
                # final_str format
                final_str = sample['final_str']
                if 'lattice' not in final_str or 'sites' not in final_str:
                    return False
                # Check that we have at least 2 atoms
                if len(final_str['sites']) < 2:
                    return False
            
            # Check for energy - handle both JARVIS formats
            has_energy = ('optb88vdw_total_energy' in sample or 'fin_en' in sample)
            if not has_energy:
                return False
            
            # Check that energy is not NaN
            energy = sample.get('optb88vdw_total_energy', sample.get('fin_en', 0.0))
            if np.isnan(energy) or np.isinf(energy):
                return False
            
            return True
            
        except (KeyError, TypeError, ValueError):
            return False
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """Get a sample from the dataset."""
        sample = self.data[idx]
        
        # Extract atomic data - handle both JARVIS formats
        if 'atoms' in sample:
            # Original atoms format
            atoms = sample['atoms']
            lattice_matrix = np.array(atoms['lattice_mat'], dtype=np.float32)
            coords = np.array(atoms['coords'], dtype=np.float32)
            elements = atoms['elements']
            
            # Convert elements to atomic numbers
            atomic_numbers = self._elements_to_atomic_numbers(elements)
            
            # Convert fractional coordinates to Cartesian
            cartesian_coords = self._frac_to_cartesian(coords, lattice_matrix)
            
        else:
            # final_str format
            final_str = sample['final_str']
            lattice_data = final_str['lattice']
            sites = final_str['sites']
            
            # Extract lattice matrix
            lattice_matrix = np.array(lattice_data['matrix'], dtype=np.float32)
            
            # Extract atomic positions and elements
            cartesian_coords = []
            elements = []
            for site in sites:
                # Extract element from species
                species = site['species'][0]  # Take first species
                element = species['element']
                elements.append(element)
                
                # Extract Cartesian coordinates
                xyz = site['xyz']
                cartesian_coords.append(xyz)
            
            cartesian_coords = np.array(cartesian_coords, dtype=np.float32)
            atomic_numbers = self._elements_to_atomic_numbers(elements)
        
        # Build graph
        graph = self.graph_builder.build_graph(
            positions=cartesian_coords,
            atomic_numbers=atomic_numbers,
            cell_vectors=lattice_matrix
        )
        
        # Extract targets - handle both JARVIS formats
        energy = sample.get('optb88vdw_total_energy', sample.get('fin_en', 0.0))
        
        # Create targets dictionary
        targets = {
            'energy': torch.tensor(energy, dtype=torch.float32),
            'forces': torch.zeros(len(atomic_numbers), 3, dtype=torch.float32),  # JARVIS doesn't have forces
            'stress': torch.zeros(3, 3, dtype=torch.float32),  # JARVIS doesn't have stress
        }
        
        return graph, targets
    
    def _elements_to_atomic_numbers(self, elements: List[str]) -> np.ndarray:
        """Convert element symbols to atomic numbers."""
        element_to_z = {
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
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
            'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }
        
        atomic_numbers = []
        for element in elements:
            if element in element_to_z:
                atomic_numbers.append(element_to_z[element])
            else:
                print(f"Warning: Unknown element {element}, using atomic number 1")
                atomic_numbers.append(1)  # Default to H
        
        return np.array(atomic_numbers, dtype=np.int32)
    
    def _frac_to_cartesian(self, frac_coords: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to Cartesian coordinates."""
        return np.dot(frac_coords, lattice_matrix)


def create_jarvis_dataloader(data_path: str,
                           batch_size: int = 16,
                           cutoff_radius: float = 6.0,
                           max_samples: int = None,
                           shuffle: bool = True,
                           num_workers: int = 0) -> DataLoader:
    """Create JARVIS-DFT dataloader.
    
    Args:
        data_path: Path to JARVIS-DFT JSON file
        batch_size: Batch size
        cutoff_radius: Cutoff radius for graph construction
        max_samples: Maximum number of samples to load
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for JARVIS-DFT dataset
    """
    dataset = JARVISDFTDataset(
        data_path=data_path,
        cutoff_radius=cutoff_radius,
        max_samples=max_samples,
        shuffle=shuffle
    )
    
    def collate_fn(batch):
        """Collate function for batching."""
        graphs, targets = zip(*batch)
        
        # Batch PyTorch Geometric Data objects
        from torch_geometric.data import Batch
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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False
    )


if __name__ == "__main__":
    # Test the dataset loader
    data_path = "/home/arash/dft/data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json"
    
    print("Testing JARVIS-DFT dataset loader...")
    
    # Create dataloader
    dataloader = create_jarvis_dataloader(
        data_path=data_path,
        batch_size=2,
        max_samples=10,
        shuffle=False
    )
    
    # Test a few batches
    for i, (graph, targets) in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Graph nodes: {graph.pos.shape[0]}")
        print(f"  Graph edges: {graph.edge_index.shape[1]}")
        print(f"  Energy shape: {targets['energy'].shape}")
        print(f"  Energy values: {targets['energy']}")
        print(f"  Forces shape: {targets['forces'].shape}")
        print(f"  Stress shape: {targets['stress'].shape}")
        
        if i >= 2:  # Test only first 3 batches
            break
    
    print("\nJARVIS-DFT dataset loader test completed!")
