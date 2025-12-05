"""Dataset loader for MPtrj figshare JSON data."""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import ase
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from graphs.periodic_graph import GraphBatch, PeriodicGraph

logger = logging.getLogger(__name__)


@dataclass
class MPtrjEntry:
    """Single MPtrj trajectory entry."""
    
    parent_id: str  # Parent structure ID
    structure_id: str
    step: int
    atoms: Atoms
    energy: float  # eV
    forces: np.ndarray  # (n_atoms, 3) eV/Å
    stress: np.ndarray  # (3, 3) eV/Å³
    magmoms: np.ndarray  # (n_atoms,) μB
    metadata: Dict[str, Any]


class MPtrjDataset:
    """Dataset loader for MPtrj figshare JSON trajectory data."""
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 graph_builder: Optional[PeriodicGraph] = None,
                 cutoff_radius: float = 5.0,
                 max_atoms: Optional[int] = None,
                 include_magmoms: bool = True,
                 max_entries: Optional[int] = None):
        """Initialize MPtrj dataset.
        
        Args:
            data_path: Path to MPtrj figshare JSON file
            graph_builder: PeriodicGraph instance for building graphs
            cutoff_radius: Cutoff radius for neighbor construction
            max_atoms: Maximum number of atoms per structure (filtering)
            include_magmoms: Whether to include magnetic moments
        """
        self.data_path = Path(data_path)
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.include_magmoms = include_magmoms
        self.max_entries = max_entries
        
        if graph_builder is None:
            self.graph_builder = PeriodicGraph(cutoff_radius=cutoff_radius)
        else:
            self.graph_builder = graph_builder
        
        # Load and parse data
        self._load_data()
        
        # Create index files if they don't exist
        self.index_path = self.data_path.parent / f"{self.data_path.stem}_index.json"
        self._ensure_index()
    
    def _load_data(self):
        """Load and parse MPtrj JSON data."""
        logger.info(f"Loading MPtrj data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"MPtrj data file not found: {self.data_path}")
        
        # Check if index exists first
        self.index_path = self.data_path.parent / f"{self.data_path.stem}_index.json"
        
        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
            
            # Use lazy loading - don't parse the entire JSON file
            logger.info(f"Using lazy loading with {len(self.index['entries'])} indexed entries")
            self.entries: List[MPtrjEntry] = []  # Will be loaded on-demand
            self._raw_data = None  # Will be loaded when needed
            self._lazy_loading = True
        else:
            # Full loading when no index exists
            logger.info("No index found, performing full data loading")
            with open(self.data_path, 'r') as f:
                self._raw_data = json.load(f)
            
            self.entries: List[MPtrjEntry] = []
            
            # The MPtrj format is: {parent_id: {structure_id: {structure, e_form, etc.}}}
            for parent_id, structures in self._raw_data.items():
                if not isinstance(structures, dict):
                    logger.warning(f"Skipping malformed parent entry: {parent_id}")
                    continue
                
                for structure_id, structure_data in structures.items():
                    try:
                        entry = self._parse_mptrj_structure(parent_id, structure_id, structure_data)
                        if entry is not None:
                            self.entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to parse {parent_id}/{structure_id}: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.entries)} structure entries")
            
            # Filter by max_atoms if specified
            if self.max_atoms is not None:
                filtered_entries = []
                for entry in self.entries:
                    if len(entry.atoms) <= self.max_atoms:
                        filtered_entries.append(entry)
                
                logger.info(f"Filtered to {len(filtered_entries)} entries with ≤{self.max_atoms} atoms")
                self.entries = filtered_entries
            
            self._lazy_loading = False
    
    def _parse_mptrj_structure(self, parent_id: str, structure_id: str, structure_data: Dict) -> Optional[MPtrjEntry]:
        """Parse a single MPtrj structure entry."""
        try:
            # Extract pymatgen Structure
            structure_dict = structure_data.get('structure')
            if not structure_dict:
                return None
            
            structure = Structure.from_dict(structure_dict)
            
            # Convert to ASE Atoms
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            
            # Extract properties
            energy = float(structure_data.get('e_form', 0.0))  # Formation energy in eV
            forces = np.array(structure_data.get('forces', np.zeros((len(atoms), 3))))  # eV/Å
            stress = np.array(structure_data.get('stress', np.zeros((3, 3))))  # eV/Å³
            magmoms = np.array(structure_data.get('magmoms', np.zeros(len(atoms))))  # μB
            
            if not self.include_magmoms:
                magmoms = np.zeros(len(atoms))
            
            # Extract metadata
            metadata = {
                'parent_id': parent_id,
                'structure_id': structure_id,
                'volume': atoms.get_volume(),
                'density': atoms.get_volume() / len(atoms) if len(atoms) > 0 else 0.0,
                'formula': structure_data.get('formula', ''),
                'e_form': energy,
            }
            
            return MPtrjEntry(
                parent_id=parent_id,
                structure_id=structure_id,
                step=0,  # No trajectory steps in this format
                atoms=atoms,
                energy=energy,
                forces=forces,
                stress=stress,
                magmoms=magmoms,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse MPtrj structure {parent_id}/{structure_id}: {e}")
            return None
    
    def _parse_trajectory_step(self, structure_id: str, step: int, step_data: Dict) -> Optional[MPtrjEntry]:
        """Parse a single trajectory step (legacy format)."""
        try:
            # Extract structure information
            lattice = np.array(step_data['lattice'])
            positions = np.array(step_data['positions'])
            atomic_numbers = np.array(step_data['atomic_numbers'])
            
            # Create ASE Atoms object
            atoms = Atoms(
                symbols=atomic_numbers,
                positions=positions,
                cell=lattice,
                pbc=True
            )
            
            # Extract properties
            energy = float(step_data.get('energy', 0.0))  # eV
            forces = np.array(step_data.get('forces', np.zeros((len(atoms), 3))))  # eV/Å
            
            # Parse stress tensor
            stress_data = step_data.get('stress', np.zeros((3, 3)))
            if isinstance(stress_data, list):
                stress = np.array(stress_data)
            else:
                stress = np.array(stress_data)
            
            # Ensure stress is (3, 3)
            if stress.shape != (3, 3):
                stress = np.zeros((3, 3))
            
            # Parse magnetic moments
            magmoms = np.array(step_data.get('magmoms', np.zeros(len(atoms))))  # μB
            if not self.include_magmoms:
                magmoms = np.zeros(len(atoms))
            
            # Extract metadata
            metadata = {
                'temperature': step_data.get('temperature', None),
                'pressure': step_data.get('pressure', None),
                'volume': atoms.get_volume(),
                'density': atoms.get_volume() / len(atoms),
                'step': step,
                'structure_id': structure_id
            }
            
            return MPtrjEntry(
                structure_id=structure_id,
                step=step,
                atoms=atoms,
                energy=energy,
                forces=forces,
                stress=stress,
                magmoms=magmoms,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse trajectory step: {e}")
            return None
    
    def _ensure_index(self):
        """Ensure index file exists for efficient data access."""
        if not self.index_path.exists():
            logger.info("Creating new index file")
            self._create_index()
    
    def _create_index(self):
        """Create index file for efficient data access."""
        self.index = {
            'entries': [
                {
                    'parent_id': entry.parent_id,
                    'structure_id': entry.structure_id,
                    'step': entry.step,
                    'n_atoms': len(entry.atoms),
                    'energy': entry.energy,
                    'volume': entry.atoms.get_volume(),
                    'metadata': entry.metadata
                }
                for entry in self.entries
            ],
            'splits': {
                'train': [],
                'val': [],
                'test': []
            },
            'statistics': self._compute_statistics()
        }
        
        # Save index
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        
        logger.info(f"Created index file: {self.index_path}")
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        energies = [entry.energy for entry in self.entries]
        n_atoms_list = [len(entry.atoms) for entry in self.entries]
        volumes = [entry.atoms.get_volume() for entry in self.entries]
        
        # Force statistics
        all_forces = np.concatenate([entry.forces.flatten() for entry in self.entries])
        force_magnitudes = np.linalg.norm(all_forces.reshape(-1, 3), axis=1)
        
        # Stress statistics
        all_stress = np.concatenate([entry.stress.flatten() for entry in self.entries])
        
        # Magnetic moment statistics
        all_magmoms = np.concatenate([entry.magmoms for entry in self.entries])
        
        return {
            'n_entries': len(self.entries),
            'n_unique_structures': len(set(entry.structure_id for entry in self.entries)),
            'energy': {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            },
            'n_atoms': {
                'mean': float(np.mean(n_atoms_list)),
                'std': float(np.std(n_atoms_list)),
                'min': int(np.min(n_atoms_list)),
                'max': int(np.max(n_atoms_list))
            },
            'volume': {
                'mean': float(np.mean(volumes)),
                'std': float(np.std(volumes)),
                'min': float(np.min(volumes)),
                'max': float(np.max(volumes))
            },
            'forces': {
                'mean_magnitude': float(np.mean(force_magnitudes)),
                'std_magnitude': float(np.std(force_magnitudes)),
                'max_magnitude': float(np.max(force_magnitudes))
            },
            'stress': {
                'mean': float(np.mean(all_stress)),
                'std': float(np.std(all_stress)),
                'min': float(np.min(all_stress)),
                'max': float(np.max(all_stress))
            },
            'magmoms': {
                'mean': float(np.mean(all_magmoms)),
                'std': float(np.std(all_magmoms)),
                'max': float(np.max(np.abs(all_magmoms)))
            }
        }
    
    def get_entry(self, idx: int) -> MPtrjEntry:
        """Get a single entry by index."""
        if self._lazy_loading:
            # Load entry on-demand from raw data
            if idx >= len(self.index['entries']):
                raise IndexError(f"Index {idx} out of range for {len(self.index['entries'])} entries")
            
            # Get entry info from index
            entry_info = self.index['entries'][idx]
            parent_id = entry_info['metadata']['parent_id']
            structure_id = entry_info['structure_id']
            
            # Load only the specific entry from JSON file
            entry = self._load_single_entry(parent_id, structure_id)
            
            if entry is None:
                raise ValueError(f"Failed to parse entry {idx}")
            
            return entry
        else:
            # Normal access for pre-loaded entries
            if idx >= len(self.entries):
                raise IndexError(f"Index {idx} out of range for {len(self.entries)} entries")
            return self.entries[idx]
    
    def _load_single_entry(self, parent_id: str, structure_id: str) -> Optional[MPtrjEntry]:
        """Load a single entry from the JSON file using cached raw data."""
        try:
            # Load raw data if not already loaded (this will happen on first access)
            if self._raw_data is None:
                logger.info("Loading raw data for lazy access (one-time operation)")
                with open(self.data_path, 'r') as f:
                    self._raw_data = json.load(f)
                logger.info(f"Loaded raw data with {len(self._raw_data)} parent structures")
            
            # Get the specific entry from cached data
            if parent_id not in self._raw_data:
                logger.warning(f"Parent ID {parent_id} not found in data")
                return None
            
            if structure_id not in self._raw_data[parent_id]:
                logger.warning(f"Structure ID {structure_id} not found in parent {parent_id}")
                return None
            
            structure_data = self._raw_data[parent_id][structure_id]
            
            # Parse the structure
            return self._parse_mptrj_structure(parent_id, structure_id, structure_data)
            
        except Exception as e:
            logger.error(f"Failed to load single entry {parent_id}/{structure_id}: {e}")
            return None
    
    def to_graphbatch(self, entry: MPtrjEntry) -> GraphBatch:
        """Convert MPtrj entry to GraphBatch."""
        # Build graph from atoms
        graph_batch = self.graph_builder.build_batch([entry.atoms])
        
        # Add magnetic moments to node features if requested
        if self.include_magmoms and hasattr(graph_batch, 'node_features'):
            # Append magnetic moments as additional node features
            magmom_features = torch.tensor(entry.magmoms, dtype=torch.float32).unsqueeze(1)
            graph_batch.node_features = torch.cat([graph_batch.node_features, magmom_features], dim=1)
        
        return graph_batch
    
    def iter_batches(self, 
                     split: str = "train",
                     batch_size: int = 32,
                     shuffle: bool = True,
                     device: str = "cpu") -> Iterator[Tuple[GraphBatch, Dict[str, torch.Tensor]]]:
        """Iterate over batches of data.
        
        Args:
            split: Data split ("train", "val", "test")
            batch_size: Batch size
            shuffle: Whether to shuffle data
            device: Device to move data to
            
        Yields:
            Tuple of (GraphBatch, targets_dict)
        """
        if split not in self.index['splits']:
            raise ValueError(f"Unknown split: {split}")
        
        # Get indices for this split
        split_indices = self.index['splits'][split]
        if not split_indices:
            logger.warning(f"No data available for split '{split}'. Using all data.")
            split_indices = list(range(len(self.entries)))
        
        # Shuffle if requested (deterministic shuffle)
        if shuffle:
            # Use deterministic shuffle based on data hash
            data_hash = hash(str(split_indices)) % 1000000
            # Create deterministic permutation using Fisher-Yates algorithm
            indices = split_indices.copy()
            for i in range(len(indices) - 1, 0, -1):
                j = (data_hash + i) % (i + 1)
                indices[i], indices[j] = indices[j], indices[i]
        else:
            indices = split_indices
        
        # Create batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Collect batch data
            batch_entries = [self.entries[idx] for idx in batch_indices]
            
            # Convert to GraphBatch
            batch_atoms = [entry.atoms for entry in batch_entries]
            graph_batch = self.graph_builder.build_batch(batch_atoms)
            
            # Prepare targets
            targets = {
                'energy': torch.tensor([entry.energy for entry in batch_entries], dtype=torch.float32),
                'forces': torch.cat([torch.tensor(entry.forces, dtype=torch.float32) for entry in batch_entries], dim=0),
                'stress': torch.stack([torch.tensor(entry.stress, dtype=torch.float32) for entry in batch_entries]),
                'magmoms': torch.cat([torch.tensor(entry.magmoms, dtype=torch.float32) for entry in batch_entries], dim=0),
                'structure_ids': [entry.structure_id for entry in batch_entries],
                'steps': torch.tensor([entry.step for entry in batch_entries], dtype=torch.long),
                'metadata': [entry.metadata for entry in batch_entries]
            }
            
            # Move to device
            graph_batch = graph_batch.to(device)
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(device)
            
            yield graph_batch, targets
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of each data split."""
        return {
            split: len(indices) 
            for split, indices in self.index['splits'].items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.index['statistics']
    
    def __len__(self):
        """Get total number of entries."""
        if self._lazy_loading:
            total_entries = len(self.index['entries'])
            if self.max_entries is not None:
                return min(total_entries, self.max_entries)
            return total_entries
        else:
            total_entries = len(self.entries)
            if self.max_entries is not None:
                return min(total_entries, self.max_entries)
            return total_entries
    
    def __getitem__(self, idx: int) -> Tuple[GraphBatch, Dict[str, torch.Tensor]]:
        """Get single item."""
        entry = self.get_entry(idx)
        graph_batch = self.to_graphbatch(entry)
        
        targets = {
            'energy': torch.tensor(entry.energy, dtype=torch.float32),
            'forces': torch.tensor(entry.forces, dtype=torch.float32),
            'stress': torch.tensor(entry.stress, dtype=torch.float32),
            'magmoms': torch.tensor(entry.magmoms, dtype=torch.float32),
            'structure_id': entry.structure_id,
            'step': entry.step,
            'metadata': entry.metadata
        }
        
        return graph_batch, targets


def create_mptrj_splits(dataset: MPtrjDataset, 
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1,
                       split_by: str = "structure") -> None:
    """Create train/val/test splits for MPtrj dataset.
    
    Args:
        dataset: MPtrjDataset instance
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        split_by: How to split ("structure" or "step")
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    n_entries = len(dataset.entries)
    indices = list(range(n_entries))
    
    if split_by == "structure":
        # Split by unique structures
        unique_structures = list(set(entry.structure_id for entry in dataset.entries))
        # Deterministic shuffle based on structure count using Fisher-Yates
        data_hash = hash(len(unique_structures)) % 1000000
        for i in range(len(unique_structures) - 1, 0, -1):
            j = (data_hash + i) % (i + 1)
            unique_structures[i], unique_structures[j] = unique_structures[j], unique_structures[i]
        
        n_train = int(len(unique_structures) * train_ratio)
        n_val = int(len(unique_structures) * val_ratio)
        
        train_structures = unique_structures[:n_train]
        val_structures = unique_structures[n_train:n_train + n_val]
        test_structures = unique_structures[n_train + n_val:]
        
        # Map back to entry indices
        train_indices = [i for i, entry in enumerate(dataset.entries) if entry.structure_id in train_structures]
        val_indices = [i for i, entry in enumerate(dataset.entries) if entry.structure_id in val_structures]
        test_indices = [i for i, entry in enumerate(dataset.entries) if entry.structure_id in test_structures]
        
    else:  # split_by == "step"
        # Split entries deterministically using Fisher-Yates
        data_hash = hash(len(indices)) % 1000000
        for i in range(len(indices) - 1, 0, -1):
            j = (data_hash + i) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        
        n_train = int(n_entries * train_ratio)
        n_val = int(n_entries * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    
    # Update index
    dataset.index['splits'] = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Save updated index
    with open(dataset.index_path, 'w') as f:
        json.dump(dataset.index, f, indent=2)
    
    logger.info(f"Created splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


# Utility functions for working with MPtrj data

def load_mptrj_sample(data_path: Union[str, Path], 
                     n_samples: int = 5) -> List[MPtrjEntry]:
    """Load a small sample of MPtrj data for testing."""
    dataset = MPtrjDataset(data_path, max_atoms=50)  # Limit for testing
    
    if len(dataset) == 0:
        raise ValueError("No data loaded from MPtrj file")
    
    # Return first n_samples
    return dataset.entries[:min(n_samples, len(dataset))]


def mptrj_to_ase_trajectory(dataset: MPtrjDataset, 
                           structure_id: str) -> Optional[ase.io.Trajectory]:
    """Convert MPtrj trajectory to ASE trajectory format."""
    structure_entries = [entry for entry in dataset.entries if entry.structure_id == structure_id]
    
    if not structure_entries:
        return None
    
    # Sort by step
    structure_entries.sort(key=lambda x: x.step)
    
    # Create ASE trajectory
    trajectory = ase.io.Trajectory(f"{structure_id}_mptrj.traj", 'w')
    
    for entry in structure_entries:
        # Add properties as arrays
        entry.atoms.arrays['forces'] = entry.forces
        entry.atoms.arrays['magmoms'] = entry.magmoms
        entry.atoms.info['energy'] = entry.energy
        entry.atoms.info['stress'] = entry.stress
        entry.atoms.info['step'] = entry.step
        
        trajectory.write(entry.atoms)
    
    trajectory.close()
    return trajectory


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python mptrj.py <path_to_mptrj.json>")
        sys.exit(1)
    
    data_path = Path(sys.argv[1])
    
    try:
        # Load dataset
        dataset = MPtrjDataset(data_path)
        print(f"Loaded {len(dataset)} trajectory entries")
        
        # Print statistics
        stats = dataset.get_statistics()
        print(f"Dataset statistics:")
        print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV")
        print(f"  Atoms range: {stats['n_atoms']['min']} to {stats['n_atoms']['max']}")
        print(f"  Unique structures: {stats['n_unique_structures']}")
        
        # Create splits
        create_mptrj_splits(dataset)
        split_sizes = dataset.get_split_sizes()
        print(f"Split sizes: {split_sizes}")
        
        # Test batch iteration
        print("\nTesting batch iteration:")
        for i, (graph_batch, targets) in enumerate(dataset.iter_batches(split="train", batch_size=2)):
            print(f"  Batch {i}: {graph_batch.positions.shape[0]} atoms, {targets['energy'].shape[0]} structures")
            if i >= 2:  # Limit output
                break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
