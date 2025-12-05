"""Dataset loader for OC20/OC22 LMDB data via PyTorch Geometric-style iterators."""

import os
import lmdb
import pickle
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
class OCPEntry:
    """Single OCP LMDB entry."""
    
    structure_id: str
    atoms: Atoms
    energy: Optional[float] = None  # eV (for S2EF, IS2RE)
    forces: Optional[np.ndarray] = None  # (n_atoms, 3) eV/Å (for S2EF)
    stress: Optional[np.ndarray] = None  # (3, 3) eV/Å³ (for S2EF)
    initial_energy: Optional[float] = None  # eV (for IS2RE, IS2RS)
    relaxed_energy: Optional[float] = None  # eV (for IS2RE, IS2RS)
    initial_positions: Optional[np.ndarray] = None  # (n_atoms, 3) Å (for IS2RE, IS2RS)
    relaxed_positions: Optional[np.ndarray] = None  # (n_atoms, 3) Å (for IS2RE, IS2RS)
    metadata: Dict[str, Any] = None


class OCPLMDBDataset:
    """Dataset loader for OC20/OC22 LMDB data."""
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 dataset_type: str = "s2ef",
                 graph_builder: Optional[PeriodicGraph] = None,
                 cutoff_radius: float = 5.0,
                 max_atoms: Optional[int] = None,
                 use_relaxed: bool = True):
        """Initialize OCP LMDB dataset.
        
        Args:
            data_path: Path to OCP LMDB directory
            dataset_type: Type of dataset ("s2ef", "is2re", "is2rs")
            graph_builder: PeriodicGraph instance for building graphs
            cutoff_radius: Cutoff radius for neighbor construction
            max_atoms: Maximum number of atoms per structure (filtering)
            use_relaxed: Whether to use relaxed structures (for IS2RE/IS2RS)
        """
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type.lower()
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.use_relaxed = use_relaxed
        
        if graph_builder is None:
            self.graph_builder = PeriodicGraph(cutoff_radius=cutoff_radius)
        else:
            self.graph_builder = graph_builder
        
        # Validate dataset type
        if self.dataset_type not in ["s2ef", "is2re", "is2rs"]:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Initialize LMDB environment
        self._init_lmdb()
        
        # Create index files if they don't exist
        self.index_path = self.data_path / "index.json"
        self._ensure_index()
    
    def _init_lmdb(self):
        """Initialize LMDB environment."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"OCP LMDB directory not found: {self.data_path}")
        
        # Check for required LMDB files
        required_files = ["data.mdb", "lock.mdb"]
        for file in required_files:
            if not (self.data_path / file).exists():
                raise FileNotFoundError(f"Required LMDB file not found: {self.data_path / file}")
        
        # Open LMDB environment
        self.env = lmdb.open(
            str(self.data_path),
            readonly=True,
            readahead=False,
            meminit=False,
            map_size=1099511627776 * 2  # 2TB
        )
        
        logger.info(f"Opened LMDB environment: {self.data_path}")
        
        # Get total number of entries
        with self.env.begin() as txn:
            self.total_entries = txn.stat()['entries']
        
        logger.info(f"LMDB contains {self.total_entries} entries")
    
    def _ensure_index(self):
        """Ensure index file exists for efficient data access."""
        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            import json
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            logger.info("Creating new index file")
            self._create_index()
    
    def _create_index(self):
        """Create index file for efficient data access."""
        logger.info("Scanning LMDB to create index...")
        
        self.index = {
            'entries': [],
            'splits': {
                'train': [],
                'val': [],
                'test': []
            },
            'statistics': {}
        }
        
        entry_count = 0
        skipped_count = 0
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    entry_data = pickle.loads(value)
                    entry_info = self._parse_entry_info(entry_data, entry_count)
                    
                    if entry_info is not None:
                        self.index['entries'].append(entry_info)
                        entry_count += 1
                    else:
                        skipped_count += 1
                        
                    if entry_count % 10000 == 0:
                        logger.info(f"Processed {entry_count} entries...")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse entry {entry_count}: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(f"Indexed {entry_count} entries, skipped {skipped_count}")
        
        # Compute statistics
        self.index['statistics'] = self._compute_statistics()
        
        # Save index
        import json
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        
        logger.info(f"Created index file: {self.index_path}")
    
    def _parse_entry_info(self, entry_data: Dict, entry_id: int) -> Optional[Dict]:
        """Parse entry data to extract basic information."""
        try:
            # Extract structure information
            if 'cell' in entry_data:
                cell = np.array(entry_data['cell'])
            else:
                cell = np.eye(3) * 10.0  # Default cell
            
            if 'pos' in entry_data:
                positions = np.array(entry_data['pos'])
            else:
                return None
            
            if 'atomic_numbers' in entry_data:
                atomic_numbers = np.array(entry_data['atomic_numbers'])
            elif 'z' in entry_data:
                atomic_numbers = np.array(entry_data['z'])
            else:
                return None
            
            n_atoms = len(atomic_numbers)
            
            # Filter by max_atoms if specified
            if self.max_atoms is not None and n_atoms > self.max_atoms:
                return None
            
            # Extract energy information based on dataset type
            energy = None
            if self.dataset_type == "s2ef" and 'y' in entry_data:
                energy = float(entry_data['y'])
            elif self.dataset_type in ["is2re", "is2rs"]:
                if 'y_relaxed' in entry_data:
                    energy = float(entry_data['y_relaxed'])
                elif 'y' in entry_data:
                    energy = float(entry_data['y'])
            
            # Extract volume
            volume = np.linalg.det(cell)
            
            # Extract metadata
            metadata = {
                'entry_id': entry_id,
                'n_atoms': n_atoms,
                'volume': volume,
                'density': volume / n_atoms if n_atoms > 0 else 0.0,
                'dataset_type': self.dataset_type
            }
            
            # Add dataset-specific metadata
            if 'tags' in entry_data:
                metadata['tags'] = entry_data['tags']
            if 'sid' in entry_data:
                metadata['sid'] = entry_data['sid']
            
            return {
                'entry_id': entry_id,
                'n_atoms': n_atoms,
                'energy': energy,
                'volume': volume,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse entry info: {e}")
            return None
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.index['entries']:
            return {}
        
        energies = [entry['energy'] for entry in self.index['entries'] if entry['energy'] is not None]
        n_atoms_list = [entry['n_atoms'] for entry in self.index['entries']]
        volumes = [entry['volume'] for entry in self.index['entries']]
        
        stats = {
            'n_entries': len(self.index['entries']),
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
            }
        }
        
        if energies:
            stats['energy'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            }
        
        return stats
    
    def get_entry(self, idx: int) -> OCPEntry:
        """Get a single entry by index."""
        if idx >= len(self.index['entries']):
            raise IndexError(f"Index {idx} out of range for {len(self.index['entries'])} entries")
        
        entry_info = self.index['entries'][idx]
        entry_id = entry_info['entry_id']
        
        # Load full entry data from LMDB
        with self.env.begin() as txn:
            key = f"{entry_id}".encode()
            value = txn.get(key)
            
            if value is None:
                raise KeyError(f"Entry {entry_id} not found in LMDB")
            
            entry_data = pickle.loads(value)
        
        return self._parse_entry(entry_data, entry_info)
    
    def _parse_entry(self, entry_data: Dict, entry_info: Dict) -> OCPEntry:
        """Parse full entry data."""
        # Extract structure information
        cell = np.array(entry_data.get('cell', np.eye(3) * 10.0))
        positions = np.array(entry_data['pos'])
        atomic_numbers = np.array(entry_data.get('atomic_numbers', entry_data.get('z')))
        
        # Create ASE Atoms object
        atoms = Atoms(
            symbols=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=True
        )
        
        # Extract properties based on dataset type
        energy = None
        forces = None
        stress = None
        initial_energy = None
        relaxed_energy = None
        initial_positions = None
        relaxed_positions = None
        
        if self.dataset_type == "s2ef":
            # Structure to Energy and Forces
            if 'y' in entry_data:
                energy = float(entry_data['y'])
            if 'force' in entry_data:
                forces = np.array(entry_data['force'])
            if 'stress' in entry_data:
                stress = np.array(entry_data['stress'])
        
        elif self.dataset_type in ["is2re", "is2rs"]:
            # Initial Structure to Relaxed Energy / Structure
            if 'y' in entry_data:
                initial_energy = float(entry_data['y'])
            if 'y_relaxed' in entry_data:
                relaxed_energy = float(entry_data['y_relaxed'])
            
            # For IS2RS, we might have relaxed positions
            if self.use_relaxed and 'pos_relaxed' in entry_data:
                relaxed_positions = np.array(entry_data['pos_relaxed'])
                # Update atoms with relaxed positions
                atoms.set_positions(relaxed_positions)
        
        # Extract metadata
        metadata = entry_info['metadata'].copy()
        metadata.update({
            'dataset_type': self.dataset_type,
            'entry_data_keys': list(entry_data.keys())
        })
        
        return OCPEntry(
            structure_id=f"ocp_{entry_info['entry_id']}",
            atoms=atoms,
            energy=energy,
            forces=forces,
            stress=stress,
            initial_energy=initial_energy,
            relaxed_energy=relaxed_energy,
            initial_positions=initial_positions,
            relaxed_positions=relaxed_positions,
            metadata=metadata
        )
    
    def to_graphbatch(self, entry: OCPEntry) -> GraphBatch:
        """Convert OCP entry to GraphBatch."""
        return self.graph_builder.build_batch([entry.atoms])
    
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
            split_indices = list(range(len(self.index['entries'])))
        
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
            batch_entries = []
            for idx in batch_indices:
                try:
                    entry = self.get_entry(idx)
                    batch_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load entry {idx}: {e}")
                    continue
            
            if not batch_entries:
                continue
            
            # Convert to GraphBatch
            batch_atoms = [entry.atoms for entry in batch_entries]
            graph_batch = self.graph_builder.build_batch(batch_atoms)
            
            # Prepare targets based on dataset type
            targets = {
                'structure_ids': [entry.structure_id for entry in batch_entries],
                'metadata': [entry.metadata for entry in batch_entries]
            }
            
            if self.dataset_type == "s2ef":
                # Structure to Energy and Forces
                if all(entry.energy is not None for entry in batch_entries):
                    targets['energy'] = torch.tensor([entry.energy for entry in batch_entries], dtype=torch.float32)
                
                if all(entry.forces is not None for entry in batch_entries):
                    targets['forces'] = torch.cat([torch.tensor(entry.forces, dtype=torch.float32) for entry in batch_entries], dim=0)
                
                if all(entry.stress is not None for entry in batch_entries):
                    targets['stress'] = torch.stack([torch.tensor(entry.stress, dtype=torch.float32) for entry in batch_entries])
            
            elif self.dataset_type in ["is2re", "is2rs"]:
                # Initial Structure to Relaxed Energy/Structure
                if all(entry.initial_energy is not None for entry in batch_entries):
                    targets['initial_energy'] = torch.tensor([entry.initial_energy for entry in batch_entries], dtype=torch.float32)
                
                if all(entry.relaxed_energy is not None for entry in batch_entries):
                    targets['relaxed_energy'] = torch.tensor([entry.relaxed_energy for entry in batch_entries], dtype=torch.float32)
                
                # For IS2RS, we might want to predict relaxed positions
                if self.dataset_type == "is2rs" and all(entry.relaxed_positions is not None for entry in batch_entries):
                    targets['relaxed_positions'] = torch.cat([torch.tensor(entry.relaxed_positions, dtype=torch.float32) for entry in batch_entries], dim=0)
            
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
        return len(self.index['entries'])
    
    def __getitem__(self, idx: int) -> Tuple[GraphBatch, Dict[str, torch.Tensor]]:
        """Get single item."""
        entry = self.get_entry(idx)
        graph_batch = self.to_graphbatch(entry)
        
        targets = {
            'structure_id': entry.structure_id,
            'metadata': entry.metadata
        }
        
        if entry.energy is not None:
            targets['energy'] = torch.tensor(entry.energy, dtype=torch.float32)
        if entry.forces is not None:
            targets['forces'] = torch.tensor(entry.forces, dtype=torch.float32)
        if entry.stress is not None:
            targets['stress'] = torch.tensor(entry.stress, dtype=torch.float32)
        if entry.initial_energy is not None:
            targets['initial_energy'] = torch.tensor(entry.initial_energy, dtype=torch.float32)
        if entry.relaxed_energy is not None:
            targets['relaxed_energy'] = torch.tensor(entry.relaxed_energy, dtype=torch.float32)
        
        return graph_batch, targets
    
    def __del__(self):
        """Close LMDB environment."""
        if hasattr(self, 'env'):
            self.env.close()


def create_ocp_splits(dataset: OCPLMDBDataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> None:
    """Create train/val/test splits for OCP dataset.
    
    Args:
        dataset: OCPLMDBDataset instance
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    n_entries = len(dataset.index['entries'])
    indices = list(range(n_entries))
    # Deterministic shuffle based on dataset size using Fisher-Yates
    data_hash = hash(n_entries) % 1000000
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
    import json
    with open(dataset.index_path, 'w') as f:
        json.dump(dataset.index, f, indent=2)
    
    logger.info(f"Created splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


# Utility functions for working with OCP data

def load_ocp_sample(data_path: Union[str, Path], 
                   dataset_type: str = "s2ef",
                   n_samples: int = 5) -> List[OCPEntry]:
    """Load a small sample of OCP data for testing."""
    dataset = OCPLMDBDataset(data_path, dataset_type, max_atoms=50)  # Limit for testing
    
    if len(dataset) == 0:
        raise ValueError("No data loaded from OCP LMDB")
    
    # Return first n_samples
    entries = []
    for i in range(min(n_samples, len(dataset))):
        try:
            entry = dataset.get_entry(i)
            entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to load entry {i}: {e}")
            continue
    
    return entries


def ocp_to_ase_trajectory(dataset: OCPLMDBDataset, 
                         structure_id: str) -> Optional[ase.io.Trajectory]:
    """Convert OCP entry to ASE trajectory format."""
    try:
        # Find entry by structure_id
        entry_idx = None
        for i, entry_info in enumerate(dataset.index['entries']):
            if f"ocp_{entry_info['entry_id']}" == structure_id:
                entry_idx = i
                break
        
        if entry_idx is None:
            return None
        
        entry = dataset.get_entry(entry_idx)
        
        # Create ASE trajectory
        trajectory = ase.io.Trajectory(f"{structure_id}_ocp.traj", 'w')
        
        # Add properties as arrays/info
        if entry.forces is not None:
            entry.atoms.arrays['forces'] = entry.forces
        if entry.energy is not None:
            entry.atoms.info['energy'] = entry.energy
        if entry.stress is not None:
            entry.atoms.info['stress'] = entry.stress
        
        trajectory.write(entry.atoms)
        trajectory.close()
        
        return trajectory
        
    except Exception as e:
        logger.error(f"Failed to create trajectory: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python ocp_lmdb.py <path_to_ocp_lmdb> [dataset_type]")
        print("Dataset types: s2ef, is2re, is2rs")
        sys.exit(1)
    
    data_path = Path(sys.argv[1])
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else "s2ef"
    
    try:
        # Load dataset
        dataset = OCPLMDBDataset(data_path, dataset_type)
        print(f"Loaded {len(dataset)} entries from {dataset_type} dataset")
        
        # Print statistics
        stats = dataset.get_statistics()
        print(f"Dataset statistics:")
        print(f"  Entries: {stats['n_entries']}")
        print(f"  Atoms range: {stats['n_atoms']['min']} to {stats['n_atoms']['max']}")
        if 'energy' in stats:
            print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV")
        
        # Create splits
        create_ocp_splits(dataset)
        split_sizes = dataset.get_split_sizes()
        print(f"Split sizes: {split_sizes}")
        
        # Test batch iteration
        print(f"\nTesting batch iteration:")
        for i, (graph_batch, targets) in enumerate(dataset.iter_batches(split="train", batch_size=2)):
            print(f"  Batch {i}: {graph_batch.positions.shape[0]} atoms, {len(targets.get('structure_ids', []))} structures")
            print(f"    Available targets: {list(targets.keys())}")
            if i >= 2:  # Limit output
                break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
