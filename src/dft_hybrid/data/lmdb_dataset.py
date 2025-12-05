"""LMDB dataset loader for efficient MPtrj data access."""

import os
import lmdb
import msgpack
import lz4.frame
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import random
from torch_geometric.data import Data, Batch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from dft_hybrid.data.io import set_safe_mp
from graphs.periodic_graph import PeriodicGraph


class LMDBDataset(Dataset):
    """Efficient LMDB dataset for MPtrj data."""
    
    def __init__(
        self,
        lmdb_dir: str,
        cutoff_radius: float = 6.0,
        max_atoms: Optional[int] = None,
        shuffle: bool = True
    ):
        """Initialize LMDB dataset.
        
        Args:
            lmdb_dir: Directory containing LMDB shards
            cutoff_radius: Cutoff radius for graph construction
            max_atoms: Maximum number of atoms per structure
            shuffle: Whether to shuffle the dataset
        """
        self.lmdb_dir = Path(lmdb_dir)
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.shuffle = shuffle
        
        # Find all LMDB shards
        self.shard_paths = sorted(list(self.lmdb_dir.glob("train_*.lmdb")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No LMDB shards found in {lmdb_dir}")
        
        print(f"Found {len(self.shard_paths)} LMDB shards")
        
        # Count total entries across all shards
        self.total_entries = 0
        self.shard_entry_counts = []
        
        for shard_path in self.shard_paths:
            with lmdb.open(str(shard_path), readonly=True) as env:
                with env.begin() as txn:
                    count = txn.stat()['entries']
                    self.shard_entry_counts.append(count)
                    self.total_entries += count
        
        print(f"Total entries: {self.total_entries}")
        
        # Create mapping from global index to (shard_idx, key)
        self.index_mapping = []
        for shard_idx, shard_path in enumerate(self.shard_paths):
            env = lmdb.open(str(shard_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                # Get all actual keys from the shard
                cursor = txn.cursor()
                for key, _ in cursor:
                    self.index_mapping.append((shard_idx, key.decode()))
            env.close()
        
        if shuffle:
            random.shuffle(self.index_mapping)
        
        # Initialize graph builder
        self.graph_builder = PeriodicGraph(cutoff_radius=cutoff_radius)
    
    def __len__(self) -> int:
        """Return total number of entries."""
        return self.total_entries
    
    def __getitem__(self, idx: int) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """Get a single entry by index."""
        shard_idx, key = self.index_mapping[idx]
        
        # Load from LMDB shard
        shard_path = self.shard_paths[shard_idx]
        with lmdb.open(str(shard_path), readonly=True) as env:
            with env.begin() as txn:
                # Use the actual key
                key_bytes = key.encode()
                
                # Get compressed data
                compressed_data = txn.get(key_bytes)
                if compressed_data is None:
                    raise KeyError(f"Key {key} not found in shard {shard_idx}")
                
                # Decompress and deserialize
                msgpack_data = lz4.frame.decompress(compressed_data)
                entry = msgpack.unpackb(msgpack_data, raw=False)
        
        # Convert to graph
        graph = self._entry_to_graph(entry)
        
        # Create targets
        targets = {
            'energy': torch.tensor(entry['energy'], dtype=torch.float32),
            'forces': torch.tensor(entry['forces'], dtype=torch.float32),
            'stress': torch.tensor(entry['stress'], dtype=torch.float32),
            'magmoms': torch.tensor(entry['magmom'], dtype=torch.float32)
        }
        
        return graph, targets
    
    def _entry_to_graph(self, entry: Dict[str, Any]) -> Data:
        """Convert LMDB entry to PyTorch Geometric Data."""
        # Extract data
        atomic_numbers = np.array(entry['Z'], dtype=np.int64)
        positions = np.array(entry['pos'], dtype=np.float32)
        cell = np.array(entry['cell'], dtype=np.float32)
        
        # Filter by max_atoms if specified
        if self.max_atoms is not None and len(atomic_numbers) > self.max_atoms:
            # Deterministically sample atoms if too many (keep first max_atoms)
            indices = np.arange(self.max_atoms)
            atomic_numbers = atomic_numbers[indices]
            positions = positions[indices]
        
        # Build graph
        graph = self.graph_builder.build_graph(positions, atomic_numbers, cell)
        
        # Add required attributes for MACE model
        graph.positions = graph.pos
        graph.edge_vectors = graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
        graph.cell_vectors = torch.tensor(cell, dtype=torch.float32).unsqueeze(0)
        graph.batch_indices = torch.zeros(len(atomic_numbers), dtype=torch.long)
        graph.cell_offsets = torch.zeros(graph.edge_index.shape[1], 3, dtype=torch.float32)
        graph.ptr = torch.tensor([0, len(atomic_numbers)], dtype=torch.long)
        
        return graph


def create_lmdb_dataloader(
    lmdb_dir: str,
    batch_size: int = 16,
    cutoff_radius: float = 6.0,
    max_atoms: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create DataLoader for LMDB dataset."""
    
    # Configure safe multiprocessing
    set_safe_mp()
    
    # Create dataset
    dataset = LMDBDataset(
        lmdb_dir=lmdb_dir,
        cutoff_radius=cutoff_radius,
        max_atoms=max_atoms,
        shuffle=shuffle
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=False
    )
    
    return dataloader


def collate_fn(batch: List[Tuple[Data, Dict[str, torch.Tensor]]]) -> Tuple[Data, Dict[str, torch.Tensor]]:
    """Collate function for batching graphs and targets."""
    graphs, targets_list = zip(*batch)
    
    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)
    
    # Batch targets
    batched_targets = {}
    for key in targets_list[0].keys():
        if key == 'energy':
            batched_targets[key] = torch.stack([t[key] for t in targets_list])
        elif key == 'stress':
            batched_targets[key] = torch.stack([t[key] for t in targets_list])  # Stack (3,3) tensors
        elif key in ['forces', 'magmoms']:
            batched_targets[key] = torch.cat([t[key] for t in targets_list], dim=0)
        else:
            batched_targets[key] = torch.cat([t[key] for t in targets_list])
    
    return batched_graph, batched_targets


if __name__ == "__main__":
    # Test the dataset
    dataset = LMDBDataset("data/mptrj_lmdb", max_atoms=50)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(3):
        graph, targets = dataset[i]
        print(f"Sample {i}: {len(graph.pos)} atoms, energy: {targets['energy'].item():.3f}")
