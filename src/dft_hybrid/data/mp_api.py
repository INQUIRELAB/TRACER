"""Wrapper around Materials Project API to fetch property tables and optional charge densities."""

import os
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

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    logger.warning("Materials Project API not available. Install with: pip install mp-api")


@dataclass
class MPEntry:
    """Single Materials Project entry."""
    
    structure_id: str  # Materials ID
    structure: Structure
    energy: Optional[float] = None  # Formation energy per atom (eV/atom)
    band_gap: Optional[float] = None  # Band gap (eV)
    density: Optional[float] = None  # Density (g/cm³)
    volume: Optional[float] = None  # Volume per atom (Å³/atom)
    magnetic_ordering: Optional[str] = None  # Magnetic ordering
    is_magnetic: Optional[bool] = None  # Whether structure is magnetic
    is_stable: Optional[bool] = None  # Whether structure is stable
    is_metal: Optional[bool] = None  # Whether structure is metallic
    charge_density: Optional[np.ndarray] = None  # Charge density data
    metadata: Dict[str, Any] = None


class MaterialsProjectDataset:
    """Dataset loader for Materials Project data via API."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 graph_builder: Optional[PeriodicGraph] = None,
                 cutoff_radius: float = 5.0,
                 max_atoms: Optional[int] = None,
                 include_charge_density: bool = False,
                 cache_dir: Optional[Union[str, Path]] = None):
        """Initialize Materials Project dataset.
        
        Args:
            api_key: Materials Project API key (or set MP_API_KEY env var)
            graph_builder: PeriodicGraph instance for building graphs
            cutoff_radius: Cutoff radius for neighbor construction
            max_atoms: Maximum number of atoms per structure (filtering)
            include_charge_density: Whether to fetch charge density data
            cache_dir: Directory to cache downloaded data
        """
        if not MP_API_AVAILABLE:
            raise ImportError("Materials Project API not available. Install with: pip install mp-api")
        
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.include_charge_density = include_charge_density
        
        # Setup API key
        if api_key is None:
            api_key = os.getenv('MP_API_KEY')
            if api_key is None:
                raise ValueError("Materials Project API key required. Set MP_API_KEY env var or pass api_key parameter")
        
        self.api_key = api_key
        
        # Setup graph builder
        if graph_builder is None:
            self.graph_builder = PeriodicGraph(cutoff_radius=cutoff_radius)
        else:
            self.graph_builder = graph_builder
        
        # Setup caching
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "materials_project"
        else:
            cache_dir = Path(cache_dir)
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API client
        self.mpr = MPRester(api_key)
        
        # Create index files if they don't exist
        self.index_path = self.cache_dir / "mp_index.json"
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure index file exists for efficient data access."""
        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            logger.info("Creating new index file")
            self.index = {
                'entries': [],
                'splits': {
                    'train': [],
                    'val': [],
                    'test': []
                },
                'statistics': {}
            }
    
    def fetch_structures(self, 
                        criteria: Dict[str, Any],
                        properties: List[str] = None,
                        max_entries: Optional[int] = None) -> List[MPEntry]:
        """Fetch structures from Materials Project.
        
        Args:
            criteria: Search criteria (e.g., {"elements": ["Fe", "O"], "nelements": 2})
            properties: List of properties to fetch
            max_entries: Maximum number of entries to fetch
            
        Returns:
            List of MPEntry objects
        """
        logger.info(f"Fetching structures with criteria: {criteria}")
        
        # Default properties (using valid field names)
        if properties is None:
            properties = [
                "structure",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "volume",
                "ordering",  # Changed from magnetic_ordering to ordering
                "is_magnetic",
                "is_stable",
                "is_metal"
            ]
        
        # Add charge density if requested
        if self.include_charge_density and "charge_density" not in properties:
            properties.append("charge_density")
        
        # Fetch data from API (using updated method)
        docs = self.mpr.materials.summary.search(
            **criteria,
            fields=properties,
            chunk_size=1000
        )
        
        entries = []
        for i, doc in enumerate(docs):
            try:
                entry = self._parse_document(doc, i)
                if entry is not None:
                    entries.append(entry)
                    
                    # Filter by max_atoms if specified
                    if self.max_atoms is not None and len(entry.structure) > self.max_atoms:
                        continue
                    
                    entries.append(entry)
                    
            except Exception as e:
                logger.warning(f"Failed to parse document {i}: {e}")
                continue
        
        logger.info(f"Fetched {len(entries)} structures")
        return entries
    
    def _parse_document(self, doc: Any, doc_id: int) -> Optional[MPEntry]:
        """Parse Materials Project document."""
        try:
            # Extract structure
            if not hasattr(doc, 'structure') or doc.structure is None:
                return None
            
            structure = doc.structure
            
            # Extract properties
            energy = getattr(doc, 'formation_energy_per_atom', None)
            band_gap = getattr(doc, 'band_gap', None)
            density = getattr(doc, 'density', None)
            volume = getattr(doc, 'volume', None)
            magnetic_ordering = getattr(doc, 'ordering', None)
            is_magnetic = getattr(doc, 'is_magnetic', None)
            is_stable = getattr(doc, 'is_stable', None)
            is_metal = getattr(doc, 'is_metal', None)
            
            # Extract charge density if available
            charge_density = None
            if self.include_charge_density and hasattr(doc, 'charge_density'):
                try:
                    charge_density = np.array(doc.charge_density)
                except Exception as e:
                    logger.warning(f"Failed to parse charge density: {e}")
            
            # Extract metadata (avoiding non-serializable objects)
            metadata = {
                'doc_id': doc_id,
                'material_id': getattr(doc, 'material_id', f'mp_{doc_id}'),
                'elements': [str(el) for el in structure.composition.elements],  # Convert to strings
                'formula': structure.composition.reduced_formula,
                'n_atoms': len(structure),
                'spacegroup': getattr(structure, 'get_space_group_info', lambda: (None, None))()[1] if hasattr(structure, 'get_space_group_info') else None,
                'volume_per_atom': structure.volume / len(structure),
                'density': structure.density,
                'is_ordered': getattr(structure, 'is_ordered', True),
                'is_valid': getattr(structure, 'is_valid', True)
            }
            
            return MPEntry(
                structure_id=getattr(doc, 'material_id', f'mp_{doc_id}'),
                structure=structure,
                energy=energy,
                band_gap=band_gap,
                density=density,
                volume=volume,
                magnetic_ordering=magnetic_ordering,
                is_magnetic=is_magnetic,
                is_stable=is_stable,
                is_metal=is_metal,
                charge_density=charge_density,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            return None
    
    def add_entries(self, entries: List[MPEntry]) -> None:
        """Add entries to dataset and update index."""
        logger.info(f"Adding {len(entries)} entries to dataset")
        
        # Update index
        for entry in entries:
            entry_info = {
                'structure_id': entry.structure_id,
                'n_atoms': len(entry.structure),
                'energy': entry.energy,
                'band_gap': entry.band_gap,
                'density': entry.density,
                'volume': entry.volume,
                'is_magnetic': entry.is_magnetic,
                'is_stable': entry.is_stable,
                'is_metal': entry.is_metal,
                'metadata': entry.metadata
            }
            self.index['entries'].append(entry_info)
        
        # Cache entries
        self._cache_entries(entries)
        
        # Update statistics
        self.index['statistics'] = self._compute_statistics()
        
        # Save index
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
        
        logger.info(f"Added entries. Total: {len(self.index['entries'])}")
    
    def _cache_entries(self, entries: List[MPEntry]) -> None:
        """Cache entries to disk."""
        for entry in entries:
            cache_file = self.cache_dir / f"{entry.structure_id}.json"
            
            # Convert entry to serializable format
            entry_data = {
                'structure_id': entry.structure_id,
                'structure': entry.structure.as_dict(),
                'energy': entry.energy,
                'band_gap': entry.band_gap,
                'density': entry.density,
                'volume': entry.volume,
                'magnetic_ordering': entry.magnetic_ordering,
                'is_magnetic': entry.is_magnetic,
                'is_stable': entry.is_stable,
                'is_metal': entry.is_metal,
                'charge_density': entry.charge_density.tolist() if entry.charge_density is not None else None,
                'metadata': {
                    k: v for k, v in entry.metadata.items() 
                    if not (hasattr(v, '__class__') and 
                           ('Element' in str(v.__class__) or 
                            'method' in str(v.__class__) or
                            callable(v)))
                }  # Filter out non-serializable objects
            }
            
            with open(cache_file, 'w') as f:
                json.dump(entry_data, f, indent=2)
    
    def load_entry(self, structure_id: str) -> Optional[MPEntry]:
        """Load entry from cache."""
        cache_file = self.cache_dir / f"{structure_id}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                entry_data = json.load(f)
            
            # Reconstruct entry
            structure = Structure.from_dict(entry_data['structure'])
            charge_density = np.array(entry_data['charge_density']) if entry_data['charge_density'] is not None else None
            
            return MPEntry(
                structure_id=entry_data['structure_id'],
                structure=structure,
                energy=entry_data['energy'],
                band_gap=entry_data['band_gap'],
                density=entry_data['density'],
                volume=entry_data['volume'],
                magnetic_ordering=entry_data['magnetic_ordering'],
                is_magnetic=entry_data['is_magnetic'],
                is_stable=entry_data['is_stable'],
                is_metal=entry_data['is_metal'],
                charge_density=charge_density,
                metadata=entry_data['metadata']
            )
            
        except Exception as e:
            logger.error(f"Failed to load entry {structure_id}: {e}")
            return None
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.index['entries']:
            return {}
        
        energies = [entry['energy'] for entry in self.index['entries'] if entry['energy'] is not None]
        band_gaps = [entry['band_gap'] for entry in self.index['entries'] if entry['band_gap'] is not None]
        n_atoms_list = [entry['n_atoms'] for entry in self.index['entries']]
        densities = [entry['density'] for entry in self.index['entries'] if entry['density'] is not None]
        
        # Count properties
        n_magnetic = sum(1 for entry in self.index['entries'] if entry.get('is_magnetic', False))
        n_stable = sum(1 for entry in self.index['entries'] if entry.get('is_stable', False))
        n_metal = sum(1 for entry in self.index['entries'] if entry.get('is_metal', False))
        
        stats = {
            'n_entries': len(self.index['entries']),
            'n_atoms': {
                'mean': float(np.mean(n_atoms_list)),
                'std': float(np.std(n_atoms_list)),
                'min': int(np.min(n_atoms_list)),
                'max': int(np.max(n_atoms_list))
            },
            'n_magnetic': n_magnetic,
            'n_stable': n_stable,
            'n_metal': n_metal
        }
        
        if energies:
            stats['energy'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies))
            }
        
        if band_gaps:
            stats['band_gap'] = {
                'mean': float(np.mean(band_gaps)),
                'std': float(np.std(band_gaps)),
                'min': float(np.min(band_gaps)),
                'max': float(np.max(band_gaps))
            }
        
        if densities:
            stats['density'] = {
                'mean': float(np.mean(densities)),
                'std': float(np.std(densities)),
                'min': float(np.min(densities)),
                'max': float(np.max(densities))
            }
        
        return stats
    
    def get_entry(self, idx: int) -> MPEntry:
        """Get a single entry by index."""
        if idx >= len(self.index['entries']):
            raise IndexError(f"Index {idx} out of range for {len(self.index['entries'])} entries")
        
        entry_info = self.index['entries'][idx]
        structure_id = entry_info['structure_id']
        
        entry = self.load_entry(structure_id)
        if entry is None:
            raise ValueError(f"Failed to load entry {structure_id}")
        
        return entry
    
    def to_graphbatch(self, entry: MPEntry) -> GraphBatch:
        """Convert MP entry to GraphBatch."""
        atoms = AseAtomsAdaptor.get_atoms(entry.structure)
        return self.graph_builder.build_batch([atoms])
    
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
            batch_structures = [entry.structure for entry in batch_entries]
            batch_atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in batch_structures]
            graph_batch = self.graph_builder.build_batch(batch_atoms)
            
            # Prepare targets
            targets = {
                'structure_ids': [entry.structure_id for entry in batch_entries],
                'metadata': [entry.metadata for entry in batch_entries]
            }
            
            # Add available properties
            if all(entry.energy is not None for entry in batch_entries):
                targets['energy'] = torch.tensor([entry.energy for entry in batch_entries], dtype=torch.float32)
            
            if all(entry.band_gap is not None for entry in batch_entries):
                targets['band_gap'] = torch.tensor([entry.band_gap for entry in batch_entries], dtype=torch.float32)
            
            if all(entry.density is not None for entry in batch_entries):
                targets['density'] = torch.tensor([entry.density for entry in batch_entries], dtype=torch.float32)
            
            if all(entry.volume is not None for entry in batch_entries):
                targets['volume'] = torch.tensor([entry.volume for entry in batch_entries], dtype=torch.float32)
            
            # Add boolean properties
            targets['is_magnetic'] = torch.tensor([entry.is_magnetic for entry in batch_entries], dtype=torch.bool)
            targets['is_stable'] = torch.tensor([entry.is_stable for entry in batch_entries], dtype=torch.bool)
            targets['is_metal'] = torch.tensor([entry.is_metal for entry in batch_entries], dtype=torch.bool)
            
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
        if entry.band_gap is not None:
            targets['band_gap'] = torch.tensor(entry.band_gap, dtype=torch.float32)
        if entry.density is not None:
            targets['density'] = torch.tensor(entry.density, dtype=torch.float32)
        if entry.volume is not None:
            targets['volume'] = torch.tensor(entry.volume, dtype=torch.float32)
        
        targets['is_magnetic'] = torch.tensor(entry.is_magnetic, dtype=torch.bool)
        targets['is_stable'] = torch.tensor(entry.is_stable, dtype=torch.bool)
        targets['is_metal'] = torch.tensor(entry.is_metal, dtype=torch.bool)
        
        return graph_batch, targets


def create_mp_splits(dataset: MaterialsProjectDataset, 
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.1) -> None:
    """Create train/val/test splits for Materials Project dataset.
    
    Args:
        dataset: MaterialsProjectDataset instance
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
    with open(dataset.index_path, 'w') as f:
        json.dump(dataset.index, f, indent=2)
    
    logger.info(f"Created splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


# Utility functions for working with Materials Project data

def fetch_common_materials(api_key: Optional[str] = None,
                          max_entries: int = 1000) -> MaterialsProjectDataset:
    """Fetch common materials from Materials Project for testing."""
    dataset = MaterialsProjectDataset(api_key=api_key, max_atoms=100)
    
    # Fetch common binary and ternary compounds
    criteria_list = [
        {"nelements": 2, "is_stable": True},  # Binary compounds
        {"nelements": 3, "is_stable": True},  # Ternary compounds
        {"elements": ["Fe", "O"], "nelements": 2},  # Iron oxides
        {"elements": ["Ti", "O"], "nelements": 2},  # Titanium oxides
    ]
    
    all_entries = []
    for criteria in criteria_list:
        entries = dataset.fetch_structures(criteria, max_entries=max_entries // len(criteria_list))
        all_entries.extend(entries)
    
    dataset.add_entries(all_entries)
    return dataset


def fetch_magnetic_materials(api_key: Optional[str] = None,
                           max_entries: int = 500) -> MaterialsProjectDataset:
    """Fetch magnetic materials from Materials Project."""
    dataset = MaterialsProjectDataset(api_key=api_key, max_atoms=100)
    
    # Fetch magnetic materials
    criteria = {
        "is_magnetic": True,
        "is_stable": True,
        "nelements": {"$lte": 4}  # Limit complexity
    }
    
    entries = dataset.fetch_structures(criteria, max_entries=max_entries)
    dataset.add_entries(entries)
    
    return dataset


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    # Check for API key
    api_key = os.getenv('MP_API_KEY')
    if api_key is None:
        print("Error: MP_API_KEY environment variable not set")
        print("Get your API key from: https://materialsproject.org/api")
        sys.exit(1)
    
    try:
        # Create dataset
        dataset = MaterialsProjectDataset(api_key=api_key, max_atoms=50)
        
        # Fetch some common materials
        print("Fetching common materials...")
        entries = dataset.fetch_structures(
            criteria={"nelements": 2, "is_stable": True},
            max_entries=100
        )
        
        if entries:
            dataset.add_entries(entries)
            print(f"Added {len(entries)} entries to dataset")
            
            # Print statistics
            stats = dataset.get_statistics()
            print(f"Dataset statistics:")
            print(f"  Entries: {stats['n_entries']}")
            print(f"  Atoms range: {stats['n_atoms']['min']} to {stats['n_atoms']['max']}")
            if 'energy' in stats:
                print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV/atom")
            
            # Create splits
            create_mp_splits(dataset)
            split_sizes = dataset.get_split_sizes()
            print(f"Split sizes: {split_sizes}")
            
            # Test batch iteration
            print(f"\nTesting batch iteration:")
            for i, (graph_batch, targets) in enumerate(dataset.iter_batches(split="train", batch_size=2)):
                print(f"  Batch {i}: {graph_batch.positions.shape[0]} atoms, {len(targets.get('structure_ids', []))} structures")
                print(f"    Available targets: {list(targets.keys())}")
                if i >= 2:  # Limit output
                    break
        else:
            print("No entries fetched")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
