"""Dataset factory for dynamic loader selection."""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from .mptrj import MPtrjDataset, create_mptrj_splits
from .ocp_lmdb import OCPLMDBDataset, create_ocp_splits
from .mp_api import MaterialsProjectDataset, create_mp_splits
from graphs.periodic_graph import PeriodicGraph


class DatasetFactory:
    """Factory class for creating dataset loaders based on configuration."""
    
    def __init__(self, config: DictConfig):
        """Initialize the dataset factory with configuration.
        
        Args:
            config: Hydra configuration containing dataset settings
        """
        self.config = config
        self.dataset_config = config.dataset
        self.root_path = Path(self.dataset_config.root)
        
        # Initialize graph builder
        self.graph_builder = PeriodicGraph(
            cutoff_radius=self.dataset_config.get('cutoff', 6.0)
        )
    
    def create_dataset(self, split: str = None) -> Union[MPtrjDataset, OCPLMDBDataset, MaterialsProjectDataset]:
        """Create a dataset instance based on the configuration.
        
        Args:
            split: Dataset split (train/val/test). If None, uses config.dataset.split
            
        Returns:
            Dataset instance
        """
        if split is None:
            split = self.dataset_config.split
            
        dataset_name = self.dataset_config.name
        
        if dataset_name == "mptrj":
            return self._create_mptrj_dataset(split)
        elif dataset_name == "oc20_s2ef":
            return self._create_oc20_dataset(split)
        elif dataset_name == "oc22":
            return self._create_oc22_dataset(split)
        elif dataset_name == "mp_props":
            return self._create_mp_dataset(split)
        elif dataset_name in ["spice", "ani", "rmd17"]:
            # TODO: Implement these datasets
            raise NotImplementedError(f"Dataset {dataset_name} not yet implemented")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def create_dataloader(self, split: str = None, shuffle: bool = None) -> DataLoader:
        """Create a DataLoader for the specified dataset.
        
        Args:
            split: Dataset split (train/val/test). If None, uses config.dataset.split
            shuffle: Whether to shuffle data. If None, shuffles for train split only
            
        Returns:
            DataLoader instance
        """
        if split is None:
            split = self.dataset_config.split
            
        if shuffle is None:
            shuffle = (split == "train")
            
        dataset = self.create_dataset(split)
        
        return DataLoader(
            dataset,
            batch_size=self.dataset_config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Disable multiprocessing to save memory
            pin_memory=False,  # Disable pin_memory to save memory
            collate_fn=self._collate_fn,
            persistent_workers=False  # Don't keep workers alive
        )
    
    def _create_mptrj_dataset(self, split: str) -> MPtrjDataset:
        """Create MPtrj dataset."""
        data_path = self.root_path / "mptrj" / "mptrj.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"MPtrj data not found at {data_path}")
        
        # MPtrjDataset doesn't use split parameter - it loads all data and 
        # splits are handled by create_mptrj_splits function
        return MPtrjDataset(
            data_path=data_path,
            graph_builder=self.graph_builder,
            cutoff_radius=self.dataset_config.get('cutoff', 6.0)
        )
    
    def _create_oc20_dataset(self, split: str) -> OCPLMDBDataset:
        """Create OC20 dataset."""
        data_path = self.root_path / "oc20"
        
        if not data_path.exists():
            raise FileNotFoundError(f"OC20 data not found at {data_path}")
        
        task = self.config.get('ocp', {}).get('task', 's2ef')
        split_size = self.config.get('ocp', {}).get('split_size', '200k')
        
        return OCPLMDBDataset(
            data_path=data_path,
            graph_builder=self.graph_builder,
            cutoff_radius=self.dataset_config.get('cutoff', 6.0)
        )
    
    def _create_oc22_dataset(self, split: str) -> OCPLMDBDataset:
        """Create OC22 dataset."""
        data_path = self.root_path / "oc22"
        
        if not data_path.exists():
            raise FileNotFoundError(f"OC22 data not found at {data_path}")
        
        return OCPLMDBDataset(
            data_path=data_path,
            graph_builder=self.graph_builder,
            cutoff_radius=self.dataset_config.get('cutoff', 6.0)
        )
    
    def _create_mp_dataset(self, split: str) -> MaterialsProjectDataset:
        """Create Materials Project dataset."""
        data_path = self.root_path / "mp"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Materials Project data not found at {data_path}")
        
        api_key_env = self.config.get('mp_api', {}).get('api_key_env', 'MP_API_KEY')
        
        return MaterialsProjectDataset(
            data_path=data_path,
            graph_builder=self.graph_builder,
            cutoff_radius=self.dataset_config.get('cutoff', 6.0),
            api_key_env=api_key_env
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching GraphBatch objects."""
        from graphs.periodic_graph import GraphBatch
        
        # If batch contains GraphBatch objects, they should already be batched
        if isinstance(batch[0], GraphBatch):
            return batch[0]  # Return the single GraphBatch
        
        # Otherwise, batch individual items
        return batch
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the configured dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        dataset_name = self.dataset_config.name
        
        info = {
            "name": dataset_name,
            "root": str(self.root_path),
            "split": self.dataset_config.split,
            "batch_size": self.dataset_config.batch_size,
            "cutoff": self.dataset_config.get('cutoff', 6.0),
            "features": self.dataset_config.get('features', ['distances', 'angles', 'frac_coords'])
        }
        
        if dataset_name in ["oc20_s2ef", "oc22"]:
            info.update({
                "task": self.config.get('ocp', {}).get('task', 's2ef'),
                "split_size": self.config.get('ocp', {}).get('split_size', '200k')
            })
        elif dataset_name == "mp_props":
            info.update({
                "use_charge_density": self.config.get('mp_api', {}).get('use_charge_density', True),
                "api_key_env": self.config.get('mp_api', {}).get('api_key_env', 'MP_API_KEY')
            })
        
        return info


def create_dataset_factory(config: DictConfig) -> DatasetFactory:
    """Create a dataset factory from configuration.
    
    Args:
        config: Hydra configuration
        
    Returns:
        DatasetFactory instance
    """
    return DatasetFactory(config)
