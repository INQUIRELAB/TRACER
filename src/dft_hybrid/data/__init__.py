"""Data loading and preprocessing module for the hybrid DFT→GNN→QNN pipeline."""

from .mptrj import (
    MPtrjDataset,
    MPtrjEntry,
    create_mptrj_splits,
    load_mptrj_sample,
    mptrj_to_ase_trajectory
)

from .ocp_lmdb import (
    OCPLMDBDataset,
    OCPEntry,
    create_ocp_splits,
    load_ocp_sample,
    ocp_to_ase_trajectory
)

from .mp_api import (
    MaterialsProjectDataset,
    MPEntry,
    create_mp_splits,
    fetch_common_materials,
    fetch_magnetic_materials
)

from .factory import (
    DatasetFactory,
    create_dataset_factory
)

__all__ = [
    # MPtrj dataset
    "MPtrjDataset",
    "MPtrjEntry", 
    "create_mptrj_splits",
    "load_mptrj_sample",
    "mptrj_to_ase_trajectory",
    
    # OCP dataset
    "OCPLMDBDataset",
    "OCPEntry",
    "create_ocp_splits", 
    "load_ocp_sample",
    "ocp_to_ase_trajectory",
    
    # Materials Project dataset
    "MaterialsProjectDataset",
    "MPEntry",
    "create_mp_splits",
    "fetch_common_materials",
    "fetch_magnetic_materials",
    
    # Dataset factory
    "DatasetFactory",
    "create_dataset_factory"
]
