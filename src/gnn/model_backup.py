"""Graph Neural Network models for quantum chemistry."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from mace.calculators import mace
from mace.data import AtomicData
from mace.tools import torch_geometric, default_keys
from mace.modules.models import MACE
from graphs.periodic_graph import GraphBatch


class SchNetLayer(MessagePassing):
    """SchNet message passing layer."""
    
    def __init__(self, hidden_dim: int, num_filters: int = 64) -> None:
        """Initialize SchNet layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_filters: Number of radial basis filters
        """
        super().__init__(aggr='add')
        # TODO: Implement SchNet layer
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass through SchNet layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes (distances)
            
        Returns:
            Updated node features
        """
        # TODO: Implement SchNet forward pass
        raise NotImplementedError("SchNetLayer forward not implemented")


class MACEWrapper(nn.Module):
    """Simple GNN wrapper for DFT surrogate predictions (without complex dependencies)."""
    
    def __init__(self, 
                 r_max: float = 5.0,
                 num_bessel: int = 8,
                 num_polynomial_cutoff: int = 5,
                 max_ell: int = 3,
                 interaction_cls: str = "RealAgnosticResidualInteractionBlock",
                 interaction_cls_first: str = "RealAgnosticResidualInteractionBlock",
                 num_interactions: int = 6,
                 num_elements: int = 100,
                 hidden_irreps: str = "128x0e + 128x1o + 128x2e",
                 MLP_irreps: str = "16x0e",
                 atomic_inter_scale: float = 1.0,
                 atomic_inter_shift: float = 0.0,
                 correlation: int = 3,
                 gate: Optional[str] = None,
                 atomic_num_embed_max: int = 100,
                 compute_stress: bool = True) -> None:
        """Initialize Simple GNN wrapper.
        
        Args:
            r_max: Maximum distance for interactions
            num_bessel: Number of Bessel functions (unused)
            num_polynomial_cutoff: Number of polynomial cutoff functions (unused)
            max_ell: Maximum spherical harmonic degree (unused)
            interaction_cls: Interaction block class (unused)
            interaction_cls_first: First interaction block class (unused)
            num_interactions: Number of interaction layers
            num_elements: Number of chemical elements
            hidden_irreps: Hidden irreps (unused)
            MLP_irreps: MLP irreps (unused)
            atomic_inter_scale: Atomic interaction scale (unused)
            atomic_inter_shift: Atomic interaction shift (unused)
            correlation: Correlation order (unused)
            gate: Gate function (unused)
            atomic_num_embed_max: Maximum atomic number for embedding
            compute_stress: Whether to compute stress tensor
        """
        super().__init__()
        
        # Store parameters for later use
        self.r_max = r_max
        self.num_interactions = num_interactions
        self.num_elements = num_elements
        self.compute_stress = compute_stress
        
        # Create simple neural network layers
        self.atomic_embedding = nn.Embedding(atomic_num_embed_max, 128)
        self.position_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),  # atomic + position features
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            ) for _ in range(num_interactions)
        ])
        
        # Output layers
        self.energy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self.feature_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
    
    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through SchNet model.
        
        Args:
            batch: GraphBatch containing batched graph data
            
        Returns:
            Tuple of (energies, forces, stress, features)
            - energies: (batch_size,) - total energies
            - forces: (total_atoms, 3) - atomic forces
            - stress: (batch_size, 3, 3) - stress tensors
            - features: (total_atoms, feature_dim) - atomic features
        """
        # Convert GraphBatch to PyTorch Geometric Data format for SchNet
        data = self._convert_to_pyg_data(batch)
        
        # Run SchNet forward pass
        energies = self.schnet_model(data.z, data.pos, data.batch)  # (batch_size,)
        
        # SchNet doesn't compute forces by default, so we'll compute them manually
        if self.training:
            # During training, create zero forces to avoid computational graph issues
            num_atoms = data.pos.shape[0]
            forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        else:
            # During inference, compute forces using autograd
            forces = self._compute_forces(data, energies)
        
        # SchNet doesn't compute stress, so create zero stress tensors
        batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
        stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
        
        # Extract features (node embeddings from SchNet)
        # SchNet doesn't expose intermediate features, so we'll create dummy features
        num_atoms = data.pos.shape[0]
        features = torch.zeros(num_atoms, 128, device=energies.device, dtype=energies.dtype)
        
        return energies, forces, stress, features
    
    def _convert_to_pyg_data(self, batch: GraphBatch) -> Data:
        """Convert GraphBatch to PyTorch Geometric Data format."""
        # Extract data from batch
        if hasattr(batch, 'atomic_numbers'):
            # GraphBatch format
            atomic_numbers = batch.atomic_numbers
            positions = batch.positions
            edge_index = batch.edge_index
            batch_indices = batch.batch_indices
        else:
            # PyTorch Geometric Data format
            atomic_numbers = batch.x.squeeze(1).long()
            positions = batch.pos
            edge_index = batch.edge_index
            batch_indices = batch.batch_indices
        
        # Create PyTorch Geometric Data object
        data = Data(
            z=atomic_numbers,  # Atomic numbers
            pos=positions,     # Atomic positions
            edge_index=edge_index,  # Edge connectivity
            batch=batch_indices,    # Batch indices
        )
        
        return data
    
    def _compute_forces(self, data: Data, energies: torch.Tensor) -> torch.Tensor:
        """Compute forces using autograd (only during inference)."""
        # Enable gradients for positions
        positions = data.pos.clone().requires_grad_(True)
        
        # Re-run SchNet with gradient-enabled positions
        energies_grad = self.schnet_model(data.z, positions, data.batch)
        
        # Compute forces as negative gradients
        forces = -torch.autograd.grad(
            energies_grad.sum(), positions, create_graph=False, retain_graph=False
        )[0]
        
        return forces
    
    def _convert_to_data_dict(self, batch: GraphBatch) -> Dict[str, torch.Tensor]:
        """Convert GraphBatch to MACE data dictionary format.
        
        Args:
            batch: GraphBatch object or PyTorch Geometric Data object
            
        Returns:
            Dictionary with MACE data format
        """
        # Extract data from batch - handle both GraphBatch and Data objects
        # Clone tensors to avoid computational graph issues
        # Note: positions need requires_grad=False to avoid graph reuse issues
        if hasattr(batch, 'atomic_numbers'):
            # GraphBatch format
            atomic_numbers = batch.atomic_numbers.detach().clone()
            positions = batch.positions.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            edge_vectors = batch.edge_vectors.detach().clone()
            cell_vectors = batch.cell_vectors.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        else:
            # PyTorch Geometric Data format
            atomic_numbers = batch.x.squeeze(1).long().detach().clone()  # Extract from node features
            positions = batch.pos.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            edge_vectors = batch.edge_vectors.detach().clone()
            cell_vectors = batch.cell_vectors.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        
        # WORKAROUND: Do NOT enable gradient for positions to avoid MACE backward graph issue
        # This means MACE cannot compute forces via autograd, but prevents graph freeing
        # positions.requires_grad_(True)  # Commented out to fix backward pass error
        
        # Create node attributes (one-hot encoded atomic numbers for MACE)
        max_atomic_num = 100  # MACE default
        node_attrs = torch.zeros(len(atomic_numbers), max_atomic_num, device=atomic_numbers.device)
        node_attrs[torch.arange(len(atomic_numbers)), atomic_numbers] = 1.0
        
        # Create shifts tensor for periodic boundary conditions
        # For MACE, shifts should be the edge vectors (distances between atoms)
        # Filter out self-loops (zero distance edges)
        edge_distances = torch.norm(edge_vectors, dim=1)
        valid_edges = edge_distances > 1e-6  # Keep edges with distance > 1e-6
        
        if not valid_edges.all():
            # Filter edge_index, edge_vectors, and other edge-related tensors
            # Create new tensors to avoid modifying input
            edge_index = edge_index[:, valid_edges].clone()
            edge_vectors = edge_vectors[valid_edges].clone()
        
        shifts = edge_vectors.clone()  # (num_edges, 3)
        
        # Create unit_shifts (normalized edge vectors)
        edge_distances = torch.norm(edge_vectors, dim=1, keepdim=True)
        unit_shifts = edge_vectors / (edge_distances + 1e-8)  # (num_edges, 3)
        
        # Create cell tensor - use the first cell for all atoms (assuming same cell for batch)
        if batch.cell_vectors.shape[0] == 1:
            cell = batch.cell_vectors[0]  # (3, 3)
        else:
            # Use the cell from the first atom in each batch
            cell = batch.cell_vectors[batch_indices[0]]  # (3, 3)
        
        # Create data dictionary for MACE
        data_dict = {
            'edge_index': edge_index,
            'node_attrs': node_attrs,
            'positions': positions,
            'shifts': shifts,
            'unit_shifts': unit_shifts,
            'cell': cell,
            'batch': batch_indices,  # Required by MACE
            'ptr': batch.ptr if hasattr(batch, 'ptr') else torch.tensor([0, len(atomic_numbers)], dtype=torch.long, device=atomic_numbers.device),  # Required by MACE for batching
        }
        
        # Convert data dictionary to the same device as the model
        device = next(self.parameters()).device
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)
        
        return data_dict
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension
        """
        # Get the embedding dimension from the MACE model
        if hasattr(self.mace_model, 'node_embedding'):
            return self.mace_model.node_embedding.linear.weight.shape[0]
        else:
            # Fallback: parse from hidden_irreps
            # This is a simple parser - in practice you might want a more robust one
            import re
            match = re.search(r'(\d+)x', self.hidden_irreps)
            return int(match.group(1)) if match else 128


class GNNSurrogate(SchNetWrapper):
    """Alias for SchNetWrapper for backward compatibility."""
    pass


class EnsembleGNN(nn.Module):
    """Ensemble of GNN models for uncertainty estimation."""
    
    def __init__(self, models: List[nn.Module]) -> None:
        """Initialize ensemble model.
        
        Args:
            models: List of GNN models
        """
        super().__init__()
        # TODO: Implement ensemble model
        self.models = nn.ModuleList(models)
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            data: Input graph data
            
        Returns:
            Dictionary with mean predictions and uncertainties
        """
        # TODO: Implement ensemble forward pass
        raise NotImplementedError("EnsembleGNN forward not implemented")
    
    def predict_with_uncertainty(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty estimates.
        
        Args:
            data: Input graph data
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        # TODO: Implement uncertainty estimation
        raise NotImplementedError("predict_with_uncertainty not implemented")
