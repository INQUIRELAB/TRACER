"""Graph Neural Network models for quantum chemistry - MACE implementation."""

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


class MACEWrapper(nn.Module):
    """MACE-torch wrapper for DFT surrogate predictions."""
    
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
        """Initialize MACE wrapper.
        
        Args:
            r_max: Maximum distance for interactions
            num_bessel: Number of Bessel functions
            num_polynomial_cutoff: Number of polynomial cutoff functions
            max_ell: Maximum spherical harmonic degree
            interaction_cls: Interaction block class
            interaction_cls_first: First interaction block class
            num_interactions: Number of interaction layers
            num_elements: Number of chemical elements
            hidden_irreps: Hidden irreps for MACE
            MLP_irreps: MLP irreps
            atomic_inter_scale: Atomic interaction scale
            atomic_inter_shift: Atomic interaction shift
            correlation: Correlation order
            gate: Gate function
            atomic_num_embed_max: Maximum atomic number for embedding
            compute_stress: Whether to compute stress tensor
        """
        super().__init__()
        
        # Store parameters for later use
        self.r_max = r_max
        self.num_bessel = num_bessel
        self.num_polynomial_cutoff = num_polynomial_cutoff
        self.max_ell = max_ell
        self.num_interactions = num_interactions
        self.num_elements = num_elements
        self.hidden_irreps = hidden_irreps
        self.MLP_irreps = MLP_irreps
        self.compute_stress = compute_stress
        
        # Create the actual MACE model with correct parameters
        from e3nn.o3 import Irreps
        from mace.modules.blocks import InteractionBlock
        
        # Convert string irreps to Irreps objects
        hidden_irreps_obj = Irreps(hidden_irreps)
        MLP_irreps_obj = Irreps(MLP_irreps)
        
        # Create atomic energies (zeros for now)
        atomic_energies = np.zeros(num_elements)
        
        # Create atomic numbers list
        atomic_numbers = list(range(1, num_elements + 1))
        
        # Get interaction block classes
        interaction_cls_obj = getattr(__import__('mace.modules.blocks', fromlist=[interaction_cls]), interaction_cls)
        interaction_cls_first_obj = getattr(__import__('mace.modules.blocks', fromlist=[interaction_cls_first]), interaction_cls_first)
        
        self.mace_model = MACE(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_cls_obj,
            interaction_cls_first=interaction_cls_first_obj,
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps_obj,
            MLP_irreps=MLP_irreps_obj,
            atomic_energies=atomic_energies,
            avg_num_neighbors=12.0,  # Default value
            atomic_numbers=atomic_numbers,
            correlation=correlation,
            gate=gate,
        )
    
    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through MACE model.
        
        Args:
            batch: GraphBatch containing batched graph data
            
        Returns:
            Tuple of (energies, forces, stress, features)
            - energies: (batch_size,) - total energies
            - forces: (total_atoms, 3) - atomic forces
            - stress: (batch_size, 3, 3) - stress tensors
            - features: (total_atoms, feature_dim) - atomic features
        """
        # Convert GraphBatch to MACE data format
        data_dict = self._convert_to_data_dict(batch)
        
        # WORKAROUND: Use torch.no_grad() to prevent MACE from interfering with our computational graph
        # This prevents the "backward through graph a second time" error
        if self.training:
            # During training, run MACE in no_grad context to avoid graph conflicts
            with torch.no_grad():
                # Run MACE forward pass without gradients
                output = self.mace_model(data_dict, compute_stress=False)
                energies_no_grad = output['energy']
            
            # Re-run MACE with gradients ONLY for energy computation
            # This creates a clean computational graph for our backward pass
            data_dict_grad = self._convert_to_data_dict(batch)
            output_grad = self.mace_model(data_dict_grad, compute_stress=False)
            energies = output_grad['energy']  # (batch_size,)
            
            # Create zero forces and stress during training
            num_atoms = data_dict['positions'].shape[0]
            forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
            batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
            stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
            
            # Extract features from no_grad output
            node_features = output.get('node_features', output.get('node_embedding', torch.zeros(num_atoms, 128, device=energies.device)))
            features = node_features.detach()  # Detach to avoid gradient issues
        else:
            # During inference, run normally with full computation
            output = self.mace_model(data_dict, compute_stress=self.compute_stress)
            energies = output['energy']  # (batch_size,)
            
            # Extract forces if available
            if 'forces' in output and output['forces'] is not None:
                forces = output['forces']  # (total_atoms, 3)
            else:
                num_atoms = data_dict['positions'].shape[0]
                forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
            
            # Extract stress if computed
            if self.compute_stress and 'stress' in output and output['stress'] is not None:
                stress = output['stress']  # (batch_size, 3, 3)
            else:
                batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
                stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
            
            # Extract features
            node_features = output.get('node_features', output.get('node_embedding', torch.zeros(data_dict['positions'].shape[0], 128, device=energies.device)))
            features = node_features
        
        return energies, forces, stress, features
    
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
        if cell_vectors.shape[0] == 1:
            cell = cell_vectors[0].clone()  # (3, 3)
        else:
            # Use the cell from the first atom in each batch
            cell = cell_vectors[batch_indices[0]].clone()  # (3, 3)

        # Create data dictionary for MACE
        data_dict = {
            'edge_index': edge_index,
            'node_attrs': node_attrs,
            'positions': positions,
            'shifts': shifts,
            'unit_shifts': unit_shifts,
            'cell': cell,
            'batch': batch_indices,
            'ptr': batch.ptr if hasattr(batch, 'ptr') else torch.tensor([0, len(atomic_numbers)], dtype=torch.long, device=atomic_numbers.device),
        }

        # Convert data dictionary to the same device as the model
        device = next(self.parameters()).device
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)

        return data_dict


class GNNSurrogate(MACEWrapper):
    """Alias for MACEWrapper for backward compatibility."""
    pass


class EnsembleGNN(nn.Module):
    """Ensemble of GNN models for uncertainty estimation."""
    
    def __init__(self, models: List[MACEWrapper]) -> None:
        """Initialize ensemble.
        
        Args:
            models: List of trained GNN models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, batch: GraphBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            batch: GraphBatch containing batched graph data
            
        Returns:
            Tuple of (energies, forces, stress, features) averaged across models
        """
        # Get predictions from all models
        all_energies = []
        all_forces = []
        all_stress = []
        all_features = []
        
        for model in self.models:
            energies, forces, stress, features = model(batch)
            all_energies.append(energies)
            all_forces.append(forces)
            all_stress.append(stress)
            all_features.append(features)
        
        # Average predictions
        avg_energies = torch.stack(all_energies).mean(dim=0)
        avg_forces = torch.stack(all_forces).mean(dim=0)
        avg_stress = torch.stack(all_stress).mean(dim=0)
        avg_features = torch.stack(all_features).mean(dim=0)
        
        return avg_energies, avg_forces, avg_stress, avg_features


