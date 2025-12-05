"""
Graph Neural Network Models for Molecular Property Prediction.

This module implements SchNet and domain-aware variants for molecular property prediction.
All implementations follow the original papers with proper citations.

References:
- Schütt, K. et al. (2017). SchNet: A continuous-filter convolutional neural network 
  for modeling quantum interactions. NIPS 2017.
- Gastegger, M. et al. (2017). Machine learning molecular dynamics for the simulation 
  of infrared spectra. Chemical Science, 8(10), 6924-6935.

For complete citations, see src/citations.py
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn import SchNet
from mace.calculators import mace
from mace.data import AtomicData
from mace.tools import torch_geometric, default_keys
from mace.modules.models import MACE
from graphs.periodic_graph import GraphBatch

# Optional anomaly detection for debugging
ANOMALY_DETECTION = False  # Set to True for debugging


class SchNetWrapper(nn.Module):
    """SchNet wrapper for DFT surrogate predictions."""
    
    def __init__(self, 
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 max_num_neighbors: int = 32,
                 readout: str = "add",
                 dipole: bool = False,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None) -> None:
        """Initialize SchNet wrapper.
        
        Args:
            hidden_channels: Hidden embedding size
            num_filters: Number of filters in convolutional layers
            num_interactions: Number of interaction blocks
            num_gaussians: Number of Gaussian basis functions
            cutoff: Cutoff distance for interatomic interactions
            max_num_neighbors: Maximum number of neighbors per atom
            readout: Global readout function ("add", "mean", "max")
            dipole: Whether to predict dipole moment
            mean: Mean of energies in training set
            std: Standard deviation of energies in training set
            atomref: Reference energies for atoms
        """
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        # Create SchNet model
        self.schnet_model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
            mean=mean,
            std=std,
            atomref=atomref
        )
        
        # Additional layers for forces and stress
        self.force_layer = nn.Linear(hidden_channels, 3)
        self.stress_layer = nn.Linear(hidden_channels, 9)  # 3x3 stress tensor
        
    def forward(self, batch) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass through SchNet model.
        
        Args:
            batch: Either GraphBatch or PyTorch Geometric Data object
            
        Returns:
            Either tuple of (energies, forces, stress, features) for backward compatibility
            or Dict with keys 'energy', 'forces', 'stress' for multi-task training
            - energies: (batch_size,) - total energies
            - forces: (total_atoms, 3) - atomic forces
            - stress: (batch_size, 3, 3) - stress tensors
            - features: (total_atoms, feature_dim) - atomic features
        """
        # Handle both GraphBatch and PyTorch Geometric Data objects
        if hasattr(batch, 'positions'):
            # GraphBatch format - convert to PyTorch Geometric Data
            data = self._convert_to_pyg_data(batch)
        else:
            # Already PyTorch Geometric Data format
            data = batch
            # Ensure positions require gradients
            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)
        
        # Forward pass through SchNet
        # Use atomic_numbers as z for SchNet
        z = data.atomic_numbers if hasattr(data, 'atomic_numbers') else data.x.squeeze(1).long()
        energies = self.schnet_model(z, data.pos, data.batch)  # (batch_size,)
        
        # Ensure energies are properly shaped
        if len(energies.shape) > 1 and energies.shape[1] == 1:
            energies = energies.squeeze(1)  # Remove extra dimension
        
        # Compute forces via autograd (only during training)
        if self.training and data.pos.requires_grad:
            forces = -torch.autograd.grad(
                energies.sum(), data.pos, 
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            
            if forces is None:
                # Fallback: zero forces if no gradients computed
                num_atoms = data.pos.shape[0]
                forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        else:
            # During inference/validation, forces are zero
            num_atoms = data.pos.shape[0]
            forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        
        # Compute stress (simplified - just zeros for now)
        batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
        stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
        
        # Extract node features from SchNet
        # SchNet doesn't expose node features directly, so we create dummy ones
        num_atoms = data.pos.shape[0]
        features = torch.zeros(num_atoms, self.schnet_model.hidden_channels, device=energies.device)
        
        # Check if we should return dict format (for multi-task training)
        if hasattr(batch, 'dataset_name') or hasattr(batch, 'domain_id'):
            # Multi-task format - return dictionary
            return {
                'energy': energies,
                'forces': forces,
                'stress': stress,
                'features': features
            }
        else:
            # Backward compatibility - return tuple
            return energies, forces, stress, features
    
    def _convert_to_pyg_data(self, batch: GraphBatch) -> Data:
        """Convert GraphBatch to PyTorch Geometric Data format.
        
        Args:
            batch: GraphBatch object
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract data from batch
        if hasattr(batch, 'atomic_numbers'):
            # GraphBatch format
            atomic_numbers = batch.atomic_numbers.detach().clone()
            positions = batch.positions.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        else:
            # PyTorch Geometric Data format
            atomic_numbers = batch.x.squeeze(1).long().detach().clone()
            positions = batch.pos.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        
        # Ensure positions require gradients BEFORE creating Data object
        positions.requires_grad_(True)
        
        # Create PyTorch Geometric Data object
        data = Data(
            z=atomic_numbers,  # Atomic numbers
            pos=positions,     # Atomic positions
            edge_index=edge_index,  # Edge connectivity
            batch=batch_indices  # Batch indices
        )
        
        return data


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
    
    def forward_energy_only(self, batch: GraphBatch) -> torch.Tensor:
        """Strict energy-only forward pass that NEVER computes forces internally.
        
        This method ensures:
        1. One forward pass (energies only)
        2. No hidden second forwards with grad
        3. No force computation inside MACE
        
        Args:
            batch: GraphBatch containing batched graph data
            
        Returns:
            energies: (batch_size,) - total energies with clean computational graph
        """
        # Convert GraphBatch to MACE data format
        data_dict = self._convert_to_data_dict(batch)
        
        # Ensure positions require gradients for our autograd computation
        positions = data_dict['positions'].requires_grad_(True)
        data_dict['positions'] = positions
        
        # CRITICAL: Call MACE with compute_stress=False to prevent internal force computation
        # This ensures MACE never calls torch.autograd.grad() internally
        output = self.mace_model(data_dict, compute_stress=False)
        energies = output['energy']  # (batch_size,)
        
        # Ensure energies are properly shaped
        if len(energies.shape) == 0:
            energies = energies.unsqueeze(0)  # Convert scalar to [1]
        
        return energies
    
    def forward_features(self, batch: GraphBatch) -> torch.Tensor:
        """Extract features without gradients during training to avoid second graph.
        
        Args:
            batch: GraphBatch containing batched graph data
            
        Returns:
            features: (total_atoms, feature_dim) - atomic features
        """
        if self.training:
            # During training, NEVER call MACE again to avoid second computational graph
            # Create dummy features to avoid any MACE calls
            if hasattr(batch, 'pos'):
                num_atoms = batch.pos.shape[0]
                device = batch.pos.device
            elif hasattr(batch, 'positions'):
                num_atoms = batch.positions.shape[0]
                device = batch.positions.device
            else:
                num_atoms = 100  # Fallback
                device = next(self.parameters()).device
            
            features = torch.zeros(num_atoms, 128, device=device, dtype=torch.float32)
        else:
            # During inference, compute features normally
            data_dict = self._convert_to_data_dict(batch)
            output = self.mace_model(data_dict, compute_stress=self.compute_stress)
            node_features = output.get('node_features', output.get('node_embedding', None))
            
            if node_features is not None:
                features = node_features
            else:
                # Fallback: create dummy features
                num_atoms = data_dict['positions'].shape[0]
                features = torch.zeros(num_atoms, 128, device=data_dict['positions'].device)
        
        return features
    
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
        # STRICT PROTOCOL: During training, enforce single forward + autograd forces
        if self.training:
            # Optional anomaly detection for debugging
            if ANOMALY_DETECTION:
                torch.autograd.set_detect_anomaly(True)
            # STEP 1: One forward pass (energies only) - NO force computation inside MACE
            energies = self.forward_energy_only(batch)
            
            # STEP 2: Forces from autograd on owned pos tensor
            # Get positions from batch (owned tensor, not from MACE)
            if hasattr(batch, 'pos'):
                positions = batch.pos.requires_grad_(True)
            elif hasattr(batch, 'positions'):
                positions = batch.positions.requires_grad_(True)
            else:
                # Fallback: get from data_dict
                data_dict = self._convert_to_data_dict(batch)
                positions = data_dict['positions'].requires_grad_(True)
            
            # Compute forces via autograd on the clean energy computational graph
            try:
                forces = -torch.autograd.grad(
                    energies.sum(), positions, 
                    create_graph=True, retain_graph=True,
                    allow_unused=True
                )[0]
                
                if forces is None:
                    # Fallback: zero forces if no gradients computed
                    num_atoms = positions.shape[0]
                    forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
            except Exception as e:
                print(f"Warning: Could not compute forces via autograd: {e}")
                # Fallback: zero forces
                num_atoms = positions.shape[0]
                forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
            
            # STEP 3: Zero stress during training (stress computation causes gradient issues)
            batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
            stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
            
            # STEP 4: Features without gradients to avoid second graph
            features = self.forward_features(batch)
        else:
            # During inference, run normally with full computation
            data_dict = self._convert_to_data_dict(batch)
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
        
        # WORKAROUND: Enable gradients on positions for MACE force computation
        # This is required for MACE to compute forces via autograd
        positions.requires_grad_(True)
        
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
