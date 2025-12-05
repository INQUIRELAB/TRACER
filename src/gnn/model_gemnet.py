"""
GemNet-inspired GNN implementation
Geometric Message Passing Neural Network with directional information.

Based on:
- Gastegger, M., et al. (2021). GemNet: Universal directional graph neural networks for molecules.
- Ganea, O., et al. (2021). Neural Message Passing for Joint Physics-Based Reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models.schnet import GaussianSmearing


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for domain adaptation."""
    
    def __init__(self, feature_dim: int, film_dim: int, use_bias: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.film_dim = film_dim
        self.use_bias = use_bias
        
        # FiLM parameters: gamma (scale) and beta (shift)
        self.gamma_proj = nn.Linear(film_dim, feature_dim)
        self.beta_proj = nn.Linear(film_dim, feature_dim) if use_bias else None
        
    def forward(self, features: torch.Tensor, film_input: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to features.
        
        Args:
            features: Input features (batch_size, feature_dim)
            film_input: FiLM conditioning input (batch_size, film_dim)
            
        Returns:
            Modulated features (batch_size, feature_dim)
        """
        gamma = self.gamma_proj(film_input)  # Scale
        beta = self.beta_proj(film_input) if self.beta_proj is not None else 0  # Shift
        
        return gamma * features + beta


class DomainEmbedding(nn.Module):
    """Domain embedding module for multi-domain training."""
    
    def __init__(self, num_domains: int = 5, embedding_dim: int = 16):
        super().__init__()
        self.num_domains = num_domains
        self.embedding_dim = embedding_dim
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        
        # Domain mapping (matches training data)
        # 0: JARVIS-DFT, 1: JARVIS-Elastic, 2: OC20-S2EF, 3: OC22-S2EF, 4: ANI1x
        self.domain_mapping = {
            'jarvis_dft': 0,
            'jarvis_elastic': 1,
            'oc20_s2ef': 2,
            'oc22_s2ef': 3,
            'ani1x': 4
        }
    
    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Get domain embeddings for batch.
        
        Args:
            domain_ids: Domain IDs (batch_size,) with values 0-4
            
        Returns:
            Domain embeddings (batch_size, embedding_dim)
        """
        return self.domain_embeddings(domain_ids)


class GemNetEmbedding(nn.Module):
    """Embedding layer for atomic numbers."""
    
    def __init__(self, num_atoms: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atoms, embedding_dim)
        
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        return self.embedding(atomic_numbers)


class DirectionalMessagePassing(nn.Module):
    """Directional message passing with geometric information."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_filters: int,
        radial_dim: int = 50,
        spherical_dim: int = 7,
        cutoff: float = 10.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.cutoff = cutoff
        
        # Radial basis functions (distances)
        self.radial_basis = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=radial_dim)
        
        # Message networks
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + radial_dim, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Interaction layers for different geometric orders
        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with directional message passing.
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, feature_dim)
            distances: Pairwise distances (num_edges,)
        
        Returns:
            Updated node features
        """
        # Expand radial basis
        radial_features = self.radial_basis(distances)
        
        # Get edge attributes (sender, receiver indices)
        row, col = edge_index
        
        # Concatenate features along edges
        edge_features = torch.cat([
            x[row],  # Sender features
            x[col],  # Receiver features
            radial_features  # Distance features
        ], dim=-1)
        
        # Compute messages
        messages = self.message_net(edge_features)
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out = out.index_add(0, col, messages)
        
        # Update with self information
        updated = self.update_net(torch.cat([x, out], dim=-1))
        updated = self.interaction_net(updated)
        
        return updated


class GemNetBlock(nn.Module):
    """One block of GemNet with directional message passing."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_filters: int,
        cutoff: float = 10.0
    ):
        super().__init__()
        
        self.message_passing = DirectionalMessagePassing(
            hidden_dim=hidden_dim,
            num_filters=num_filters,
            cutoff=cutoff
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        # Message passing
        out = self.message_passing(x, edge_index, edge_attr, distances)
        
        # Residual connection + normalization
        out = x + out
        out = self.norm(out)
        
        return out


class GemNetOutput(nn.Module):
    """Output layer for GemNet with optional FiLM adaptation and multi-property support."""
    
    def __init__(
        self,
        hidden_dim: int,
        readout: str = "sum",
        use_film: bool = False,
        film_dim: int = 16,
        properties: list = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.use_film = use_film
        
        # Properties to predict (default: formation_energy_per_atom)
        if properties is None:
            properties = ['formation_energy_per_atom']
        self.properties = properties
        
        # FiLM layer (if enabled)
        if use_film:
            self.film_layer = FiLMLayer(feature_dim=hidden_dim, film_dim=film_dim, use_bias=True)
        else:
            self.film_layer = None
        
        # Multi-property output heads
        self.output_heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for prop in properties
        })
    
    def forward(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor,
        domain_emb: Optional[torch.Tensor] = None,
        num_atoms: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute graph-level outputs for multiple properties with optional FiLM adaptation.
        
        Args:
            x: Node features (num_nodes, hidden_dim)
            batch: Batch indices (num_nodes,)
            domain_emb: Domain embeddings (batch_size, film_dim) for FiLM, or None
            num_atoms: Number of atoms per sample (batch_size,) for computing total energy
        
        Returns:
            Dictionary of property predictions {property_name: predictions (batch_size,)}
        """
        # Readout aggregation
        if self.readout == "sum":
            graph_features = []
            for b in range(batch.max().item() + 1):
                mask = (batch == b)
                graph_features.append(x[mask].sum(dim=0))
            graph_features = torch.stack(graph_features)
        elif self.readout == "mean":
            graph_features = []
            for b in range(batch.max().item() + 1):
                mask = (batch == b)
                graph_features.append(x[mask].mean(dim=0))
            graph_features = torch.stack(graph_features)
        else:
            # Default to sum
            graph_features = []
            for b in range(batch.max().item() + 1):
                mask = (batch == b)
                graph_features.append(x[mask].sum(dim=0))
            graph_features = torch.stack(graph_features)
        
        # Apply FiLM if enabled and domain embeddings provided
        if self.use_film and self.film_layer is not None and domain_emb is not None:
            graph_features = self.film_layer(graph_features, domain_emb)
        
        # Compute predictions for each property
        predictions = {}
        for prop in self.properties:
            predictions[prop] = self.output_heads[prop](graph_features).squeeze(-1)
        
        return predictions


class GemNetWrapper(nn.Module):
    """Wrapper for GemNet model with forces, stress prediction, and domain-aware FiLM adaptation."""
    
    def __init__(
        self,
        num_atoms: int = 100,
        hidden_dim: int = 256,
        num_filters: int = 256,
        num_interactions: int = 6,
        cutoff: float = 10.0,
        readout: str = "sum",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        use_film: bool = True,
        num_domains: int = 5,
        film_dim: int = 16
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.mean = mean
        self.std = std
        self.use_film = use_film
        self.num_domains = num_domains
        
        # Embedding layer
        self.embedding = GemNetEmbedding(num_atoms, hidden_dim)
        
        # Domain embedding (for FiLM)
        if use_film:
            self.domain_embedding = DomainEmbedding(num_domains=num_domains, embedding_dim=film_dim)
        else:
            self.domain_embedding = None
        
        # Interaction blocks
        self.blocks = nn.ModuleList([
            GemNetBlock(
                hidden_dim=hidden_dim,
                num_filters=num_filters,
                cutoff=cutoff
            ) for _ in range(num_interactions)
        ])
        
        # Output layer with FiLM support
        # Note: Architecture supports multi-property prediction (can be extended to band gap, etc.)
        # Currently focused on formation energy per atom for this study
        self.output = GemNetOutput(hidden_dim, readout, use_film=use_film, film_dim=film_dim,
                                   properties=['formation_energy_per_atom'])
        
        # Additional layers for forces and stress
        self.force_layer = nn.Linear(hidden_dim, 3)
        self.stress_layer = nn.Linear(hidden_dim, 9)
        
    def forward(
        self, 
        batch, 
        compute_forces: bool = False,
        domain_id: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through GemNet with optional domain adaptation.
        
        Args:
            batch: PyTorch Geometric Data or Batch object
            compute_forces: Whether to compute forces via autograd
            domain_id: Domain IDs (batch_size,) for FiLM adaptation, or None
        
        Returns:
            energies: Total energies (batch_size,)
            forces: Atomic forces (num_atoms, 3) or None
            stress: Stress tensors (batch_size, 3, 3) or None
        """
        # Get node features and atomic numbers
        if hasattr(batch, 'atomic_numbers'):
            atomic_numbers = batch.atomic_numbers
        elif hasattr(batch, 'z'):
            atomic_numbers = batch.z
        else:
            atomic_numbers = batch.x.squeeze(1).long()
        
        # Get positions
        positions = batch.pos
        
        # Get cell if available (for PBC)
        cell = None
        if hasattr(batch, 'cell') and batch.cell is not None:
            cell = batch.cell
        elif hasattr(batch, 'lattice') and batch.lattice is not None:
            cell = batch.lattice
        
        # Embed atomic numbers
        x = self.embedding(atomic_numbers)
        
        # Compute distances and edges (with PBC if cell available)
        edge_index, edge_attr, distances = self._compute_edges(positions, batch.batch, cell=cell)
        
        # Pass through interaction blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_attr, distances)
        
        # Get domain embeddings if FiLM is enabled
        domain_emb = None
        if self.use_film and self.domain_embedding is not None:
            if domain_id is not None:
                domain_emb = self.domain_embedding(domain_id)
            elif hasattr(batch, 'domain_id'):
                # Extract domain_id from batch if available
                batch_size = batch.batch.max().item() + 1
                batch_domain_ids = []
                for b in range(batch_size):
                    mask = (batch.batch == b)
                    # Get domain_id for first node in each graph
                    domain_idx = batch.domain_id[mask][0] if hasattr(batch, 'domain_id') else 0
                    batch_domain_ids.append(domain_idx)
                domain_id = torch.tensor(batch_domain_ids, dtype=torch.long, device=x.device)
                domain_emb = self.domain_embedding(domain_id)
        
        # Compute energies (in normalized space) with optional FiLM
        # Get number of atoms per sample for total energy computation
        num_atoms_per_sample = torch.tensor(
            [torch.sum(batch.batch == i).item() for i in range(batch.batch.max().item() + 1)],
            device=x.device, dtype=x.dtype
        )
        
        # Get property predictions (currently formation energy per atom)
        property_predictions = self.output(x, batch.batch, domain_emb=domain_emb)
        
        # Extract formation energy per atom
        energies = property_predictions.get('formation_energy_per_atom')
        
        # Store all property predictions for access
        self._property_predictions = property_predictions
        
        # Do NOT denormalize here - keep in normalized space for training
        # We'll denormalize only when needed for final predictions
        
        # Compute forces if requested
        forces = None
        if compute_forces and self.training:
            positions.requires_grad_(True)
            if positions.grad is not None:
                positions.grad.zero_()
            force_energy = self.forward_single(batch)
            forces = -torch.autograd.grad(
                force_energy.sum(), 
                positions, 
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if forces is None:
                num_atoms = positions.shape[0]
                forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        
        # Compute stress (simplified - zeros for now)
        batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
        stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
        
        return energies, forces, stress
    
    def predict_properties(self, data: Data, domain_id: int = 0) -> dict:
        """
        Predict properties for a single structure.
        Currently returns formation energy per atom.
        Architecture supports extension to additional properties (band gap, etc.).
        
        Args:
            data: PyG Data object with structure
            domain_id: Domain ID for FiLM adaptation
        
        Returns:
            Dictionary of property predictions (currently: formation_energy_per_atom)
        """
        self.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([data])
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            # Forward pass
            _ = self.forward(batch, domain_id=torch.tensor([domain_id], device=batch.batch.device))
            
            # Return stored property predictions
            return self._property_predictions
    
    def _compute_edges(self, positions: torch.Tensor, batch_indices: torch.Tensor, 
                      cell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute edges based on cutoff distance with optional PBC (Periodic Boundary Conditions).
        
        Args:
            positions: Atomic positions (N, 3)
            batch_indices: Batch indices (N,)
            cell: Unit cell matrix (N_cells, 3, 3) or (3, 3) if single structure. If None, no PBC.
            
        Returns:
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge attributes (E, 3)
            distances: Edge distances (E,)
        """
        device = positions.device
        n_atoms = positions.shape[0]
        
        # If cell is provided, use PBC-aware edge computation
        if cell is not None:
            return self._compute_edges_pbc(positions, batch_indices, cell)
        
        # Non-PBC fallback (for molecules or non-periodic systems)
        try:
            from torch_geometric.nn import radius_graph
            # Use radius_graph if available (requires torch-cluster)
            edge_index = radius_graph(positions, r=self.cutoff, batch=batch_indices)
        except (ImportError, AttributeError):
            # Fallback: compute edges using distance matrix
            # This is slower but doesn't require torch-cluster
            distances_matrix = torch.cdist(positions, positions)
            
            # Create edge mask (within cutoff, no self-loops)
            edge_mask = (distances_matrix < self.cutoff) & (distances_matrix > 1e-8)
            
            # Handle batching: only connect atoms within same graph
            if batch_indices is not None:
                batch_i = batch_indices.unsqueeze(1)
                batch_j = batch_indices.unsqueeze(0)
                same_batch = (batch_i == batch_j)
                edge_mask = edge_mask & same_batch
            
            # Get edge indices
            edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        
        # Compute distances and edge attributes
        row, col = edge_index
        edge_vec = positions[row] - positions[col]
        distances = torch.norm(edge_vec, dim=-1)
        
        # Edge attributes (normalized edge vectors)
        edge_attr = edge_vec / (distances.unsqueeze(-1) + 1e-8)
        
        return edge_index, edge_attr, distances
    
    def _compute_edges_pbc(self, positions: torch.Tensor, batch_indices: torch.Tensor, 
                           cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute edges with Periodic Boundary Conditions using minimum image convention.
        
        This is critical for crystal structures where atoms near cell boundaries
        must connect to periodic images.
        """
        import numpy as np
        device = positions.device
        
        # Convert to numpy for cell operations
        positions_np = positions.detach().cpu().numpy()
        
        # Handle cell format: (N_cells, 3, 3) or (3, 3)
        if len(cell.shape) == 2:
            cell_np = cell.detach().cpu().numpy()
            # Verify it's 3x3
            if cell_np.shape != (3, 3):
                # Invalid cell shape, fall back to non-PBC
                return self._compute_edges(positions, batch_indices, cell=None)
            # Expand to (N_atoms, 3, 3) for per-atom cell
            n_atoms = positions.shape[0]
            cell_np = np.tile(cell_np[None, :, :], (n_atoms, 1, 1))
        else:
            cell_np = cell.detach().cpu().numpy()
            # Verify shape
            if len(cell_np.shape) == 3 and cell_np.shape[-2:] != (3, 3):
                return self._compute_edges(positions, batch_indices, cell=None)
        
        # Compute minimum image distances
        edge_list = []
        distances_list = []
        edge_vec_list = []
        
        n_atoms = positions.shape[0]
        cutoff_sq = self.cutoff ** 2
        
        for i in range(n_atoms):
            pos_i = positions_np[i]
            cell_i = cell_np[i] if len(cell_np.shape) == 3 else cell_np[0]
            
            for j in range(i + 1, n_atoms):
                # Check if same batch
                if batch_indices is not None:
                    if batch_indices[i].item() != batch_indices[j].item():
                        continue
                
                pos_j = positions_np[j]
                
                # Apply minimum image convention
                dr = pos_j - pos_i
                
                # Convert to fractional coordinates
                inv_cell = np.linalg.inv(cell_i)
                dr_frac = dr @ inv_cell
                
                # Apply periodic boundary conditions
                dr_frac = dr_frac - np.round(dr_frac)
                
                # Convert back to cartesian
                dr_min = dr_frac @ cell_i
                
                # Check if within cutoff
                dist_sq = np.sum(dr_min ** 2)
                if dist_sq < cutoff_sq and dist_sq > 1e-16:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected
                    
                    dist = np.sqrt(dist_sq)
                    distances_list.append(dist)
                    distances_list.append(dist)
                    
                    edge_vec_list.append(dr_min)
                    edge_vec_list.append(-dr_min)
        
        if len(edge_list) == 0:
            # Fallback to non-PBC if no edges found
            return self._compute_edges(positions, batch_indices, cell=None)
        
        # Convert back to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        distances = torch.tensor(distances_list, dtype=torch.float32, device=device)
        edge_attr = torch.tensor(edge_vec_list, dtype=torch.float32, device=device)
        edge_attr = edge_attr / (distances.unsqueeze(-1) + 1e-8)
        
        return edge_index, edge_attr, distances
    
    def forward_single(self, batch, domain_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single forward pass for force computation."""
        return self.forward(batch, compute_forces=False, domain_id=domain_id)[0]
    
    def denormalize_energy(self, energy_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize energy from normalized to eV scale."""
        if self.mean is not None and self.std is not None:
            return energy_norm * self.std + self.mean
        return energy_norm
    
    def normalize_energy(self, energy: torch.Tensor) -> torch.Tensor:
        """Normalize energy from eV to normalized scale."""
        if self.mean is not None and self.std is not None:
            return (energy - self.mean) / self.std
        return energy

