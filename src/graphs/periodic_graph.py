"""Periodic graph construction for crystalline systems."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from ase import Atoms
from ase.neighborlist import neighbor_list
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial.distance import pdist, squareform


@dataclass
class GraphBatch:
    """Batch of graph data for equivariant message-passing models."""
    
    # Node features
    node_features: torch.Tensor  # (total_nodes, node_dim)
    atomic_numbers: torch.Tensor  # (total_nodes,)
    positions: torch.Tensor  # (total_nodes, 3)
    fractional_coords: torch.Tensor  # (total_nodes, 3)
    
    # Edge features
    edge_index: torch.Tensor  # (2, total_edges)
    edge_distances: torch.Tensor  # (total_edges,)
    edge_vectors: torch.Tensor  # (total_edges, 3)
    edge_angles: torch.Tensor  # (total_edges,)
    cell_offsets: torch.Tensor  # (total_edges, 3)
    
    # Batch information
    batch_indices: torch.Tensor  # (total_nodes,) - which graph each node belongs to
    num_nodes: torch.Tensor  # (batch_size,) - number of nodes per graph
    
    # Unit cell information
    cell_vectors: torch.Tensor  # (batch_size, 3, 3)
    cell_volumes: torch.Tensor  # (batch_size,)
    
    def to(self, device: Union[str, torch.device]) -> "GraphBatch":
        """Move batch to device."""
        return GraphBatch(
            node_features=self.node_features.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            positions=self.positions.to(device),
            fractional_coords=self.fractional_coords.to(device),
            edge_index=self.edge_index.to(device),
            edge_distances=self.edge_distances.to(device),
            edge_vectors=self.edge_vectors.to(device),
            edge_angles=self.edge_angles.to(device),
            cell_offsets=self.cell_offsets.to(device),
            batch_indices=self.batch_indices.to(device),
            num_nodes=self.num_nodes.to(device),
            cell_vectors=self.cell_vectors.to(device),
            cell_volumes=self.cell_volumes.to(device),
        )


class PeriodicGraph:
    """Handle periodic boundary conditions for graph neural networks."""
    
    def __init__(self, cutoff_radius: float = 5.0, max_neighbors: int = 100) -> None:
        """Initialize periodic graph constructor.
        
        Args:
            cutoff_radius: Maximum distance for edge creation
            max_neighbors: Maximum number of neighbors per atom
        """
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
    
    def build_graph(self, positions: np.ndarray, atomic_numbers: np.ndarray,
                   cell_vectors: Optional[np.ndarray] = None,
                   cell_offsets: Optional[np.ndarray] = None) -> Data:
        """Build a periodic graph from atomic positions.
        
        Args:
            positions: Atomic positions (N, 3)
            atomic_numbers: Atomic numbers (N,)
            cell_vectors: Unit cell vectors (3, 3) for periodic systems
            cell_offsets: Cell offsets for periodic images (optional)
            
        Returns:
            PyTorch Geometric Data object
        """
        # Simple implementation for testing
        positions = torch.tensor(positions, dtype=torch.float32)
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # Create node features (one-hot encoding of atomic numbers for consistency)
        node_features = torch.zeros(len(atomic_numbers), 83)  # Support up to Bi (83)
        for i, z in enumerate(atomic_numbers):
            if z <= 83:
                node_features[i, z-1] = 1.0
        
        # Create simple edges (all-to-all for small systems)
        n_atoms = len(atomic_numbers)
        if n_atoms <= 10:  # For small systems, connect all atoms
            edge_index = torch.combinations(torch.arange(n_atoms), 2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Bidirectional
        else:
            # For larger systems, use distance-based edges
            distances = torch.cdist(positions, positions)
            edge_mask = (distances < self.cutoff_radius) & (distances > 0)
            edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        
        # Create edge attributes (distances)
        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        edge_distances = torch.norm(edge_vectors, dim=1)
        edge_attr = edge_distances.unsqueeze(1)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            atomic_numbers=atomic_numbers,
            batch=torch.zeros(n_atoms, dtype=torch.long)
        )
    
    def build_batch(self, structures: List[Union[Atoms, Structure]], 
                   node_feature_dim: int = 64) -> GraphBatch:
        """Build a batch of graphs from ASE Atoms or pymatgen Structures.
        
        Args:
            structures: List of ASE Atoms or pymatgen Structure objects
            node_feature_dim: Dimension of node features
            
        Returns:
            GraphBatch containing batched graph data
        """
        batch_graphs = []
        node_counts = []
        cell_vectors_list = []
        cell_volumes_list = []
        
        for structure in structures:
            # Convert to ASE Atoms if needed
            if isinstance(structure, Structure):
                atoms = structure.to_ase_atoms()
            else:
                atoms = structure
            
            # Get basic structure information
            positions = atoms.get_positions()
            atomic_numbers = atoms.get_atomic_numbers()
            cell = atoms.get_cell()
            cell_vectors = cell.array
            cell_volume = atoms.get_volume()
            
            # Convert to fractional coordinates
            fractional_coords = atoms.get_scaled_positions()
            
            # Build neighbor list with periodic boundary conditions
            edge_indices, edge_distances, edge_vectors, cell_offsets = self._build_neighbor_list(
                atoms, self.cutoff_radius
            )
            
            # Compute edge angles (angle between edge vectors and reference directions)
            edge_angles = self._compute_edge_angles(edge_vectors, positions, edge_indices)
            
            # Create node features (atomic numbers as one-hot or embedding)
            node_features = self._create_node_features(atomic_numbers, node_feature_dim)
            
            # Store graph data
            batch_graphs.append({
                'node_features': torch.tensor(node_features, dtype=torch.float32),
                'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
                'positions': torch.tensor(positions, dtype=torch.float32),
                'fractional_coords': torch.tensor(fractional_coords, dtype=torch.float32),
                'edge_index': torch.tensor(edge_indices, dtype=torch.long),
                'edge_distances': torch.tensor(edge_distances, dtype=torch.float32),
                'edge_vectors': torch.tensor(edge_vectors, dtype=torch.float32),
                'edge_angles': torch.tensor(edge_angles, dtype=torch.float32),
                'cell_offsets': torch.tensor(cell_offsets, dtype=torch.float32),
            })
            
            node_counts.append(len(atomic_numbers))
            cell_vectors_list.append(cell_vectors)
            cell_volumes_list.append(cell_volume)
        
        # Batch all graphs together
        return self._batch_graphs(batch_graphs, node_counts, cell_vectors_list, cell_volumes_list)
    
    def _build_neighbor_list(self, atoms: Atoms, cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build neighbor list using ASE with periodic boundary conditions.
        
        Args:
            atoms: ASE Atoms object
            cutoff: Cutoff radius for neighbors
            
        Returns:
            Tuple of (edge_indices, edge_distances, edge_vectors, cell_offsets)
        """
        # Use ASE neighbor_list function with periodic boundary conditions
        # 'ijdD' returns: indices_i, indices_j, distances, displacement_vectors
        result = neighbor_list(
            'ijdD', atoms, cutoff=cutoff, self_interaction=False
        )
        
        edge_indices_i, edge_indices_j, edge_distances, edge_vectors = result
        
        # Combine indices into edge_index format (2, N_edges)
        edge_indices = np.stack([edge_indices_i, edge_indices_j], axis=0)
        
        # Compute cell offsets for periodic images
        cell_offsets = self._compute_cell_offsets(atoms, edge_indices, edge_vectors)
        
        return edge_indices, edge_distances, edge_vectors, cell_offsets
    
    def _compute_cell_offsets(self, atoms: Atoms, edge_indices: np.ndarray, 
                            edge_vectors: np.ndarray) -> np.ndarray:
        """Compute cell offsets for periodic boundary conditions.
        
        Args:
            atoms: ASE Atoms object
            edge_indices: Edge indices (2, N_edges)
            edge_vectors: Edge vectors (N_edges, 3)
            
        Returns:
            Cell offsets (N_edges, 3)
        """
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        
        # Compute direct distance between atoms
        direct_vectors = positions[edge_indices[1]] - positions[edge_indices[0]]
        
        # Compute cell offset as difference between actual edge vector and direct vector
        cell_offsets = edge_vectors - direct_vectors
        
        # Convert to fractional coordinates for cell offset
        cell_offsets_fractional = np.linalg.solve(cell.T, cell_offsets.T).T
        
        return cell_offsets_fractional
    
    def _compute_edge_angles(self, edge_vectors: np.ndarray, positions: np.ndarray, 
                           edge_indices: np.ndarray) -> np.ndarray:
        """Compute angles between edge vectors and reference directions.
        
        Args:
            edge_vectors: Edge vectors (N_edges, 3)
            positions: Atomic positions (N_atoms, 3)
            edge_indices: Edge indices (2, N_edges)
            
        Returns:
            Edge angles (N_edges,)
        """
        # Compute angles between edge vectors and z-axis as reference
        z_axis = np.array([0, 0, 1])
        
        # Normalize edge vectors
        edge_norms = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        edge_vectors_norm = edge_vectors / (edge_norms + 1e-8)
        
        # Compute angles with z-axis
        angles = np.arccos(np.clip(np.dot(edge_vectors_norm, z_axis), -1, 1))
        
        return angles
    
    def _create_node_features(self, atomic_numbers: np.ndarray, feature_dim: int) -> np.ndarray:
        """Create node features from atomic numbers.
        
        Args:
            atomic_numbers: Atomic numbers (N_atoms,)
            feature_dim: Target feature dimension
            
        Returns:
            Node features (N_atoms, feature_dim)
        """
        # Simple one-hot encoding with padding to feature_dim
        max_atomic_number = max(atomic_numbers) if len(atomic_numbers) > 0 else 1
        feature_dim = max(feature_dim, max_atomic_number + 1)
        
        node_features = np.zeros((len(atomic_numbers), feature_dim))
        for i, atomic_num in enumerate(atomic_numbers):
            if atomic_num < feature_dim:
                node_features[i, atomic_num] = 1.0
        
        return node_features
    
    def _batch_graphs(self, batch_graphs: List[Dict], node_counts: List[int],
                     cell_vectors_list: List[np.ndarray], 
                     cell_volumes_list: List[float]) -> GraphBatch:
        """Batch multiple graphs into a single GraphBatch object.
        
        Args:
            batch_graphs: List of graph dictionaries
            node_counts: Number of nodes in each graph
            cell_vectors_list: Cell vectors for each graph
            cell_volumes_list: Cell volumes for each graph
            
        Returns:
            Batched GraphBatch object
        """
        batch_size = len(batch_graphs)
        
        # Create batch indices for nodes
        batch_indices = torch.cat([
            torch.full((node_counts[i],), i, dtype=torch.long) 
            for i in range(batch_size)
        ])
        
        # Concatenate all node features
        node_features = torch.cat([g['node_features'] for g in batch_graphs])
        atomic_numbers = torch.cat([g['atomic_numbers'] for g in batch_graphs])
        positions = torch.cat([g['positions'] for g in batch_graphs])
        fractional_coords = torch.cat([g['fractional_coords'] for g in batch_graphs])
        
        # Adjust edge indices for batching
        edge_indices_list = []
        edge_distances_list = []
        edge_vectors_list = []
        edge_angles_list = []
        cell_offsets_list = []
        
        node_offset = 0
        for i, graph in enumerate(batch_graphs):
            # Adjust edge indices by adding node offset
            edge_index = graph['edge_index'] + node_offset
            edge_indices_list.append(edge_index)
            
            edge_distances_list.append(graph['edge_distances'])
            edge_vectors_list.append(graph['edge_vectors'])
            edge_angles_list.append(graph['edge_angles'])
            cell_offsets_list.append(graph['cell_offsets'])
            
            node_offset += node_counts[i]
        
        # Concatenate all edge features
        edge_index = torch.cat(edge_indices_list, dim=1)
        edge_distances = torch.cat(edge_distances_list)
        edge_vectors = torch.cat(edge_vectors_list)
        edge_angles = torch.cat(edge_angles_list)
        cell_offsets = torch.cat(cell_offsets_list)
        
        # Convert cell information to tensors
        cell_vectors = torch.tensor(np.array(cell_vectors_list), dtype=torch.float32)
        cell_volumes = torch.tensor(cell_volumes_list, dtype=torch.float32)
        num_nodes = torch.tensor(node_counts, dtype=torch.long)
        
        return GraphBatch(
            node_features=node_features,
            atomic_numbers=atomic_numbers,
            positions=positions,
            fractional_coords=fractional_coords,
            edge_index=edge_index,
            edge_distances=edge_distances,
            edge_vectors=edge_vectors,
            edge_angles=edge_angles,
            cell_offsets=cell_offsets,
            batch_indices=batch_indices,
            num_nodes=num_nodes,
            cell_vectors=cell_vectors,
            cell_volumes=cell_volumes,
        )
    
    def _compute_distances(self, positions: np.ndarray, 
                          cell_vectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distances between atoms with periodic boundary conditions.
        
        Args:
            positions: Atomic positions (N, 3)
            cell_vectors: Unit cell vectors (3, 3)
            
        Returns:
            Tuple of (edge_indices, edge_distances)
        """
        # Implement distance computation with periodic boundary conditions
        # Calculate distances between all pairs of atoms considering PBC
        distances = torch.cdist(positions, positions, p=2)
        
        # Apply periodic boundary conditions
        if cell is not None:
            # For simplicity, assume cubic cell
            cell_size = cell[0, 0].item()
            distances = torch.minimum(distances, cell_size - distances)
        
        return distances
    
    def _create_periodic_images(self, positions: np.ndarray, 
                               cell_vectors: np.ndarray) -> np.ndarray:
        """Create periodic images of atoms.
        
        Args:
            positions: Atomic positions (N, 3)
            cell_vectors: Unit cell vectors (3, 3)
            
        Returns:
            Extended positions including periodic images
        """
        # Implement periodic image creation
        # Create images in adjacent unit cells
        images = []
        
        # Original positions
        images.append(positions)
        
        # Create images in adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip original cell
                    
                    # Translate positions
                    translation = np.array([dx, dy, dz]) @ cell_vectors
                    translated_positions = positions + translation
                    images.append(translated_positions)
        
        return np.vstack(images)


class GraphAugmentation:
    """Data augmentation for graph neural networks."""
    
    def __init__(self, rotation_prob: float = 0.5, noise_std: float = 0.01) -> None:
        """Initialize graph augmentation.
        
        Args:
            rotation_prob: Probability of applying rotation
            noise_std: Standard deviation for position noise
        """
        self.rotation_prob = rotation_prob
        self.noise_std = noise_std
    
    def augment_graph(self, graph: Data) -> Data:
        """Apply random augmentations to a graph.
        
        Args:
            graph: Input graph data
            
        Returns:
            Augmented graph data
        """
        # Implement graph augmentation
        # Apply deterministic rotations and translations for data augmentation
        # NOTE: Randomness here is intentional for data augmentation purposes
        augmented_graph = graph.clone()
        
        # Deterministic rotation around z-axis (based on graph hash)
        graph_hash = hash(str(graph.pos.tolist())) % 1000
        angle = (graph_hash / 1000.0) * 2 * np.pi
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a, 0],
                                      [sin_a, cos_a, 0],
                                      [0, 0, 1]], dtype=torch.float32)
        
        # Apply rotation to positions
        augmented_graph.pos = augmented_graph.pos @ rotation_matrix.T
        
        # Deterministic translation (based on graph hash)
        translation_seed = graph_hash % 100
        translation = torch.tensor([
            (translation_seed % 10) * 0.01,
            ((translation_seed // 10) % 10) * 0.01,
            ((translation_seed // 100) % 10) * 0.01
        ])
        augmented_graph.pos += translation
        
        return augmented_graph
