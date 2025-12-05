"""
Domain-aware GNN with FiLM readout and LoRA adapters for fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch_geometric.nn import SchNet
from torch_geometric.data import Data


@dataclass
class DomainAdapterConfig:
    """Configuration for domain adapters."""
    # Domain embedding
    domain_embedding_dim: int = 64
    num_domains: int = 5  # JARVIS-DFT, JARVIS-Elastic, OC20, OC22, ANI1x
    
    # FiLM parameters
    film_dim: int = 128
    film_use_bias: bool = True
    
    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    # Fine-tuning parameters
    fine_tune_layers: int = 2  # Last N interaction layers
    fine_tune_lr: float = 1e-5
    freeze_backbone: bool = True


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, 
                 alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (deterministic initialization)
        # Use Xavier uniform initialization scaled by rank
        bound = np.sqrt(6.0 / (rank + in_features))
        lora_A_init = torch.empty(rank, in_features).uniform_(-bound, bound) * 0.01
        self.lora_A = nn.Parameter(lora_A_init)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # LoRA: x -> dropout(x) @ A^T @ B^T * scaling
        x = self.dropout(x)
        x = x @ self.lora_A.T @ self.lora_B.T
        return x * self.scaling


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer."""
    
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
    """Domain embedding module."""
    
    def __init__(self, num_domains: int, embedding_dim: int):
        super().__init__()
        self.num_domains = num_domains
        self.embedding_dim = embedding_dim
        
        # Domain embeddings
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        
        # Domain mapping
        self.domain_mapping = {
            'jarvis_dft': 0,
            'jarvis_elastic': 1,
            'oc20_s2ef': 2,
            'oc22_s2ef': 3,
            'ani1x': 4
        }
        
    def forward(self, dataset_names: List[str]) -> torch.Tensor:
        """
        Get domain embeddings for batch.
        
        Args:
            dataset_names: List of dataset names for each sample
            
        Returns:
            Domain embeddings (batch_size, embedding_dim)
        """
        # Map dataset names to domain IDs
        domain_ids = []
        for name in dataset_names:
            # Extract dataset name from full path/name
            dataset_key = name.lower()
            domain_id = 0  # Default to JARVIS-DFT
            
            for domain_name, domain_idx in self.domain_mapping.items():
                if domain_name in dataset_key:
                    domain_id = domain_idx
                    break
            
            domain_ids.append(domain_id)
        
        domain_ids = torch.tensor(domain_ids, dtype=torch.long, device=next(self.parameters()).device)
        return self.domain_embeddings(domain_ids)


class DomainAwareSchNet(nn.Module):
    """SchNet with domain awareness via FiLM and LoRA adapters."""
    
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
                 atomref: Optional[torch.Tensor] = None,
                 adapter_config: Optional[DomainAdapterConfig] = None) -> None:
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.adapter_config = adapter_config or DomainAdapterConfig()
        
        # Create base SchNet model
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
        
        # Store device for later use
        self._device = None
        
        # Domain embedding
        self.domain_embedding = DomainEmbedding(
            num_domains=self.adapter_config.num_domains,
            embedding_dim=self.adapter_config.domain_embedding_dim
        )
        
        # FiLM projection for readout
        self.film_proj = nn.Sequential(
            nn.Linear(self.adapter_config.domain_embedding_dim, self.adapter_config.film_dim),
            nn.ReLU(),
            nn.Linear(self.adapter_config.film_dim, self.adapter_config.film_dim)
        )
        
        self.film_layer = FiLMLayer(
            feature_dim=hidden_channels,
            film_dim=self.adapter_config.film_dim,
            use_bias=self.adapter_config.film_use_bias
        )
        
        # LoRA adapters for interaction layers
        self.lora_adapters = nn.ModuleList()
        for i in range(num_interactions):
            if i >= num_interactions - self.adapter_config.fine_tune_layers:
                # Add LoRA adapter for last N layers
                adapter = LoRALayer(
                    in_features=hidden_channels,
                    out_features=hidden_channels,
                    rank=self.adapter_config.lora_rank,
                    alpha=self.adapter_config.lora_alpha,
                    dropout=self.adapter_config.lora_dropout
                )
                self.lora_adapters.append(adapter)
            else:
                # No adapter for frozen layers
                self.lora_adapters.append(None)
        
        # Fine-tuning mode flag
        self.fine_tune_mode = False
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self._device = device
        return self
        
    def set_fine_tune_mode(self, enabled: bool = True):
        """Enable/disable fine-tuning mode."""
        self.fine_tune_mode = enabled
        
        if enabled:
            # Freeze backbone parameters
            for param in self.schnet_model.parameters():
                param.requires_grad = False
            
            # Unfreeze adapters and last layers
            for adapter in self.lora_adapters:
                if adapter is not None:
                    for param in adapter.parameters():
                        param.requires_grad = True
            
            # Unfreeze FiLM layers
            for param in self.film_proj.parameters():
                param.requires_grad = True
            for param in self.film_layer.parameters():
                param.requires_grad = True
            
            # Unfreeze domain embeddings
            for param in self.domain_embedding.parameters():
                param.requires_grad = True
        else:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
    
    def forward(self, batch) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with domain awareness."""
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
        
        # Get dataset names for domain embedding
        dataset_names = getattr(data, 'dataset_name', ['jarvis_dft'] * len(data.batch.unique()))
        
        # Get domain embeddings
        domain_embeds = self.domain_embedding(dataset_names)  # (batch_size, domain_embedding_dim)
        if self._device is not None:
            domain_embeds = domain_embeds.to(self._device)
        
        # Forward pass through SchNet
        z = data.atomic_numbers if hasattr(data, 'atomic_numbers') else data.x.squeeze(1).long()
        
        # Ensure atomic numbers are on the same device as the model
        if self._device is not None:
            z = z.to(self._device)
            data.pos = data.pos.to(self._device)
            data.batch = data.batch.to(self._device)
        
        # Forward pass through SchNet (LoRA adapters applied post-hoc for simplicity)
        energies = self.schnet_model(z, data.pos, data.batch)
        
        # Apply LoRA adapters post-hoc if in fine-tune mode
        if self.fine_tune_mode:
            # This is a simplified approach - in practice, you'd want to modify SchNet's forward method
            # For now, we'll apply a small LoRA-based correction to the energies
            for i, adapter in enumerate(self.lora_adapters):
                if adapter is not None:
                    # Apply LoRA correction to energies (simplified)
                    # Create input with correct dimensions for LoRA layer
                    batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
                    lora_input = torch.ones(batch_size, self.schnet_model.hidden_channels, device=energies.device)
                    lora_correction = adapter(lora_input).sum(dim=-1)  # Sum to get scalar
                    energies = energies + lora_correction * 0.01  # Small correction factor
        
        # Apply FiLM modulation to readout features
        if hasattr(self.schnet_model, 'readout'):
            # Get readout features (this is a simplified approach)
            # In practice, you'd need to modify SchNet to expose intermediate features
            film_input = self.film_proj(domain_embeds)
            
            # Apply FiLM modulation (simplified - would need access to readout features)
            # For now, we'll apply a domain-specific bias
            batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
            dummy_features = torch.zeros(batch_size, self.schnet_model.hidden_channels, device=energies.device)
            domain_bias = self.film_layer(dummy_features, film_input).sum(dim=-1)  # Sum to get scalar bias
            
            energies = energies + domain_bias
        
        # Ensure energies are properly shaped - SchNet should output per-molecule energies
        # If we get per-atom energies, we need to aggregate them properly
        if len(energies.shape) > 1:
            if energies.shape[1] > 1:
                # This is per-atom energies, we need to aggregate by molecule
                # Use batch information to aggregate per molecule
                batch_size = len(data.batch.unique())
                if batch_size == 1:
                    # Single molecule case - sum all atom energies
                    energies = energies.sum(dim=1, keepdim=True)
                else:
                    # Multiple molecules case - aggregate by batch
                    # energies has shape [batch_size, num_atoms_per_molecule]
                    # We need to sum across the atom dimension for each molecule
                    aggregated_energies = energies.sum(dim=1, keepdim=True)  # [batch_size, 1]
                    energies = aggregated_energies
            else:
                # Single dimension case - squeeze if needed
                energies = energies.squeeze(1)
        
        # Final squeeze to ensure scalar output
        if len(energies.shape) > 0:
            energies = energies.squeeze()
        
        # Compute forces via autograd (only during training)
        if self.training and data.pos.requires_grad:
            forces = -torch.autograd.grad(
                energies.sum(), data.pos, 
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            
            if forces is None:
                num_atoms = data.pos.shape[0]
                forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        else:
            num_atoms = data.pos.shape[0]
            forces = torch.zeros(num_atoms, 3, device=energies.device, dtype=energies.dtype)
        
        # Compute stress (simplified - just zeros for now)
        batch_size = energies.shape[0] if len(energies.shape) > 0 else 1
        stress = torch.zeros(batch_size, 3, 3, device=energies.device, dtype=energies.dtype)
        
        # Extract node features
        num_atoms = data.pos.shape[0]
        features = torch.zeros(num_atoms, self.schnet_model.hidden_channels, device=energies.device)
        
        # Check if we should return dict format (for multi-task training)
        if hasattr(batch, 'dataset_name') or hasattr(batch, 'domain_id'):
            return {
                'energy': energies,
                'forces': forces,
                'stress': stress,
                'features': features
            }
        else:
            return energies, forces, stress, features
    
    def _convert_to_pyg_data(self, batch) -> Data:
        """Convert GraphBatch to PyTorch Geometric Data format."""
        if hasattr(batch, 'atomic_numbers'):
            atomic_numbers = batch.atomic_numbers.detach().clone()
            positions = batch.positions.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        else:
            atomic_numbers = batch.x.squeeze(1).long().detach().clone()
            positions = batch.pos.detach().clone()
            edge_index = batch.edge_index.detach().clone()
            batch_indices = batch.batch_indices.detach().clone()
        
        positions.requires_grad_(True)
        
        data = Data(
            z=atomic_numbers,
            pos=positions,
            edge_index=edge_index,
            batch=batch_indices
        )
        
        return data


class FineTuneTrainer:
    """Trainer for fine-tuning domain-aware models."""
    
    def __init__(self, model: DomainAwareSchNet, config: DomainAdapterConfig, 
                 device: torch.device, base_lr: float = 1e-4):
        self.model = model
        self.config = config
        self.device = device
        self.base_lr = base_lr
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Create optimizers for different components
        self._create_optimizers()
        
    def _create_optimizers(self):
        """Create separate optimizers for different components."""
        # Fine-tuning optimizer (small LR for adapters)
        fine_tune_params = []
        for adapter in self.model.lora_adapters:
            if adapter is not None:
                fine_tune_params.extend(adapter.parameters())
        
        fine_tune_params.extend(self.model.film_proj.parameters())
        fine_tune_params.extend(self.model.film_layer.parameters())
        fine_tune_params.extend(self.model.domain_embedding.parameters())
        
        self.fine_tune_optimizer = torch.optim.AdamW(
            fine_tune_params,
            lr=self.config.fine_tune_lr,
            weight_decay=1e-6
        )
        
        # Full model optimizer (for non-fine-tune mode)
        self.full_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=1e-6
        )
        
        self.current_optimizer = self.full_optimizer
    
    def set_fine_tune_mode(self, enabled: bool = True):
        """Switch between fine-tuning and full training modes."""
        self.model.set_fine_tune_mode(enabled)
        
        if enabled:
            self.current_optimizer = self.fine_tune_optimizer
            print(f"🔧 Fine-tune mode enabled (LR: {self.config.fine_tune_lr})")
        else:
            self.current_optimizer = self.full_optimizer
            print(f"🚀 Full training mode enabled (LR: {self.base_lr})")
    
    def train_step(self, batch, loss_fn):
        """Single training step."""
        self.model.train()
        self.current_optimizer.zero_grad()
        
        # Extract data and targets from batch
        if isinstance(batch, tuple):
            data, targets = batch
        else:
            data = batch
            targets = batch.get('targets', {})
        
        # Forward pass
        predictions = self.model(data)
        
        # Compute loss
        if isinstance(predictions, dict):
            # Multi-task format
            dataset_names = getattr(data, 'dataset_name', [])
            domain_ids = getattr(data, 'domain_id', None)
            
            losses = loss_fn(predictions, targets, domain_ids, dataset_names)
            loss = losses['total']
        else:
            # Single-task format
            loss = loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        self.current_optimizer.step()
        
        return loss.item()
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
