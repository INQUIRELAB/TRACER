"""Training utilities for Graph Neural Networks."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from graphs.periodic_graph import GraphBatch


class SupervisedLoss(nn.Module):
    """Supervised loss function for energy, forces, and stress prediction."""
    
    def __init__(self, w_e: float = 1.0, w_f: float = 100.0, w_s: float = 10.0) -> None:
        """Initialize supervised loss.
        
        Args:
            w_e: Weight for energy loss
            w_f: Weight for force loss (should be >> w_e)
            w_s: Weight for stress loss
        """
        super().__init__()
        self.w_e = w_e
        self.w_f = w_f
        self.w_s = w_s
        
    def forward(self, 
                pred_energies: torch.Tensor, pred_forces: torch.Tensor, pred_stress: torch.Tensor,
                target_energies: torch.Tensor, target_forces: torch.Tensor, target_stress: torch.Tensor,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """Compute supervised loss.
        
        Args:
            pred_energies: Predicted energies (batch_size,)
            pred_forces: Predicted forces (total_atoms, 3)
            pred_stress: Predicted stress tensors (batch_size, 3, 3)
            target_energies: Target energies (batch_size,)
            target_forces: Target forces (total_atoms, 3)
            target_stress: Target stress tensors (batch_size, 3, 3)
            training: Whether in training mode (not used for SchNet)
            
        Returns:
            Dictionary with individual and total losses
        """
        # Energy loss (MSE)
        energy_loss = F.mse_loss(pred_energies, target_energies)
        
        # Force loss (MSE)
        force_loss = F.mse_loss(pred_forces, target_forces)
        
        # Stress loss (MSE)
        stress_loss = F.mse_loss(pred_stress, target_stress)
        
        # Total loss
        total_loss = self.w_e * energy_loss + self.w_f * force_loss + self.w_s * stress_loss
        
        return {
            'total_loss': total_loss,
            'energy_loss': energy_loss,
            'force_loss': force_loss,
            'stress_loss': stress_loss,
        }


class GNNTrainer:
    """Trainer for Graph Neural Network models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda", 
                 w_e: float = 1.0, w_f: float = 100.0, w_s: float = 10.0,
                 use_gradient_clipping: bool = True, max_grad_norm: float = 1.0,
                 use_early_stopping: bool = True, patience: int = 10) -> None:
        """Initialize GNN trainer.
        
        Args:
            model: GNN model to train
            device: Device for training (cuda/cpu)
            w_e: Weight for energy loss
            w_f: Weight for force loss (should be >> w_e)
            w_s: Weight for stress loss
            use_gradient_clipping: Whether to use gradient clipping
            max_grad_norm: Maximum gradient norm for clipping
            use_early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = SupervisedLoss(w_e=w_e, w_f=w_f, w_s=w_s)
        self.best_val_loss = float('inf')
        self.use_gradient_clipping = use_gradient_clipping
        self.max_grad_norm = max_grad_norm
        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.early_stop_counter = 0
        
    def train_epoch(self, dataloader: DataLoader, optimizer: Optimizer) -> Dict[str, float]:
        """Train model for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer for training
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        energy_loss = 0.0
        force_loss = 0.0
        stress_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_data in progress_bar:
            # Handle different batch formats
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                # LMDB dataset format: (batched_graph, batched_targets)
                batch_graph, batch_targets = batch_data
                batch_graph = batch_graph.to(self.device)
                
                # Extract targets
                target_energies = batch_targets['energy'].to(self.device)
                target_forces = batch_targets['forces'].to(self.device)
                target_stress = batch_targets['stress'].to(self.device)
                
            elif isinstance(batch_data, GraphBatch):
                # Original GraphBatch format
                batch_graph = batch_data.to(self.device)
                
                # Extract targets from batch (assuming they're stored as attributes)
                target_energies = getattr(batch_graph, 'target_energies', torch.zeros(batch_graph.num_nodes.shape[0], device=self.device))
                target_forces = getattr(batch_graph, 'target_forces', torch.zeros(batch_graph.positions.shape, device=self.device))
                target_stress = getattr(batch_graph, 'target_stress', torch.zeros(batch_graph.num_nodes.shape[0], 3, 3, device=self.device))
            else:
                # Skip unknown formats
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_energies, pred_forces, pred_stress, _ = self.model(batch_graph)
            
            # Compute loss
            losses = self.loss_fn(
                pred_energies, pred_forces, pred_stress,
                target_energies, target_forces, target_stress
            )
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping for training stability
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            energy_loss += losses['energy_loss'].item()
            force_loss += losses['force_loss'].item()
            stress_loss += losses['stress_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'E': f"{losses['energy_loss'].item():.4f}",
                'F': f"{losses['force_loss'].item():.4f}",
                'S': f"{losses['stress_loss'].item():.4f}"
            })
        
        return {
            'train_loss': total_loss / max(num_batches, 1),
            'train_energy_loss': energy_loss / max(num_batches, 1),
            'train_force_loss': force_loss / max(num_batches, 1),
            'train_stress_loss': stress_loss / max(num_batches, 1),
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        energy_loss = 0.0
        force_loss = 0.0
        stress_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            for batch_data in progress_bar:
                # Handle different batch formats
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    # LMDB dataset format: (batched_graph, batched_targets)
                    batch_graph, batch_targets = batch_data
                    batch_graph = batch_graph.to(self.device)
                    
                    # Extract targets
                    target_energies = batch_targets['energy'].to(self.device)
                    target_forces = batch_targets['forces'].to(self.device)
                    target_stress = batch_targets['stress'].to(self.device)
                    
                elif isinstance(batch_data, GraphBatch):
                    # Original GraphBatch format
                    batch_graph = batch_data.to(self.device)
                    
                    # Extract targets from batch
                    target_energies = getattr(batch_graph, 'target_energies', torch.zeros(batch_graph.num_nodes.shape[0], device=self.device))
                    target_forces = getattr(batch_graph, 'target_forces', torch.zeros(batch_graph.positions.shape, device=self.device))
                    target_stress = getattr(batch_graph, 'target_stress', torch.zeros(batch_graph.num_nodes.shape[0], 3, 3, device=self.device))
                else:
                    continue
                
                # Forward pass
                pred_energies, pred_forces, pred_stress, _ = self.model(batch_graph)
                
                # Compute loss
                losses = self.loss_fn(
                    pred_energies, pred_forces, pred_stress,
                    target_energies, target_forces, target_stress
                )
                
                # Accumulate losses
                total_loss += losses['total_loss'].item()
                energy_loss += losses['energy_loss'].item()
                force_loss += losses['force_loss'].item()
                stress_loss += losses['stress_loss'].item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{losses['total_loss'].item():.4f}",
                    'E': f"{losses['energy_loss'].item():.4f}",
                    'F': f"{losses['force_loss'].item():.4f}",
                    'S': f"{losses['stress_loss'].item():.4f}"
                })
        
        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_energy_loss': energy_loss / max(num_batches, 1),
            'val_force_loss': force_loss / max(num_batches, 1),
            'val_stress_loss': stress_loss / max(num_batches, 1),
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              optimizer: Optional[Optimizer] = None, 
              scheduler: Optional[object] = None,
              num_epochs: int = 100,
              save_dir: Optional[Union[str, Path]] = None,
              save_every: int = 10) -> List[Dict[str, float]]:
        """Full training loop with advanced optimizations.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (if None, uses AdamW)
            scheduler: Learning rate scheduler (if None, uses ReduceLROnPlateau)
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            List of epoch metrics
        """
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        if scheduler is None:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_history = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            metrics_history.append(epoch_metrics)
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['train_loss']:.6f} "
                  f"(E: {train_metrics['train_energy_loss']:.6f}, "
                  f"F: {train_metrics['train_force_loss']:.6f}, "
                  f"S: {train_metrics['train_stress_loss']:.6f})")
            print(f"Val Loss: {val_metrics['val_loss']:.6f} "
                  f"(E: {val_metrics['val_energy_loss']:.6f}, "
                  f"F: {val_metrics['val_force_loss']:.6f}, "
                  f"S: {val_metrics['val_stress_loss']:.6f})")
            
            # Learning rate scheduling
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
            
            # Save checkpoint if validation loss improved
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.early_stop_counter = 0  # Reset early stopping counter
                if save_dir is not None:
                    self.save_checkpoint(save_dir / "best_model.pt", epoch, optimizer, epoch_metrics)
                print(f"✅ New best validation loss: {self.best_val_loss:.6f}")
            else:
                self.early_stop_counter += 1
                print(f"⏳ No improvement for {self.early_stop_counter} epochs")
            
            # Early stopping
            if self.use_early_stopping and self.early_stop_counter >= self.patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                break
            
            # Save periodic checkpoints
            if save_dir is not None and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch+1}.pt", epoch, optimizer, epoch_metrics)
        
        # Save final model
        if save_dir is not None:
            self.save_checkpoint(save_dir / "final_model.pt", num_epochs-1, optimizer, metrics_history[-1])
            
            # Save training history
            with open(save_dir / "training_history.json", 'w') as f:
                json.dump(metrics_history, f, indent=2)
        
        return metrics_history
    
    def save_checkpoint(self, filepath: Union[str, Path], epoch: int, 
                       optimizer: Optimizer, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            metrics: Training metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, float]:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Loaded metrics
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['metrics']


class EnsembleTrainer:
    """Train ensemble of GNN models for uncertainty estimation."""
    
    def __init__(self, model_class: type, model_configs: List[Dict],
                 device: str = "cuda") -> None:
        """Initialize ensemble trainer.
        
        Args:
            model_class: GNN model class
            model_configs: List of model configurations
            device: Device for training
        """
        # Implement ensemble training with multiple configurations
        self.model_class = model_class
        self.model_configs = model_configs
        self.device = device
    
    def train_ensemble(self, train_loader: DataLoader, val_loader: DataLoader,
                      num_epochs: int) -> List[nn.Module]:
        """Train ensemble of models.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            List of trained models
        """
        # Implement ensemble training with multiple configurations
        trained_models = []
        
        for i, config in enumerate(self.model_configs):
            logger.info(f"Training ensemble model {i + 1}/{len(self.model_configs)}")
            
            # Create model instance
            model = self.model_class(**config).to(self.device)
            
            # Create trainer for this model
            trainer = GNNTrainer(
                model=model,
                device=self.device,
                learning_rate=config.get('learning_rate', 1e-3),
                weight_decay=config.get('weight_decay', 1e-6),
                max_epochs=num_epochs
            )
            
            # Train model
            trainer.train(train_loader, val_loader)
            
            trained_models.append(model)
        
        return trained_models
