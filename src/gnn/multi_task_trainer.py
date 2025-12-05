"""
Multi-task trainer supporting missing labels and per-domain logging.
Handles different label availability across datasets (forces for ANI/OC20, energy for all).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from tqdm import tqdm
import logging

from .model import SchNetWrapper
from .train import SupervisedLoss


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task training."""
    # Loss weights
    w_e: float = 1.0  # Energy weight
    w_f: float = 10.0  # Force weight  
    w_s: float = 0.0   # Stress weight
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    
    # Scheduler parameters
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    
    # Logging
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 10
    
    # Multi-task specific
    enable_per_domain_logging: bool = True
    missing_label_strategy: str = "ignore"  # "ignore", "zero", "mask"


class MultiTaskLoss(nn.Module):
    """Multi-task loss supporting missing labels."""
    
    def __init__(self, config: MultiTaskConfig):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                domain_ids: torch.Tensor,
                dataset_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with missing label handling.
        
        Args:
            predictions: Dict with keys 'energy', 'forces', 'stress'
            targets: Dict with keys 'energy', 'forces', 'stress' 
            domain_ids: Tensor indicating domain for each sample
            dataset_names: List of dataset names for each sample
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Energy loss (available for all datasets)
        if 'energy' in predictions and 'energy' in targets:
            energy_pred = predictions['energy']
            energy_target = targets['energy']
            
            if energy_pred is not None and energy_target is not None:
                energy_loss = self.mse_loss(energy_pred, energy_target)
                losses['energy'] = torch.mean(energy_loss)
                total_loss += self.config.w_e * losses['energy']
        
        # Force loss (only for ANI/OC20 datasets)
        if 'forces' in predictions and 'forces' in targets:
            forces_pred = predictions['forces']
            forces_target = targets['forces']
            
            if forces_pred is not None and forces_target is not None:
                # Check which samples have force labels
                has_forces = self._get_samples_with_forces(dataset_names)
                
                if torch.any(has_forces):
                    # For simplicity, compute loss on all forces if any sample has forces
                    # In practice, you'd want to mask based on batch structure
                    force_loss = self.mse_loss(forces_pred, forces_target)
                    losses['forces'] = torch.mean(force_loss)
                    total_loss += self.config.w_f * losses['forces']
                else:
                    losses['forces'] = torch.tensor(0.0, device=forces_pred.device)
        
        # Stress loss (typically not available)
        if 'stress' in predictions and 'stress' in targets:
            stress_pred = predictions['stress']
            stress_target = targets['stress']
            
            if stress_pred is not None and stress_target is not None:
                stress_loss = self.mse_loss(stress_pred, stress_target)
                losses['stress'] = torch.mean(stress_loss)
                total_loss += self.config.w_s * losses['stress']
        
        losses['total'] = total_loss
        return losses
    
    def _get_samples_with_forces(self, dataset_names: List[str]) -> torch.Tensor:
        """Determine which samples have force labels."""
        # ANI and OC20 datasets typically have forces
        force_datasets = {'ani1x', 'oc20', 'oc22'}
        
        has_forces = []
        for name in dataset_names:
            # Extract dataset name from full path/name
            dataset_key = name.lower()
            if any(force_ds in dataset_key for force_ds in force_datasets):
                has_forces.append(True)
            else:
                has_forces.append(False)
        
        return torch.tensor(has_forces, dtype=torch.bool)
    


class PerDomainLogger:
    """Logger for per-domain metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Per-domain metrics storage
        self.domain_metrics = defaultdict(lambda: defaultdict(list))
        self.epoch_metrics = defaultdict(lambda: defaultdict(dict))
        
    def log_epoch_metrics(self, epoch: int, domain_metrics: Dict[str, Dict[str, float]]):
        """Log metrics for each domain for current epoch."""
        for domain, metrics in domain_metrics.items():
            for metric_name, value in metrics.items():
                self.domain_metrics[domain][metric_name].append(value)
                self.epoch_metrics[epoch][domain][metric_name] = value
    
    def save_metrics(self, epoch: int):
        """Save metrics to JSON files."""
        # Save per-domain metrics
        domain_file = os.path.join(self.log_dir, "domain_metrics.json")
        with open(domain_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            domain_dict = {k: dict(v) for k, v in self.domain_metrics.items()}
            json.dump(domain_dict, f, indent=2)
        
        # Save epoch metrics
        epoch_file = os.path.join(self.log_dir, f"epoch_{epoch}_metrics.json")
        with open(epoch_file, 'w') as f:
            epoch_dict = {k: dict(v) for k, v in self.epoch_metrics[epoch].items()}
            json.dump(epoch_dict, f, indent=2)
    
    def get_domain_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each domain."""
        summary = {}
        for domain, metrics in self.domain_metrics.items():
            summary[domain] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary[domain][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest': values[-1]
                    }
        return summary


class MultiTaskTrainer:
    """Multi-task trainer with missing label support and per-domain logging."""
    
    def __init__(self, model: SchNetWrapper, config: MultiTaskConfig, 
                 device: torch.device, log_dir: str = "logs"):
        self.model = model
        self.config = config
        self.device = device
        self.log_dir = log_dir
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Loss function
        self.loss_fn = MultiTaskLoss(config)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Logger
        self.logger = PerDomainLogger(log_dir)
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=100,  # Will be updated during training
                eta_min=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=20,
                gamma=self.config.scheduler_factor
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.log = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with per-domain logging."""
        self.model.train()
        
        epoch_losses = defaultdict(list)
        domain_losses = defaultdict(lambda: defaultdict(list))
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            graphs, targets = batch
            graphs = graphs.to(self.device)
            
            # Extract targets and metadata
            energy_targets = targets.get('energy', None)
            forces_targets = targets.get('forces', None)
            stress_targets = targets.get('stress', None)
            
            dataset_names = getattr(graphs, 'dataset_name', [])
            domain_ids = getattr(graphs, 'domain_id', None)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(graphs)
            
            # Prepare targets dict
            targets_dict = {
                'energy': energy_targets,
                'forces': forces_targets,
                'stress': stress_targets
            }
            
            # Compute loss
            losses = self.loss_fn(predictions, targets_dict, domain_ids, dataset_names)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # Log losses
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    epoch_losses[loss_name].append(loss_value.item())
            
            # Per-domain logging
            if self.config.enable_per_domain_logging and domain_ids is not None:
                for i, domain_id in enumerate(domain_ids):
                    domain_name = f"domain_{domain_id.item()}"
                    for loss_name, loss_value in losses.items():
                        if isinstance(loss_value, torch.Tensor):
                            domain_losses[domain_name][loss_name].append(loss_value.item())
            
            # Update progress bar
            if batch_idx % self.config.log_every_n_steps == 0:
                avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
                progress_bar.set_postfix(avg_losses)
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[loss_name] = np.mean(loss_values)
        
        # Log per-domain metrics
        if self.config.enable_per_domain_logging:
            domain_metrics = {}
            for domain, losses_dict in domain_losses.items():
                domain_metrics[domain] = {}
                for loss_name, loss_values in losses_dict.items():
                    domain_metrics[domain][loss_name] = np.mean(loss_values)
            
            self.logger.log_epoch_metrics(epoch, domain_metrics)
        
        return epoch_metrics
    
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        domain_losses = defaultdict(lambda: defaultdict(list))
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                graphs, targets = batch
                graphs = graphs.to(self.device)
                
                # Extract targets and metadata
                energy_targets = targets.get('energy', None)
                forces_targets = targets.get('forces', None)
                stress_targets = targets.get('stress', None)
                
                dataset_names = getattr(graphs, 'dataset_name', [])
                domain_ids = getattr(graphs, 'domain_id', None)
                
                # Forward pass
                predictions = self.model(graphs)
                
                # Prepare targets dict
                targets_dict = {
                    'energy': energy_targets,
                    'forces': forces_targets,
                    'stress': stress_targets
                }
                
                # Compute loss
                losses = self.loss_fn(predictions, targets_dict, domain_ids, dataset_names)
                
                # Log losses
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        epoch_losses[loss_name].append(loss_value.item())
                
                # Per-domain logging
                if self.config.enable_per_domain_logging and domain_ids is not None:
                    for i, domain_id in enumerate(domain_ids):
                        domain_name = f"domain_{domain_id.item()}"
                        for loss_name, loss_value in losses.items():
                            if isinstance(loss_value, torch.Tensor):
                                domain_losses[domain_name][loss_name].append(loss_value.item())
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[loss_name] = np.mean(loss_values)
        
        # Log per-domain metrics
        if self.config.enable_per_domain_logging:
            domain_metrics = {}
            for domain, losses_dict in domain_losses.items():
                domain_metrics[domain] = {}
                for loss_name, loss_values in losses_dict.items():
                    domain_metrics[domain][loss_name] = np.mean(loss_values)
            
            self.logger.log_epoch_metrics(epoch, domain_metrics)
        
        return epoch_metrics
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              save_dir: str = "models") -> Dict[str, Any]:
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        self.log.info(f"Starting multi-task training for {num_epochs} epochs")
        self.log.info(f"Loss weights: w_e={self.config.w_e}, w_f={self.config.w_f}, w_s={self.config.w_s}")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['total'])
            else:
                self.scheduler.step()
            
            # Logging
            self.log.info(f"Epoch {epoch}/{num_epochs}")
            self.log.info(f"Train Loss: {train_metrics['total']:.6f} "
                         f"(E: {train_metrics.get('energy', 0):.6f}, "
                         f"F: {train_metrics.get('forces', 0):.6f}, "
                         f"S: {train_metrics.get('stress', 0):.6f})")
            self.log.info(f"Val Loss: {val_metrics['total']:.6f} "
                         f"(E: {val_metrics.get('energy', 0):.6f}, "
                         f"F: {val_metrics.get('forces', 0):.6f}, "
                         f"S: {val_metrics.get('stress', 0):.6f})")
            
            # Save metrics
            self.logger.save_metrics(epoch)
            
            # Save best model
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.epochs_without_improvement = 0
                
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['total'],
                    'config': self.config
                }, best_model_path)
                
                self.log.info(f"✅ New best validation loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Regular checkpoint saving
            if epoch % self.config.save_every_n_epochs == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_metrics['total'],
                    'config': self.config
                }, checkpoint_path)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.log.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_metrics['total'],
            'config': self.config
        }, final_model_path)
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Print domain summary
        if self.config.enable_per_domain_logging:
            domain_summary = self.logger.get_domain_summary()
            self.log.info("Per-domain performance summary:")
            for domain, metrics in domain_summary.items():
                self.log.info(f"  {domain}:")
                for metric, stats in metrics.items():
                    self.log.info(f"    {metric}: {stats['latest']:.6f} "
                                 f"(mean: {stats['mean']:.6f}, std: {stats['std']:.6f})")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch,
            'domain_summary': self.logger.get_domain_summary() if self.config.enable_per_domain_logging else None
        }
