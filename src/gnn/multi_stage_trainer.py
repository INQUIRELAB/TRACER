"""
Multi-stage training pipeline for domain-aware GNN with FiLM and LoRA.
Implements progressive training strategy for optimal domain adaptation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from tqdm import tqdm
import logging

from .domain_aware_model import DomainAwareSchNet, DomainAdapterConfig, FineTuneTrainer
from .multi_task_trainer import MultiTaskLoss, MultiTaskConfig


@dataclass
class TrainingStageConfig:
    """Configuration for each training stage."""
    name: str
    epochs: int
    learning_rate: float
    freeze_backbone: bool = False
    freeze_film: bool = False
    freeze_lora: bool = False
    freeze_domain_embeddings: bool = False
    
    # Stage-specific parameters
    domain_loss_weight: float = 1.0
    curriculum_learning: bool = False
    domain_specific_lr: bool = False


class DomainSpecificLoss(nn.Module):
    """Enhanced loss with domain-specific weighting and curriculum learning."""
    
    def __init__(self, config: MultiTaskConfig, domain_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Domain-specific weights
        self.domain_weights = domain_weights or {
            'jarvis_dft': 1.0,
            'jarvis_elastic': 1.0,
            'oc20_s2ef': 1.5,  # Higher weight for force-rich data
            'oc22_s2ef': 1.5,
            'ani1x': 1.2
        }
        
        # Curriculum learning parameters
        self.curriculum_enabled = False
        self.domain_difficulty = {
            'jarvis_dft': 0.3,      # Easy (small molecules)
            'jarvis_elastic': 0.4,  # Easy-Medium
            'ani1x': 0.6,          # Medium (organic molecules)
            'oc20_s2ef': 0.8,      # Hard (catalysts)
            'oc22_s2ef': 0.9       # Hardest (complex surfaces)
        }
    
    def enable_curriculum_learning(self, enabled: bool = True):
        """Enable/disable curriculum learning."""
        self.curriculum_enabled = enabled
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                domain_ids: torch.Tensor,
                dataset_names: List[str],
                epoch: int = 0,
                total_epochs: int = 100) -> Dict[str, torch.Tensor]:
        """
        Compute domain-aware loss with curriculum learning.
        """
        losses = {}
        total_loss = 0.0
        
        # Energy loss with domain weighting
        if 'energy' in predictions and 'energy' in targets:
            energy_pred = predictions['energy']
            energy_target = targets['energy']
            
            if energy_pred is not None and energy_target is not None:
                # Ensure targets are on the same device as predictions
                energy_target = energy_target.to(energy_pred.device)
                energy_loss = self.mse_loss(energy_pred, energy_target)
                
                # Apply domain-specific weighting
                domain_weights = self._get_domain_weights(dataset_names, epoch, total_epochs)
                weighted_loss = energy_loss * domain_weights
                
                losses['energy'] = torch.mean(weighted_loss)
                total_loss += self.config.w_e * losses['energy']
        
        # Force loss with domain weighting
        if 'forces' in predictions and 'forces' in targets:
            forces_pred = predictions['forces']
            forces_target = targets['forces']
            
            if forces_pred is not None and forces_target is not None:
                # Ensure targets are on the same device as predictions
                forces_target = forces_target.to(forces_pred.device)
                has_forces = self._get_samples_with_forces(dataset_names)
                
                if torch.any(has_forces):
                    force_loss = self.mse_loss(forces_pred, forces_target)
                    
                    # Apply domain-specific weighting (simplified for now)
                    # In practice, you'd need to map batch-level weights to atom-level
                    domain_weights = self._get_domain_weights(dataset_names, epoch, total_epochs)
                    
                    # For simplicity, use mean domain weight for all forces
                    mean_domain_weight = torch.mean(domain_weights)
                    weighted_loss = force_loss * mean_domain_weight
                    
                    losses['forces'] = torch.mean(weighted_loss)
                    total_loss += self.config.w_f * losses['forces']
                else:
                    losses['forces'] = torch.tensor(0.0, device=forces_pred.device)
        
        losses['total'] = total_loss
        return losses
    
    def _get_domain_weights(self, dataset_names: List[str], epoch: int, total_epochs: int) -> torch.Tensor:
        """Get domain-specific weights with curriculum learning."""
        weights = []
        
        for name in dataset_names:
            # Extract domain name
            domain_name = self._extract_domain_name(name)
            base_weight = self.domain_weights.get(domain_name, 1.0)
            
            if self.curriculum_enabled:
                # Apply curriculum learning
                difficulty = self.domain_difficulty.get(domain_name, 0.5)
                curriculum_factor = self._get_curriculum_factor(difficulty, epoch, total_epochs)
                weight = base_weight * curriculum_factor
            else:
                weight = base_weight
            
            weights.append(weight)
        
        return torch.tensor(weights, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def _extract_domain_name(self, dataset_name: str) -> str:
        """Extract domain name from dataset name."""
        dataset_key = dataset_name.lower()
        for domain in self.domain_weights.keys():
            if domain in dataset_key:
                return domain
        return 'jarvis_dft'  # Default
    
    def _get_curriculum_factor(self, difficulty: float, epoch: int, total_epochs: int) -> float:
        """Get curriculum learning factor based on difficulty and progress."""
        progress = epoch / total_epochs
        
        # Easy domains: start early, hard domains: start later
        if progress < difficulty:
            return 0.1  # Very low weight initially
        else:
            # Gradually increase weight
            ramp_factor = (progress - difficulty) / (1.0 - difficulty)
            return 0.1 + 0.9 * ramp_factor
    
    def _get_samples_with_forces(self, dataset_names: List[str]) -> torch.Tensor:
        """Determine which samples have force labels."""
        force_datasets = {'ani1x', 'oc20', 'oc22'}
        
        has_forces = []
        for name in dataset_names:
            dataset_key = name.lower()
            has_forces.append(any(force_ds in dataset_key for force_ds in force_datasets))
        
        return torch.tensor(has_forces, dtype=torch.bool)


class MultiStageTrainer:
    """Multi-stage trainer for domain-aware GNN with FiLM and LoRA."""
    
    def __init__(self, model: DomainAwareSchNet, config: DomainAdapterConfig, 
                 device: torch.device, base_lr: float = 1e-4):
        self.model = model
        self.config = config
        self.device = device
        self.base_lr = base_lr
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Training stages
        self.stages = self._create_training_stages()
        self.current_stage = 0
        
        # Optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        self._create_optimizers()
        
        # Training history
        self.training_history = defaultdict(list)
        
        # Setup logging
        self._setup_logging()
    
    def _create_training_stages(self) -> List[TrainingStageConfig]:
        """Create training stages configuration."""
        return [
            # Stage 1: Foundation Training
            TrainingStageConfig(
                name="foundation",
                epochs=30,
                learning_rate=1e-4,
                freeze_backbone=False,
                freeze_film=False,
                freeze_lora=False,
                freeze_domain_embeddings=False,
                domain_loss_weight=1.0,
                curriculum_learning=False
            ),
            
            # Stage 2: FiLM Specialization
            TrainingStageConfig(
                name="film_specialization",
                epochs=20,
                learning_rate=5e-5,
                freeze_backbone=True,
                freeze_film=False,
                freeze_lora=True,
                freeze_domain_embeddings=False,
                domain_loss_weight=2.0,
                curriculum_learning=True
            ),
            
            # Stage 3: LoRA Fine-Tuning
            TrainingStageConfig(
                name="lora_finetuning",
                epochs=15,
                learning_rate=1e-5,
                freeze_backbone=True,
                freeze_film=True,
                freeze_lora=False,
                freeze_domain_embeddings=False,
                domain_loss_weight=1.5,
                curriculum_learning=True
            ),
            
            # Stage 4: Joint Refinement
            TrainingStageConfig(
                name="joint_refinement",
                epochs=10,
                learning_rate=2e-5,
                freeze_backbone=True,
                freeze_film=False,
                freeze_lora=False,
                freeze_domain_embeddings=False,
                domain_loss_weight=1.0,
                curriculum_learning=False
            )
        ]
    
    def _create_optimizers(self):
        """Create optimizers for each stage."""
        for i, stage in enumerate(self.stages):
            # Get trainable parameters for this stage
            trainable_params = self._get_trainable_parameters(stage)
            
            # Create optimizer
            optimizer = optim.AdamW(
                trainable_params,
                lr=stage.learning_rate,
                weight_decay=1e-6
            )
            
            # Create scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
            
            self.optimizers[i] = optimizer
            self.schedulers[i] = scheduler
    
    def _get_trainable_parameters(self, stage: TrainingStageConfig) -> List[torch.nn.Parameter]:
        """Get trainable parameters for a specific stage."""
        trainable_params = []
        
        # SchNet backbone
        if not stage.freeze_backbone:
            trainable_params.extend(self.model.schnet_model.parameters())
        
        # Domain embeddings
        if not stage.freeze_domain_embeddings:
            trainable_params.extend(self.model.domain_embedding.parameters())
        
        # FiLM layers
        if not stage.freeze_film:
            trainable_params.extend(self.model.film_proj.parameters())
            trainable_params.extend(self.model.film_layer.parameters())
        
        # LoRA adapters
        if not stage.freeze_lora:
            for adapter in self.model.lora_adapters:
                if adapter is not None:
                    trainable_params.extend(adapter.parameters())
        
        return trainable_params
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/multi_stage_training.log'),
                logging.StreamHandler()
            ]
        )
        self.log = logging.getLogger(__name__)
    
    def train_stage(self, stage_idx: int, train_loader, val_loader, 
                   loss_fn: DomainSpecificLoss) -> Dict[str, float]:
        """Train a specific stage."""
        stage = self.stages[stage_idx]
        optimizer = self.optimizers[stage_idx]
        scheduler = self.schedulers[stage_idx]
        
        self.log.info(f"🚀 Starting Stage {stage_idx + 1}: {stage.name}")
        self.log.info(f"Epochs: {stage.epochs}, LR: {stage.learning_rate}")
        self.log.info(f"Freeze - Backbone: {stage.freeze_backbone}, FiLM: {stage.freeze_film}, LoRA: {stage.freeze_lora}")
        
        # Set model to appropriate mode
        self._set_model_mode(stage)
        
        # Enable curriculum learning if specified
        if stage.curriculum_learning:
            loss_fn.enable_curriculum_learning(True)
            self.log.info("📚 Curriculum learning enabled")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, stage.epochs + 1):
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, loss_fn, epoch, stage.epochs)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader, loss_fn, epoch, stage.epochs)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total'])
            
            # Logging
            self.log.info(f"Stage {stage_idx + 1} - Epoch {epoch}/{stage.epochs}")
            self.log.info(f"Train Loss: {train_metrics['total']:.6f}")
            self.log.info(f"Val Loss: {val_metrics['total']:.6f}")
            
            # Save best model for this stage
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                self._save_stage_checkpoint(stage_idx, epoch, val_metrics['total'])
            
            # Store metrics
            self.training_history[f'stage_{stage_idx}_train'].append(train_metrics)
            self.training_history[f'stage_{stage_idx}_val'].append(val_metrics)
        
        self.log.info(f"✅ Stage {stage_idx + 1} completed. Best val loss: {best_val_loss:.6f}")
        return {'best_val_loss': best_val_loss}
    
    def _set_model_mode(self, stage: TrainingStageConfig):
        """Set model parameters to appropriate trainable/frozen state."""
        # Freeze/unfreeze parameters based on stage
        for param in self.model.schnet_model.parameters():
            param.requires_grad = not stage.freeze_backbone
        
        for param in self.model.domain_embedding.parameters():
            param.requires_grad = not stage.freeze_domain_embeddings
        
        for param in self.model.film_proj.parameters():
            param.requires_grad = not stage.freeze_film
        
        for param in self.model.film_layer.parameters():
            param.requires_grad = not stage.freeze_film
        
        for adapter in self.model.lora_adapters:
            if adapter is not None:
                for param in adapter.parameters():
                    param.requires_grad = not stage.freeze_lora
    
    def _train_epoch(self, train_loader, optimizer, loss_fn, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = defaultdict(list)
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            # Extract data and targets
            if isinstance(batch, tuple):
                data, targets = batch
            elif isinstance(batch, list):
                # Handle list format from dataloader
                data, targets = batch[0], batch[1] if len(batch) > 1 else {}
            else:
                data = batch
                targets = batch.get('targets', {})
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(data)
            
            # Compute loss
            dataset_names = getattr(data, 'dataset_name', [])
            domain_ids = getattr(data, 'domain_id', None)
            
            losses = loss_fn(predictions, targets, domain_ids, dataset_names, epoch, total_epochs)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            # Store losses
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    epoch_losses[loss_name].append(loss_value.item())
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[loss_name] = np.mean(loss_values)
        
        return epoch_metrics
    
    def _validate_epoch(self, val_loader, loss_fn, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                # Extract data and targets
                if isinstance(batch, tuple):
                    data, targets = batch
                elif isinstance(batch, list):
                    # Handle list format from dataloader
                    data, targets = batch[0], batch[1] if len(batch) > 1 else {}
                else:
                    data = batch
                    targets = batch.get('targets', {})
                
                # Forward pass
                predictions = self.model(data)
                
                # Compute loss
                dataset_names = getattr(data, 'dataset_name', [])
                domain_ids = getattr(data, 'domain_id', None)
                
                losses = loss_fn(predictions, targets, domain_ids, dataset_names, epoch, total_epochs)
                
                # Store losses
                for loss_name, loss_value in losses.items():
                    if isinstance(loss_value, torch.Tensor):
                        epoch_losses[loss_name].append(loss_value.item())
        
        # Compute epoch averages
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            epoch_metrics[loss_name] = np.mean(loss_values)
        
        return epoch_metrics
    
    def _save_stage_checkpoint(self, stage_idx: int, epoch: int, val_loss: float):
        """Save checkpoint for a specific stage."""
        os.makedirs('models/multi_stage', exist_ok=True)
        
        checkpoint = {
            'stage_idx': stage_idx,
            'stage_name': self.stages[stage_idx].name,
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizers[stage_idx].state_dict(),
            'scheduler_state_dict': self.schedulers[stage_idx].state_dict(),
            'config': self.config
        }
        
        checkpoint_path = f'models/multi_stage/stage_{stage_idx}_{self.stages[stage_idx].name}_best.pt'
        torch.save(checkpoint, checkpoint_path)
    
    def train_all_stages(self, train_loader, val_loader, 
                        multi_task_config: MultiTaskConfig) -> Dict[str, Any]:
        """Train all stages sequentially."""
        self.log.info("🎯 Starting Multi-Stage Training Pipeline")
        
        # Create domain-specific loss
        loss_fn = DomainSpecificLoss(multi_task_config)
        
        stage_results = {}
        
        for stage_idx in range(len(self.stages)):
            stage_result = self.train_stage(stage_idx, train_loader, val_loader, loss_fn)
            stage_results[f'stage_{stage_idx}'] = stage_result
            
            # Save final model after each stage
            self._save_stage_checkpoint(stage_idx, self.stages[stage_idx].epochs, stage_result['best_val_loss'])
        
        # Save final model
        final_model_path = 'models/multi_stage/final_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'stage_results': stage_results,
            'training_history': dict(self.training_history)
        }, final_model_path)
        
        self.log.info("🎉 Multi-stage training completed!")
        return stage_results
    
    def get_parameter_summary(self) -> Dict[str, Dict[str, int]]:
        """Get parameter summary for each stage."""
        summary = {}
        
        for i, stage in enumerate(self.stages):
            trainable_params = self._get_trainable_parameters(stage)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_count = sum(p.numel() for p in trainable_params)
            
            summary[f'stage_{i}_{stage.name}'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_count,
                'frozen_parameters': total_params - trainable_count,
                'trainable_ratio': trainable_count / total_params
            }
        
        return summary
