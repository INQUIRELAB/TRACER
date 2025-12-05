"""Uncertainty quantification for ensemble GNN models."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from gnn.model import SchNetWrapper
from dft_hybrid.data.jarvis_dft import create_jarvis_dataloader


class EnsembleUncertainty:
    """Ensemble-based uncertainty quantification for GNN models."""
    
    def __init__(self, model_class: type = SchNetWrapper, 
                 model_config: Optional[Dict] = None,
                 device: str = "cuda") -> None:
        """Initialize ensemble uncertainty estimator.
        
        Args:
            model_class: GNN model class to use
            model_config: Configuration for model initialization
            device: Device for computation
        """
        self.model_class = model_class
        self.model_config = model_config or {
            'hidden_channels': 256,
            'num_filters': 256,
            'num_interactions': 8,
            'num_gaussians': 64,
            'cutoff': 6.0,
            'max_num_neighbors': 64
        }
        self.device = device
        self.models = []
        self.ensemble_size = 0
        
    def load_ensemble(self, checkpoint_dir: str) -> None:
        """Load ensemble models from checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing ensemble checkpoints
        """
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Find all checkpoint files
        checkpoint_files = list(checkpoint_path.glob("ckpt_*.pt"))
        checkpoint_files.sort()  # Ensure consistent ordering
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        print(f"🔄 Loading {len(checkpoint_files)} ensemble models...")
        
        self.models = []
        for i, ckpt_file in enumerate(tqdm(checkpoint_files, desc="Loading models")):
            model = self.model_class(**self.model_config).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.models.append(model)
        
        self.ensemble_size = len(self.models)
        print(f"✅ Loaded {self.ensemble_size} ensemble models")
    
    def ensemble_predict(self, batch_data, return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions with uncertainty quantification.
        
        Args:
            batch_data: Input batch data (graph, targets tuple)
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary containing:
                - 'mean': Mean predictions across ensemble
                - 'variance': Variance across ensemble
                - 'std': Standard deviation across ensemble
                - 'individual': Individual predictions (if return_individual=True)
        """
        if self.ensemble_size == 0:
            raise ValueError("No models loaded. Call load_ensemble() first.")
        
        batch_graph, batch_targets = batch_data
        batch_graph = batch_graph.to(self.device)
        
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                # Forward pass
                pred_energies, pred_forces, pred_stress, _ = model(batch_graph)
                all_predictions.append(pred_energies)
        
        # Stack predictions: (ensemble_size, batch_size)
        predictions = torch.stack(all_predictions, dim=0)
        
        # Calculate ensemble statistics
        mean_pred = torch.mean(predictions, dim=0)
        variance_pred = torch.var(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        result = {
            'mean': mean_pred,
            'variance': variance_pred,
            'std': std_pred
        }
        
        if return_individual:
            result['individual'] = predictions
        
        return result
    
    def evaluate_uncertainty(self, test_loader, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate ensemble uncertainty on test dataset.
        
        Args:
            test_loader: Test data loader
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with uncertainty metrics
        """
        if self.ensemble_size == 0:
            raise ValueError("No models loaded. Call load_ensemble() first.")
        
        print(f"🔍 Evaluating uncertainty on {self.ensemble_size} ensemble models...")
        
        all_means = []
        all_vars = []
        all_targets = []
        all_num_atoms = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Uncertainty evaluation"):
                batch_graph, batch_targets = batch_data
                batch_graph = batch_graph.to(self.device)
                
                # Extract targets
                target_energies = batch_targets['energy'].to(self.device)
                
                # Get ensemble predictions
                ensemble_result = self.ensemble_predict(batch_data)
                
                # Get number of atoms per structure
                num_atoms = batch_graph.batch.bincount()
                
                all_means.append(ensemble_result['mean'].cpu())
                all_vars.append(ensemble_result['variance'].cpu())
                all_targets.append(target_energies.cpu())
                all_num_atoms.append(num_atoms.cpu())
                
                sample_count += len(target_energies)
                
                if num_samples and sample_count >= num_samples:
                    break
        
        # Concatenate all results
        all_means = torch.cat(all_means, dim=0)
        all_vars = torch.cat(all_vars, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_num_atoms = torch.cat(all_num_atoms, dim=0)
        
        # Calculate uncertainty metrics
        mae_mean = torch.mean(torch.abs(all_means - all_targets)).item()
        mae_per_atom = torch.mean(torch.abs(all_means - all_targets) / all_num_atoms.float()).item()
        
        # Uncertainty metrics
        mean_uncertainty = torch.mean(torch.sqrt(all_vars)).item()
        mean_uncertainty_per_atom = torch.mean(torch.sqrt(all_vars) / all_num_atoms.float()).item()
        
        # Calibration metrics (how well uncertainty correlates with error)
        errors = torch.abs(all_means - all_targets)
        uncertainty_correlation = torch.corrcoef(torch.stack([
            torch.sqrt(all_vars).flatten(),
            errors.flatten()
        ]))[0, 1].item()
        
        metrics = {
            'mae_mean': mae_mean,
            'mae_per_atom': mae_per_atom,
            'mean_uncertainty': mean_uncertainty,
            'mean_uncertainty_per_atom': mean_uncertainty_per_atom,
            'uncertainty_correlation': uncertainty_correlation,
            'num_samples': len(all_means),
            'ensemble_size': self.ensemble_size
        }
        
        return metrics


class EnsembleTrainer:
    """Enhanced trainer for ensemble GNN models with uncertainty quantification."""
    
    def __init__(self, model_class: type = SchNetWrapper,
                 model_config: Optional[Dict] = None,
                 device: str = "cuda") -> None:
        """Initialize ensemble trainer.
        
        Args:
            model_class: GNN model class
            model_config: Model configuration
            device: Device for training
        """
        self.model_class = model_class
        self.model_config = model_config or {
            'hidden_channels': 256,
            'num_filters': 256,
            'num_interactions': 8,
            'num_gaussians': 64,
            'cutoff': 6.0,
            'max_num_neighbors': 64
        }
        self.device = device
        
    def train_ensemble(self, train_loader, val_loader, 
                      ensemble_size: int = 5,
                      num_epochs: int = 50,
                      learning_rate: float = 1e-4,
                      save_dir: Optional[str] = None) -> str:
        """Train ensemble of models with different random seeds.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            ensemble_size: Number of models in ensemble
            num_epochs: Number of training epochs per model
            learning_rate: Learning rate for training
            save_dir: Directory to save ensemble checkpoints
            
        Returns:
            Path to ensemble checkpoint directory
        """
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"artifacts/gnn/ensemble_{timestamp}"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Training ensemble of {ensemble_size} models")
        print(f"📁 Saving to: {save_path}")
        print("=" * 60)
        
        # Import trainer here to avoid circular imports
        from gnn.train import GNNTrainer
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        for i in range(ensemble_size):
            print(f"\n🚀 Training ensemble member {i+1}/{ensemble_size}")
            print("-" * 40)
            
            # Set different seed for each ensemble member
            seed = 42 + i * 1000
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # Create model
            model = self.model_class(**self.model_config).to(self.device)
            
            # Create trainer
            trainer = GNNTrainer(
                model=model,
                device=self.device,
                w_e=1.0,
                w_f=100.0,
                w_s=10.0,
                use_gradient_clipping=True,
                max_grad_norm=1.0,
                use_early_stopping=True,
                patience=15
            )
            
            # Create optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)
            
            # Train model
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=num_epochs,
                save_dir=None,  # We'll save manually
                save_every=10
            )
            
            # Save ensemble member checkpoint
            ckpt_path = save_path / f"ckpt_{i}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ensemble_member': i,
                'seed': seed,
                'model_config': self.model_config,
                'best_val_loss': trainer.best_val_loss
            }, ckpt_path)
            
            print(f"✅ Ensemble member {i+1} saved to {ckpt_path}")
            print(f"   Best validation loss: {trainer.best_val_loss:.6f}")
        
        # Save ensemble metadata
        metadata = {
            'ensemble_size': ensemble_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'model_config': self.model_config,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_files': [f"ckpt_{i}.pt" for i in range(ensemble_size)]
        }
        
        with open(save_path / "ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 Ensemble training completed!")
        print(f"📁 Ensemble saved to: {save_path}")
        print(f"📊 Ensemble size: {ensemble_size}")
        
        return str(save_path)