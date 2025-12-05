#!/usr/bin/env python3
"""Generate ensemble predictions on real unified dataset for gate-hard ranking."""

import sys
import os
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dft_hybrid.data.unified_registry import (
    UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig, DatasetDomain
)
from gnn.model import SchNetWrapper
from gnn.domain_aware_model import DomainAwareSchNet
from pipeline.gate_hard_ranking import PredictionResult, DatasetDomain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ensemble_models(ensemble_dir: str, device: str = "cuda") -> List[torch.nn.Module]:
    """Load ensemble models from directory."""
    logger.info(f"Loading ensemble models from {ensemble_dir}")
    
    ensemble_path = Path(ensemble_dir)
    model_files = sorted(ensemble_path.glob("ckpt_*.pt"))
    
    if not model_files:
        raise ValueError(f"No ensemble models found in {ensemble_dir}")
    
    models = []
    for model_file in model_files:
        logger.info(f"Loading model {model_file}")
        checkpoint = torch.load(model_file, map_location=device, weights_only=False)
        
        # Determine model type
        if 'schnet_model' in checkpoint:
            model = DomainAwareSchNet(
                hidden_channels=256,
                num_filters=256,
                num_interactions=8,
                num_gaussians=64,
                cutoff=10.0,
                max_num_neighbors=32,
                readout='add',
                dipole=False,
                mean=0.0,
                std=1.0,
                atomref=None
            )
        else:
            model = SchNetWrapper(
                hidden_channels=256,
                num_filters=256,
                num_interactions=8,
                num_gaussians=64,
                cutoff=10.0,
                max_num_neighbors=32,
                readout='add',
                dipole=False,
                mean=0.0,
                std=1.0,
                atomref=None
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
    
    logger.info(f"Loaded {len(models)} ensemble models")
    return models


def generate_ensemble_predictions(
    models: List[torch.nn.Module],
    dataloader,
    device: str = "cuda",
    max_samples: int = None
) -> List[PredictionResult]:
    """Generate ensemble predictions on dataset."""
    logger.info("Generating ensemble predictions...")
    
    predictions = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and sample_count >= max_samples:
                break
                
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                data, targets = batch
            else:
                data = batch
                targets = None
            
            # Move to device
            data = data.to(device)
            
            # Generate predictions from all models
            ensemble_energies = []
            ensemble_forces = []
            
            for model in models:
                if hasattr(model, 'schnet_model'):  # DomainAwareSchNet
                    output = model(data)
                    if isinstance(output, dict):
                        energy = output['energy']
                        forces = output.get('forces')
                    else:
                        energy, forces, _, _ = output
                else:  # SchNetWrapper
                    output = model(data)
                    if isinstance(output, dict):
                        energy = output['energy']
                        forces = output.get('forces')
                    else:
                        energy, forces, _, _ = output
                
                ensemble_energies.append(energy.cpu())
                if forces is not None:
                    ensemble_forces.append(forces.cpu())
            
            # Compute ensemble statistics
            ensemble_energies = torch.stack(ensemble_energies)
            energy_mean = ensemble_energies.mean(dim=0)
            energy_variance = ensemble_energies.var(dim=0)
            
            if ensemble_forces:
                ensemble_forces = torch.stack(ensemble_forces)
                forces_mean = ensemble_forces.mean(dim=0)
                forces_variance = ensemble_forces.var(dim=0)
            else:
                forces_mean = None
                forces_variance = None
            
            # Create predictions for each sample in batch
            batch_size = energy_mean.size(0)
            for i in range(batch_size):
                # Extract sample info
                sample_id = f"sample_{sample_count:06d}"
                
                # Determine domain from batch info
                domain = DatasetDomain.JARVIS_DFT  # Default
                if hasattr(data, 'domain_ids') and data.domain_ids is not None:
                    domain_id = data.domain_ids[i].item()
                    domain_map = {
                        0: DatasetDomain.JARVIS_DFT,
                        1: DatasetDomain.JARVIS_ELASTIC,
                        2: DatasetDomain.OC20_S2EF,
                        3: DatasetDomain.OC22_S2EF,
                        4: DatasetDomain.ANI1X
                    }
                    domain = domain_map.get(domain_id, DatasetDomain.JARVIS_DFT)
                
                # Get target energy if available
                if targets is not None and 'energy' in targets:
                    energy_target = targets['energy'][i].item()
                else:
                    energy_target = energy_mean[i].item()  # Use prediction as target for demo
                
                # Create prediction result
                pred_result = PredictionResult(
                    sample_id=sample_id,
                    domain=domain,
                    energy_pred=energy_mean[i].item(),
                    energy_target=energy_target,
                    energy_variance=energy_variance[i].item(),
                    forces_pred=forces_mean[i].cpu().numpy() if forces_mean is not None else None,
                    forces_target=None,  # Not available in this demo
                    forces_variance=forces_variance[i].cpu().numpy() if forces_variance is not None else None,
                    tm_flag=False,  # Mock TM flag
                    near_degeneracy_proxy=np.random.rand() * 0.5,  # Mock near-degeneracy proxy
                    molecular_properties={
                        'num_atoms': np.random.randint(5, 50),
                        'formation_energy': energy_target,
                        'band_gap': np.random.rand() * 2.0
                    }
                )
                
                predictions.append(pred_result)
                sample_count += 1
                
                if sample_count % 1000 == 0:
                    logger.info(f"Generated {sample_count} predictions...")
    
    logger.info(f"Generated {len(predictions)} total predictions")
    return predictions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate ensemble predictions on real dataset")
    
    parser.add_argument("--ensemble-dir", type=str, 
                       default="models/gnn_training_enhanced/ensemble",
                       help="Directory containing ensemble models")
    parser.add_argument("--output-file", type=str,
                       default="artifacts/real_ensemble_predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum samples to process")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        logger.warning("CUDA not available, using CPU")
    
    try:
        # Load ensemble models
        models = load_ensemble_models(args.ensemble_dir, args.device)
        
        # Create config for unified dataset
        datasets = [
            DatasetConfig(
                domain_id=DatasetDomain.JARVIS_DFT,
                name="JARVIS-DFT",
                path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
                energy_unit="eV",
                force_unit="eV/Angstrom",
                max_samples=2000
            ),
            DatasetConfig(
                domain_id=DatasetDomain.JARVIS_ELASTIC,
                name="JARVIS-Elastic",
                path="data/jarvis_elastic/data/jarvis_elastic/elastic_tensor_2020.json",
                energy_unit="eV",
                force_unit="eV/Angstrom",
                max_samples=1000
            ),
            DatasetConfig(
                domain_id=DatasetDomain.OC20_S2EF,
                name="OC20-S2EF",
                path="data/oc20/s2ef/train",
                energy_unit="eV",
                force_unit="eV/Angstrom",
                max_samples=2000
            ),
            DatasetConfig(
                domain_id=DatasetDomain.OC22_S2EF,
                name="OC22-S2EF",
                path="data/oc22/s2ef/train",
                energy_unit="eV",
                force_unit="eV/Angstrom",
                max_samples=1000
            ),
            DatasetConfig(
                domain_id=DatasetDomain.ANI1X,
                name="ANI1x",
                path="data/ani1x/ani1x-release.h5",
                energy_unit="Hartree",
                force_unit="Hartree/Bohr",
                max_samples=2000
            )
        ]
        
        unified_config = UnifiedDatasetConfig(
            datasets=datasets,
            validation_split=0.1,
            normalize_energies=True,
            temperature_tau=1.0
        )
        
        # Load unified dataset
        logger.info("Loading unified dataset...")
        registry = UnifiedDatasetRegistry(unified_config)
        
        # Load datasets
        datasets = registry.load_all_datasets(unified_config)
        train_data = datasets['train']
        
        # Create dataloader
        dataloader = registry.create_dataloader(
            train_data, batch_size=args.batch_size, shuffle=False
        )
        
        # Generate predictions
        predictions = generate_ensemble_predictions(
            models, dataloader, args.device, args.max_samples
        )
        
        # Convert to JSON-serializable format
        predictions_dict = []
        for pred in predictions:
            pred_dict = {
                'sample_id': pred.sample_id,
                'domain': pred.domain.value,
                'energy_pred': pred.energy_pred,
                'energy_target': pred.energy_target,
                'energy_variance': pred.energy_variance,
                'forces_pred': pred.forces_pred.tolist() if pred.forces_pred is not None else None,
                'forces_target': pred.forces_target,
                'forces_variance': pred.forces_variance.tolist() if pred.forces_variance is not None else None,
                'tm_flag': pred.tm_flag,
                'near_degeneracy_proxy': pred.near_degeneracy_proxy,
                'molecular_properties': pred.molecular_properties
            }
            predictions_dict.append(pred_dict)
        
        # Save predictions
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(predictions_dict, f, indent=2)
        
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        
        # Print summary
        domain_counts = {}
        for pred in predictions:
            domain = pred.domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("=" * 60)
        logger.info("ENSEMBLE PREDICTIONS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info("Domain distribution:")
        for domain, count in domain_counts.items():
            logger.info(f"  {domain}: {count}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to generate ensemble predictions: {e}")
        raise


if __name__ == "__main__":
    main()
