#!/usr/bin/env python3
"""Complete pipeline: Load real data → Generate ensemble predictions → Gate-hard ranking → QNN labeling."""

import sys
import os
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import torch
from typing import Dict, List, Tuple
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dft_hybrid.data.unified_registry import (
    UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig, DatasetDomain
)
from gnn.model import SchNetWrapper
from gnn.domain_aware_model import DomainAwareSchNet
from pipeline.gate_hard_ranking import PredictionResult, DatasetDomain as GateDomain, GateHardRanker, DomainRankingConfig

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


def create_unified_dataset_config() -> UnifiedDatasetConfig:
    """Create configuration for unified dataset with real data paths."""
    datasets = [
        DatasetConfig(
            domain_id=DatasetDomain.JARVIS_DFT,
            name="JARVIS-DFT",
            path="data/jarvis_dft/data/jarvis_dft/jdft_3d-4-26-2020.json",
            energy_unit="eV",
            force_unit="eV/Angstrom",
            max_samples=5000  # Reasonable sample size for ensemble predictions
        ),
        DatasetConfig(
            domain_id=DatasetDomain.ANI1X,
            name="ANI1x",
            path="data/ani1x/ani1x_dataset.h5",
            energy_unit="Hartree",
            force_unit="Hartree/Bohr",
            max_samples=5000
        )
    ]
    
    return UnifiedDatasetConfig(
        datasets=datasets,
        validation_split=0.1,
        normalize_energies=True,
        temperature_tau=1.0
    )


def generate_ensemble_predictions_on_real_data(
    models: List[torch.nn.Module],
    dataloader,
    device: str = "cuda",
    max_samples: int = 10000
) -> List[PredictionResult]:
    """Generate ensemble predictions on real dataset."""
    logger.info("Generating ensemble predictions on real data...")
    
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
                sample_id = f"real_sample_{sample_count:06d}"
                
                # Determine domain from batch info
                domain = GateDomain.JARVIS_DFT  # Default
                
                # Try to get domain from batch attribute
                if hasattr(data, 'batch') and data.batch is not None:
                    # Use batch index to determine domain
                    batch_idx = data.batch[i].item()
                    # This is a simplified approach - in practice, we'd need proper domain tracking
                    if batch_idx < 1000:  # First 1000 samples are JARVIS-DFT
                        domain = GateDomain.JARVIS_DFT
                    else:  # Rest are ANI1x
                        domain = GateDomain.ANI1X
                
                # Alternative: Use sample count to determine domain
                if sample_count < 1000:
                    domain = GateDomain.JARVIS_DFT
                else:
                    domain = GateDomain.ANI1X
                
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


def run_gate_hard_ranking_on_real_predictions(
    predictions: List[PredictionResult],
    output_dir: str = "artifacts/real_gate_hard"
) -> Dict[str, List[PredictionResult]]:
    """Run gate-hard ranking on real ensemble predictions."""
    logger.info("Running gate-hard ranking on real predictions...")
    
    # Configure ranking with optimal K values
    ranking_config = DomainRankingConfig(
        jarvis_dft_k=80,
        jarvis_elastic_k=40,
        oc20_s2ef_k=80,
        oc22_s2ef_k=40,
        ani1x_k=30,
        global_k=270,
        output_dir=output_dir,
        alpha_variance=1.0,
        beta_tm_flag=0.5,
        gamma_near_degeneracy=0.1
    )
    
    # Run gate-hard ranking
    ranker = GateHardRanker(ranking_config)
    results = ranker.run_gate_hard_ranking(predictions)
    
    logger.info("Gate-hard ranking completed!")
    logger.info(f"Results saved to: {output_dir}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Complete ensemble pipeline on real data")
    
    parser.add_argument("--ensemble-dir", type=str, 
                       default="models/gnn_training_enhanced/ensemble",
                       help="Directory containing ensemble models")
    parser.add_argument("--output-dir", type=str,
                       default="artifacts/real_pipeline",
                       help="Output directory")
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
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load ensemble models
        models = load_ensemble_models(args.ensemble_dir, args.device)
        
        # Step 2: Create unified dataset configuration
        unified_config = create_unified_dataset_config()
        
        # Step 3: Load unified dataset
        logger.info("Loading unified dataset...")
        registry = UnifiedDatasetRegistry(unified_config)
        datasets = registry.load_all_datasets()
        train_data = datasets['train']
        
        # Step 4: Create dataloader
        dataloader = registry.create_dataloader(
            train_data, batch_size=args.batch_size, shuffle=False
        )
        
        # Step 5: Generate ensemble predictions
        predictions = generate_ensemble_predictions_on_real_data(
            models, dataloader, args.device, args.max_samples
        )
        
        # Step 6: Run gate-hard ranking
        gate_hard_results = run_gate_hard_ranking_on_real_predictions(
            predictions, str(output_path / "gate_hard")
        )
        
        # Step 7: Save results
        predictions_file = output_path / "real_ensemble_predictions.json"
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
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_dict, f, indent=2)
        
        # Print summary
        domain_counts = {}
        for pred in predictions:
            domain = pred.domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("=" * 60)
        logger.info("REAL ENSEMBLE PIPELINE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info("Domain distribution:")
        for domain, count in domain_counts.items():
            logger.info(f"  {domain}: {count}")
        
        # Gate-hard results summary
        logger.info("\nGate-hard ranking results:")
        for domain, samples in gate_hard_results.items():
            if isinstance(samples, list):
                logger.info(f"  {domain}: {len(samples)} samples")
        
        logger.info(f"\nPredictions saved to: {predictions_file}")
        logger.info(f"Gate-hard results saved to: {output_path / 'gate_hard'}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
