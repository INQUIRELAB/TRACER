#!/usr/bin/env python3
"""Generate sample ensemble predictions for testing gate-hard ranking."""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_predictions(num_samples: int = 1500, 
                               output_file: str = "ensemble_predictions.json") -> None:
    """Generate sample ensemble predictions.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Output file path
    """
    np.random.seed(42)
    
    # Domain distribution
    domains = {
        'jarvis_dft': 0.4,      # 40% of samples
        'jarvis_elastic': 0.1,  # 10% of samples
        'oc20_s2ef': 0.3,       # 30% of samples
        'oc22_s2ef': 0.15,      # 15% of samples
        'ani1x': 0.05           # 5% of samples
    }
    
    predictions = []
    
    for i in range(num_samples):
        # Select domain based on distribution
        domain_probs = list(domains.values())
        domain_names = list(domains.keys())
        domain = np.random.choice(domain_names, p=domain_probs)
        
        # Generate prediction based on domain characteristics
        if domain == 'jarvis_dft':
            # JARVIS-DFT: formation energies, more stable
            energy_target = np.random.normal(-2.0, 1.5)
            energy_pred = energy_target + np.random.normal(0.0, 0.3)
            energy_variance = np.random.exponential(0.2)
            tm_flag = np.random.random() < 0.2  # Less TM in JARVIS
            
        elif domain == 'jarvis_elastic':
            # JARVIS-Elastic: elastic properties
            energy_target = np.random.normal(-1.5, 1.0)
            energy_pred = energy_target + np.random.normal(0.0, 0.4)
            energy_variance = np.random.exponential(0.3)
            tm_flag = np.random.random() < 0.1
            
        elif domain == 'oc20_s2ef':
            # OC20: catalyst surfaces, more TM
            energy_target = np.random.normal(-1.0, 2.0)
            energy_pred = energy_target + np.random.normal(0.0, 0.5)
            energy_variance = np.random.exponential(0.4)
            tm_flag = np.random.random() < 0.6  # More TM in catalysts
            
        elif domain == 'oc22_s2ef':
            # OC22: newer catalyst data
            energy_target = np.random.normal(-0.8, 1.8)
            energy_pred = energy_target + np.random.normal(0.0, 0.4)
            energy_variance = np.random.exponential(0.35)
            tm_flag = np.random.random() < 0.5
            
        else:  # ani1x
            # ANI1x: small molecules
            energy_target = np.random.normal(-0.5, 0.8)
            energy_pred = energy_target + np.random.normal(0.0, 0.2)
            energy_variance = np.random.exponential(0.15)
            tm_flag = np.random.random() < 0.05  # Very few TM
        
        # Near-degeneracy proxy (higher for complex systems)
        if tm_flag:
            near_degeneracy = np.random.exponential(0.3)
        else:
            near_degeneracy = np.random.exponential(0.1)
        
        # Generate forces (optional)
        num_atoms = np.random.randint(5, 50)
        forces_pred = np.random.normal(0.0, 0.1, (num_atoms, 3))
        forces_target = np.random.normal(0.0, 0.1, (num_atoms, 3))
        forces_variance = np.random.exponential(0.05, (num_atoms, 3))
        
        # Molecular properties
        molecular_properties = {
            'num_atoms': num_atoms,
            'formation_energy': energy_target,
            'band_gap': np.random.exponential(1.0) if domain.startswith('jarvis') else None,
            'surface_area': np.random.exponential(100.0) if domain.startswith('oc') else None,
            'molecular_weight': np.random.uniform(50.0, 500.0)
        }
        
        prediction = {
            'sample_id': f"{domain}_{i:04d}",
            'domain': domain,
            'energy_pred': float(energy_pred),
            'energy_target': float(energy_target),
            'energy_variance': float(energy_variance),
            'forces_pred': forces_pred.tolist(),
            'forces_target': forces_target.tolist(),
            'forces_variance': forces_variance.tolist(),
            'tm_flag': bool(tm_flag),
            'near_degeneracy_proxy': float(near_degeneracy),
            'molecular_properties': molecular_properties
        }
        
        predictions.append(prediction)
    
    # Save predictions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Generated {len(predictions)} ensemble predictions")
    logger.info(f"Saved to: {output_path}")
    
    # Print domain distribution
    domain_counts = {}
    for pred in predictions:
        domain = pred['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain:15s}: {count:4d} samples ({count/len(predictions)*100:.1f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate sample ensemble predictions")
    parser.add_argument("--num-samples", type=int, default=1500,
                       help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="ensemble_predictions.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    generate_sample_predictions(args.num_samples, args.output)


if __name__ == "__main__":
    main()
