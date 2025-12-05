#!/usr/bin/env python3
"""
Evaluate the realistic delta head on gate-hard samples.
This script uses the newly trained delta head with realistic data.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import typer
from rich.console import Console
import pandas as pd
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig

console = Console()

def load_realistic_delta_head(model_path: str, device: torch.device) -> DeltaHead:
    """Load the realistic delta head model."""
    console.print(f"🔄 Loading realistic delta head from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config')
    if config is None:
        # Fallback to default config
        config = DeltaHeadConfig()
    
    # Create and load model
    model = DeltaHead(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    console.print(f"✅ Loaded realistic delta head")
    console.print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    console.print(f"   Train Loss: {checkpoint.get('train_loss', 'unknown'):.6f}")
    console.print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
    
    return model

def load_realistic_data(data_dir: str) -> tuple:
    """Load realistic SchNet features and QNN labels."""
    data_path = Path(data_dir)
    
    # Load SchNet features
    features_file = data_path / "realistic_schnet_features.json"
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    
    # Load QNN labels
    labels_file = data_path / "realistic_qnn_labels.csv"
    qnn_labels = pd.read_csv(labels_file)
    
    console.print(f"📊 Loaded realistic data:")
    console.print(f"   SchNet features: {len(features_data)} samples")
    console.print(f"   QNN labels: {len(qnn_labels)} samples")
    
    return features_data, qnn_labels

def evaluate_realistic_delta_head(
    model: DeltaHead,
    features_data: Dict[str, List[float]],
    qnn_labels: pd.DataFrame,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate the realistic delta head."""
    console.print("🔄 Evaluating realistic delta head...")
    
    results = {
        'total_samples': 0,
        'corrected_samples': 0,
        'domain_results': {},
        'overall_mae': 0.0,
        'improvement_percentage': 0.0
    }
    
    domain_map = {
        'jarvis_dft': 0,
        'jarvis_elastic': 1, 
        'oc20_s2ef': 2,
        'oc22_s2ef': 3,
        'ani1x': 4
    }
    
    total_mae_before = 0.0
    total_mae_after = 0.0
    
    with torch.no_grad():
        for _, row in qnn_labels.iterrows():
            sample_id = row['sample_id']
            domain = row['domain_id']
            
            if sample_id not in features_data:
                continue
            
            # Get SchNet features
            features = torch.tensor(features_data[sample_id], dtype=torch.float32).to(device)
            domain_id = torch.tensor([domain_map.get(domain, 0)], dtype=torch.long).to(device)
            
            # Get delta prediction
            delta_output = model(features, domain_id)
            delta_pred = delta_output['delta_energy'].cpu().item()
            
            # Calculate corrections
            gnn_energy = row['gnn_energy']
            qnn_energy = row['qnn_energy']
            delta_target = row['delta_energy']
            
            corrected_energy = gnn_energy + delta_pred
            
            # Calculate errors
            error_before = abs(gnn_energy - qnn_energy)
            error_after = abs(corrected_energy - qnn_energy)
            
            total_mae_before += error_before
            total_mae_after += error_after
            
            # Track domain results
            if domain not in results['domain_results']:
                results['domain_results'][domain] = {
                    'samples': 0,
                    'mae_before': 0.0,
                    'mae_after': 0.0,
                    'improvement': 0.0
                }
            
            domain_result = results['domain_results'][domain]
            domain_result['samples'] += 1
            domain_result['mae_before'] += error_before
            domain_result['mae_after'] += error_after
            
            results['total_samples'] += 1
            results['corrected_samples'] += 1
    
    # Calculate overall metrics
    if results['total_samples'] > 0:
        results['overall_mae'] = total_mae_after / results['total_samples']
        mae_before_avg = total_mae_before / results['total_samples']
        results['improvement_percentage'] = ((mae_before_avg - results['overall_mae']) / mae_before_avg) * 100
        
        # Calculate domain metrics
        for domain, domain_result in results['domain_results'].items():
            if domain_result['samples'] > 0:
                domain_result['mae_before'] /= domain_result['samples']
                domain_result['mae_after'] /= domain_result['samples']
                domain_result['improvement'] = ((domain_result['mae_before'] - domain_result['mae_after']) / domain_result['mae_before']) * 100
    
    console.print(f"✅ Evaluation completed for {results['total_samples']} samples")
    return results

def print_evaluation_results(results: Dict[str, Any]):
    """Print detailed evaluation results."""
    console.print("\n" + "="*60)
    console.print("📊 REALISTIC DELTA HEAD EVALUATION RESULTS")
    console.print("="*60)
    
    console.print(f"\n🎯 Overall Results:")
    console.print(f"   Total samples: {results['total_samples']}")
    console.print(f"   Corrected samples: {results['corrected_samples']}")
    console.print(f"   Overall MAE: {results['overall_mae']:.6f}")
    console.print(f"   Improvement: {results['improvement_percentage']:.2f}%")
    
    console.print(f"\n📈 Per-Domain Results:")
    for domain, domain_result in results['domain_results'].items():
        console.print(f"   {domain}:")
        console.print(f"     Samples: {domain_result['samples']}")
        console.print(f"     MAE Before: {domain_result['mae_before']:.6f}")
        console.print(f"     MAE After: {domain_result['mae_after']:.6f}")
        console.print(f"     Improvement: {domain_result['improvement']:.2f}%")
    
    console.print("\n" + "="*60)

def main(
    model_path: str = typer.Option("artifacts/delta_head_realistic/best_model.pt", help="Path to realistic delta head model"),
    data_dir: str = typer.Option("artifacts/real_data_fixed", help="Directory containing realistic data"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
):
    """Evaluate the realistic delta head on gate-hard samples."""
    
    console.print("🚀 Evaluating Realistic Delta Head")
    console.print(f"🤖 Model path: {model_path}")
    console.print(f"📁 Data directory: {data_dir}")
    console.print(f"🔧 Device: {device}")
    
    # Set device
    device = torch.device(device)
    
    # Load model
    model = load_realistic_delta_head(model_path, device)
    
    # Load data
    features_data, qnn_labels = load_realistic_data(data_dir)
    
    # Evaluate
    results = evaluate_realistic_delta_head(model, features_data, qnn_labels, device)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    output_path = Path("artifacts/realistic_delta_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n💾 Results saved to: {output_path}")
    console.print("\n✅ Realistic delta head evaluation completed!")

if __name__ == "__main__":
    typer.run(main)
