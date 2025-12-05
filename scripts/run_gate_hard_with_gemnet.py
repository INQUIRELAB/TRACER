#!/usr/bin/env python3
"""
Run Gate-Hard Ranking on GemNet Predictions
Selects the top-K hardest cases for quantum labeling.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.gate_hard_ranking import GateHardRanker, DomainRankingConfig, PredictionResult, DatasetDomain

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_predictions():
    """Load GemNet predictions."""
    pred_path = Path("artifacts/gemnet_predictions/ensemble_predictions.json")
    
    logger.info(f"📥 Loading predictions from: {pred_path}")
    
    if not pred_path.exists():
        logger.error(f"Predictions not found: {pred_path}")
        return []
    
    with open(pred_path, 'r') as f:
        predictions_dict = json.load(f)
    
    logger.info(f"   Loaded {len(predictions_dict)} predictions")
    return predictions_dict


def convert_to_prediction_results(predictions_dict):
    """Convert dictionary predictions to PredictionResult objects."""
    logger.info("🔄 Converting to PredictionResult objects...")
    
    results = []
    
    for pred in predictions_dict:
        # Map domain string to enum
        domain_str = pred.get('domain', 'jarvis_dft')
        domain_map = {
            'jarvis_dft': DatasetDomain.JARVIS_DFT,
            'jarvis_elastic': DatasetDomain.JARVIS_ELASTIC,
            'oc20_s2ef': DatasetDomain.OC20_S2EF,
            'oc22_s2ef': DatasetDomain.OC22_S2EF,
            'ani1x': DatasetDomain.ANI1X
        }
        domain = domain_map.get(domain_str, DatasetDomain.JARVIS_DFT)
        
        # Create PredictionResult
        result = PredictionResult(
            sample_id=pred['sample_id'],
            domain=domain,
            energy_pred=pred['energy_pred'],
            energy_target=pred['energy_target'],
            energy_variance=pred.get('energy_variance', 0.0),
            forces_pred=np.array(pred.get('forces_pred', [])) if pred.get('forces_pred') else None,
            forces_target=None,  # Not available in predictions
            forces_variance=None,
            tm_flag=pred.get('tm_flag', False),
            near_degeneracy_proxy=pred.get('near_degeneracy_proxy', 0.0)
        )
        
        results.append(result)
    
    logger.info(f"   Converted {len(results)} results")
    return results


def run_gate_hard_ranking(results):
    """Run gate-hard ranking to select hard cases."""
    logger.info("🎯 Running Gate-Hard Ranking")
    logger.info("="*80)
    
    # Configuration for optimal top-K selection - increased for single domain
    config = DomainRankingConfig(
        jarvis_dft_k=270,  # Increased since all samples are from this domain
        jarvis_elastic_k=40,
        oc20_s2ef_k=80,
        oc22_s2ef_k=40,
        ani1x_k=30,
        global_k=270,
        output_dir="artifacts/gate_hard_gemnet"
    )
    
    # Initialize ranker
    ranker = GateHardRanker(config)
    
    # Run ranking
    ranking_results = ranker.run_gate_hard_ranking(results)
    
    return ranking_results


def save_hard_cases_for_quantum(ranking_results):
    """Save hard cases for quantum labeling."""
    logger.info("💾 Saving hard cases for quantum labeling...")
    
    # Extract selected samples from ranking results
    selected_samples = []
    
    if 'per_domain_results' in ranking_results:
        for domain, domain_results in ranking_results['per_domain_results'].items():
            if 'top_k_samples' in domain_results:
                selected_samples.extend(domain_results['top_k_samples'])
    
    # Also check global results
    if 'global_results' in ranking_results and 'merged_top_k' in ranking_results['global_results']:
        selected_samples.extend(ranking_results['global_results']['merged_top_k'])
    
    # Save for quantum labeling
    output_path = Path("artifacts/quantum_input")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert back to dictionary format
    hard_cases = []
    for sample in selected_samples:
        hard_cases.append({
            'sample_id': sample.sample_id,
            'domain': sample.domain.value,
            'energy_pred': sample.energy_pred,
            'energy_target': sample.energy_target,
            'energy_variance': sample.energy_variance,
            'tm_flag': sample.tm_flag,
            'near_degeneracy_proxy': sample.near_degeneracy_proxy
        })
    
    with open(output_path / 'hard_cases.json', 'w') as f:
        json.dump(hard_cases, f, indent=2)
    
    logger.info(f"   Saved {len(hard_cases)} hard cases")
    logger.info(f"   Output: {output_path / 'hard_cases.json'}")
    
    return hard_cases


def main():
    """Main execution."""
    logger.info("🚀 GATE-HARD RANKING WITH GEMNET")
    logger.info("="*80)
    
    # 1. Load predictions
    predictions_dict = load_predictions()
    
    if not predictions_dict:
        logger.error("❌ No predictions loaded!")
        return
    
    # 2. Convert to PredictionResult objects
    results = convert_to_prediction_results(predictions_dict)
    
    # 3. Run gate-hard ranking
    ranking_results = run_gate_hard_ranking(results)
    
    # 4. Save hard cases for quantum labeling
    hard_cases = save_hard_cases_for_quantum(ranking_results)
    
    logger.info("\n✅ GATE-HARD RANKING COMPLETE!")
    logger.info("="*80)
    logger.info(f"   Selected {len(hard_cases)} hard cases")
    logger.info(f"   Ready for quantum labeling")
    logger.info("="*80)
    
    # Print summary
    logger.info("\n📊 SUMMARY:")
    logger.info("-"*80)
    if 'per_domain_results' in ranking_results:
        logger.info("   Per-domain results:")
        for domain, domain_results in ranking_results['per_domain_results'].items():
            if 'top_k_samples' in domain_results:
                logger.info(f"   - {domain}: {len(domain_results['top_k_samples'])} samples")
    
    if 'global_results' in ranking_results:
        if 'merged_top_k' in ranking_results['global_results']:
            logger.info(f"\n   Global merged: {len(ranking_results['global_results']['merged_top_k'])} samples")


if __name__ == "__main__":
    main()

