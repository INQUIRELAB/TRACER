"""Gate-hard ranking system with domain-specific selection and merging."""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DatasetDomain(Enum):
    """Dataset domain enumeration."""
    JARVIS_DFT = "jarvis_dft"
    JARVIS_ELASTIC = "jarvis_elastic"
    OC20_S2EF = "oc20_s2ef"
    OC22_S2EF = "oc22_s2ef"
    ANI1X = "ani1x"


@dataclass
class PredictionResult:
    """Container for prediction results."""
    sample_id: str
    domain: DatasetDomain
    energy_pred: float
    energy_target: float
    energy_variance: float
    forces_pred: Optional[np.ndarray] = None
    forces_target: Optional[np.ndarray] = None
    forces_variance: Optional[np.ndarray] = None
    tm_flag: bool = False
    near_degeneracy_proxy: float = 0.0
    molecular_properties: Optional[Dict] = None


@dataclass
class DomainRankingConfig:
    """Configuration for domain-specific ranking."""
    # Top-K per domain (OPTIMAL CONFIGURATION)
    jarvis_dft_k: int = 80
    jarvis_elastic_k: int = 40
    oc20_s2ef_k: int = 80
    oc22_s2ef_k: int = 40
    ani1x_k: int = 30
    
    # Scoring weights
    alpha_variance: float = 1.0
    beta_tm_flag: float = 0.5
    gamma_near_degeneracy: float = 0.1
    
    # Global settings
    global_k: int = 270
    output_dir: str = "artifacts/gate_hard"


class GateHardRanker:
    """Gate-hard ranking system with domain-specific selection."""
    
    def __init__(self, config: DomainRankingConfig):
        """Initialize gate-hard ranker.
        
        Args:
            config: Ranking configuration
        """
        self.config = config
        self.domain_k_map = {
            DatasetDomain.JARVIS_DFT: config.jarvis_dft_k,
            DatasetDomain.JARVIS_ELASTIC: config.jarvis_elastic_k,
            DatasetDomain.OC20_S2EF: config.oc20_s2ef_k,
            DatasetDomain.OC22_S2EF: config.oc22_s2ef_k,
            DatasetDomain.ANI1X: config.ani1x_k,
        }
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def compute_ensemble_statistics(self, predictions: List[PredictionResult]) -> Dict[str, Dict]:
        """Compute ensemble mean and variance per dataset split.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with ensemble statistics per domain
        """
        domain_stats = {}
        
        for domain in DatasetDomain:
            domain_predictions = [p for p in predictions if p.domain == domain]
            
            if not domain_predictions:
                continue
                
            # Extract energies and variances
            energies = np.array([p.energy_pred for p in domain_predictions])
            variances = np.array([p.energy_variance for p in domain_predictions])
            targets = np.array([p.energy_target for p in domain_predictions])
            
            # Compute ensemble statistics
            domain_stats[domain.value] = {
                'mean_energy': np.mean(energies),
                'std_energy': np.std(energies),
                'mean_variance': np.mean(variances),
                'std_variance': np.std(variances),
                'mean_target': np.mean(targets),
                'std_target': np.std(targets),
                'mae': np.mean(np.abs(energies - targets)),
                'rmse': np.sqrt(np.mean((energies - targets) ** 2)),
                'count': len(domain_predictions)
            }
            
        return domain_stats
    
    def compute_scoring_function(self, prediction: PredictionResult) -> float:
        """Compute scoring function for gate-hard selection.
        
        Score = α·variance + β·TM_flag + γ·near_degeneracy_proxy
        
        Args:
            prediction: Prediction result
            
        Returns:
            Score for ranking
        """
        score = (
            self.config.alpha_variance * prediction.energy_variance +
            self.config.beta_tm_flag * (1.0 if prediction.tm_flag else 0.0) +
            self.config.gamma_near_degeneracy * prediction.near_degeneracy_proxy
        )
        
        return score
    
    def rank_per_domain(self, predictions: List[PredictionResult]) -> Dict[str, List[PredictionResult]]:
        """Rank predictions per domain and select top-K.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with top-K predictions per domain
        """
        domain_rankings = {}
        
        for domain in DatasetDomain:
            domain_predictions = [p for p in predictions if p.domain == domain]
            
            if not domain_predictions:
                logger.warning(f"No predictions found for domain {domain.value}")
                continue
            
            # Compute scores
            scored_predictions = []
            for pred in domain_predictions:
                score = self.compute_scoring_function(pred)
                scored_predictions.append((pred, score))
            
            # Sort by score (descending - higher scores are "harder" cases)
            scored_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Select top-K
            k = self.domain_k_map[domain]
            top_k = [pred for pred, score in scored_predictions[:k]]
            
            domain_rankings[domain.value] = top_k
            
            logger.info(f"Domain {domain.value}: Selected top-{len(top_k)} from {len(domain_predictions)} predictions")
        
        return domain_rankings
    
    def merge_top_k_global(self, domain_rankings: Dict[str, List[PredictionResult]]) -> List[PredictionResult]:
        """Merge top-K selections from all domains into global ranking.
        
        Args:
            domain_rankings: Top-K predictions per domain
            
        Returns:
            Global top-K predictions with domain tags
        """
        all_top_k = []
        
        for domain_name, predictions in domain_rankings.items():
            for pred in predictions:
                # Add domain tag
                pred_with_tag = PredictionResult(
                    sample_id=f"{domain_name}_{pred.sample_id}",
                    domain=pred.domain,
                    energy_pred=pred.energy_pred,
                    energy_target=pred.energy_target,
                    energy_variance=pred.energy_variance,
                    forces_pred=pred.forces_pred,
                    forces_target=pred.forces_target,
                    forces_variance=pred.forces_variance,
                    tm_flag=pred.tm_flag,
                    near_degeneracy_proxy=pred.near_degeneracy_proxy,
                    molecular_properties=pred.molecular_properties
                )
                all_top_k.append(pred_with_tag)
        
        # Sort globally by score
        scored_global = []
        for pred in all_top_k:
            score = self.compute_scoring_function(pred)
            scored_global.append((pred, score))
        
        scored_global.sort(key=lambda x: x[1], reverse=True)
        
        # Select global top-K
        global_top_k = [pred for pred, score in scored_global[:self.config.global_k]]
        
        logger.info(f"Global ranking: Selected top-{len(global_top_k)} from {len(all_top_k)} domain selections")
        
        return global_top_k
    
    def save_results(self, domain_rankings: Dict[str, List[PredictionResult]], 
                    global_top_k: List[PredictionResult],
                    ensemble_stats: Dict[str, Dict]) -> None:
        """Save ranking results to files.
        
        Args:
            domain_rankings: Top-K predictions per domain
            global_top_k: Global top-K predictions
            ensemble_stats: Ensemble statistics
        """
        output_dir = Path(self.config.output_dir)
        
        # Save per-domain files
        for domain_name, predictions in domain_rankings.items():
            domain_file = output_dir / f"topK_{domain_name}.jsonl"
            
            with open(domain_file, 'w') as f:
                for pred in predictions:
                    result_dict = {
                        'sample_id': pred.sample_id,
                        'domain': pred.domain.value,
                        'energy_pred': pred.energy_pred,
                        'energy_target': pred.energy_target,
                        'energy_variance': pred.energy_variance,
                        'tm_flag': pred.tm_flag,
                        'near_degeneracy_proxy': pred.near_degeneracy_proxy,
                        'score': self.compute_scoring_function(pred)
                    }
                    
                    if pred.forces_pred is not None:
                        result_dict['forces_pred'] = pred.forces_pred.tolist()
                    if pred.forces_target is not None:
                        result_dict['forces_target'] = pred.forces_target.tolist()
                    if pred.forces_variance is not None:
                        result_dict['forces_variance'] = pred.forces_variance.tolist()
                    if pred.molecular_properties is not None:
                        result_dict['molecular_properties'] = pred.molecular_properties
                    
                    f.write(json.dumps(result_dict) + '\n')
            
            logger.info(f"Saved {len(predictions)} predictions to {domain_file}")
        
        # Save merged global file
        global_file = output_dir / "topK_all.jsonl"
        with open(global_file, 'w') as f:
            for pred in global_top_k:
                result_dict = {
                    'sample_id': pred.sample_id,
                    'domain': pred.domain.value,
                    'energy_pred': pred.energy_pred,
                    'energy_target': pred.energy_target,
                    'energy_variance': pred.energy_variance,
                    'tm_flag': pred.tm_flag,
                    'near_degeneracy_proxy': pred.near_degeneracy_proxy,
                    'score': self.compute_scoring_function(pred)
                }
                
                if pred.forces_pred is not None:
                    result_dict['forces_pred'] = pred.forces_pred.tolist()
                if pred.forces_target is not None:
                    result_dict['forces_target'] = pred.forces_target.tolist()
                if pred.forces_variance is not None:
                    result_dict['forces_variance'] = pred.forces_variance.tolist()
                if pred.molecular_properties is not None:
                    result_dict['molecular_properties'] = pred.molecular_properties
                
                f.write(json.dumps(result_dict) + '\n')
        
        logger.info(f"Saved {len(global_top_k)} global predictions to {global_file}")
        
        # Save ensemble statistics
        stats_file = output_dir / "ensemble_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(ensemble_stats, f, indent=2)
        
        logger.info(f"Saved ensemble statistics to {stats_file}")
    
    def print_ranking_table(self, domain_rankings: Dict[str, List[PredictionResult]], 
                           global_top_k: List[PredictionResult]) -> None:
        """Print ranking results table.
        
        Args:
            domain_rankings: Top-K predictions per domain
            global_top_k: Global top-K predictions
        """
        print("\n" + "="*80)
        print("GATE-HARD RANKING RESULTS")
        print("="*80)
        
        # Domain-specific rankings
        print("\nTOP-K PER DOMAIN:")
        print("-" * 50)
        total_domain_selections = 0
        
        for domain_name, predictions in domain_rankings.items():
            print(f"{domain_name:15s}: {len(predictions):3d} samples")
            total_domain_selections += len(predictions)
            
            # Show top 5 for each domain
            print(f"  Top 5 scores: ", end="")
            top_5_scores = [self.compute_scoring_function(p) for p in predictions[:5]]
            print(", ".join([f"{score:.4f}" for score in top_5_scores]))
        
        print(f"\nTotal domain selections: {total_domain_selections}")
        
        # Global ranking
        print(f"\nGLOBAL TOP-K: {len(global_top_k)} samples")
        print("-" * 50)
        
        # Show distribution by domain
        domain_counts = {}
        for pred in global_top_k:
            domain = pred.domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print("Distribution by domain:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain:15s}: {count:3d} samples")
        
        # Show top 10 global scores
        print(f"\nTop 10 global scores:")
        top_10_scores = [self.compute_scoring_function(p) for p in global_top_k[:10]]
        for i, score in enumerate(top_10_scores, 1):
            print(f"  {i:2d}. {score:.6f}")
        
        print("="*80)
    
    def run_gate_hard_ranking(self, predictions: List[PredictionResult]) -> Dict:
        """Run complete gate-hard ranking pipeline.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Ranking results dictionary
        """
        logger.info(f"Starting gate-hard ranking with {len(predictions)} predictions")
        
        # Compute ensemble statistics
        ensemble_stats = self.compute_ensemble_statistics(predictions)
        
        # Rank per domain
        domain_rankings = self.rank_per_domain(predictions)
        
        # Merge global top-K
        global_top_k = self.merge_top_k_global(domain_rankings)
        
        # Save results
        self.save_results(domain_rankings, global_top_k, ensemble_stats)
        
        # Print table
        self.print_ranking_table(domain_rankings, global_top_k)
        
        return {
            'domain_rankings': domain_rankings,
            'global_top_k': global_top_k,
            'ensemble_stats': ensemble_stats,
            'config': self.config
        }


def create_sample_predictions() -> List[PredictionResult]:
    """Create sample predictions for testing purposes ONLY.
    
    WARNING: This function generates synthetic data for testing only.
    It is NOT used in production or final results.
    """
    predictions = []
    
    # Sample data for different domains
    domains = [
        (DatasetDomain.JARVIS_DFT, 200),
        (DatasetDomain.OC20_S2EF, 150),
        (DatasetDomain.OC22_S2EF, 100),
        (DatasetDomain.ANI1X, 50),
        (DatasetDomain.JARVIS_ELASTIC, 80)
    ]
    
    np.random.seed(42)
    
    for domain, count in domains:
        for i in range(count):
            # Generate sample predictions
            energy_pred = np.random.normal(0.0, 1.0)
            energy_target = np.random.normal(0.0, 0.8)
            energy_variance = np.random.exponential(0.5)
            
            # TM flag (transition metal)
            tm_flag = np.random.random() < 0.3
            
            # Near-degeneracy proxy
            near_degeneracy = np.random.exponential(0.2)
            
            pred = PredictionResult(
                sample_id=f"{domain.value}_{i:04d}",
                domain=domain,
                energy_pred=energy_pred,
                energy_target=energy_target,
                energy_variance=energy_variance,
                tm_flag=tm_flag,
                near_degeneracy_proxy=near_degeneracy,
                molecular_properties={
                    'num_atoms': np.random.randint(5, 50),
                    'formation_energy': energy_target,
                    'band_gap': np.random.exponential(1.0)
                }
            )
            predictions.append(pred)
    
    return predictions


if __name__ == "__main__":
    # Test the gate-hard ranking system
    logging.basicConfig(level=logging.INFO)
    
    # Create sample predictions
    predictions = create_sample_predictions()
    
    # Configure ranking (OPTIMAL CONFIGURATION)
    config = DomainRankingConfig(
        jarvis_dft_k=80,
        jarvis_elastic_k=40,
        oc20_s2ef_k=80,
        oc22_s2ef_k=40,
        ani1x_k=30,
        global_k=270,
        alpha_variance=1.0,
        beta_tm_flag=0.5,
        gamma_near_degeneracy=0.1
    )
    
    # Run ranking
    ranker = GateHardRanker(config)
    results = ranker.run_gate_hard_ranking(predictions)
    
    print(f"\nGate-hard ranking completed successfully!")
    print(f"Results saved to: {config.output_dir}")
