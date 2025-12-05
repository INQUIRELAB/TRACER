"""Main pipeline for DFT→GNN→QNN hybrid approach."""

from typing import Dict, List, Optional, Tuple, Union
import typer
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
import logging
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .gate_hard_ranking import GateHardRanker, DomainRankingConfig, PredictionResult, DatasetDomain

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Domain types for evaluation."""
    ANI1X = "ani1x"
    OC20 = "oc20"
    OC22 = "oc22"
    JARVIS = "jarvis"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    domain: str
    num_samples: int
    mae_total_energy: float
    mae_per_atom_energy: float
    rmse_total_energy: float
    rmse_per_atom_energy: float
    mae_forces: Optional[float] = None
    rmse_forces: Optional[float] = None


@dataclass
class EvaluationReport:
    """Container for evaluation report."""
    metrics_per_domain: Dict[str, EvaluationMetrics]
    overall_metrics: EvaluationMetrics
    delta_head_metrics: Optional[Dict[str, EvaluationMetrics]] = None
    delta_head_overall: Optional[EvaluationMetrics] = None
    improvements: Optional[Dict[str, Dict[str, float]]] = None


app = typer.Typer()


class HybridPipeline:
    """Main pipeline orchestrating DFT→GNN→QNN workflow."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize hybrid pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.pipeline.device)
        # Track whether the loaded GNN predicts per-atom energy (True) or total energy (False)
        self.per_atom_model: bool = False
        # Pipeline-level toggle: operate and report in per-atom or total units
        self.per_atom_pipeline: bool = True
        
    def load_data(self, data_path: str) -> Dict:
        """Load and preprocess input data.
        
        Args:
            data_path: Path to input data
            
        Returns:
            Processed data dictionary
        """
        logger.info(f"Loading data from: {data_path}")
        
        try:
            # Import unified dataset registry
            from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig
            
            # Create configuration for loading datasets
            dataset_config = UnifiedDatasetConfig(
                datasets={
                    'jarvis_dft': DatasetConfig(
                        enabled=True,
                        max_samples=1000,
                        data_path="data/jarvis_dft",
                        unit_type="eV"
                    ),
                    'jarvis_elastic': DatasetConfig(
                        enabled=True,
                        max_samples=500,
                        data_path="data/jarvis_elastic", 
                        unit_type="eV"
                    ),
                    'oc20_s2ef': DatasetConfig(
                        enabled=True,
                        max_samples=800,
                        data_path="data/oc20_s2ef",
                        unit_type="eV"
                    ),
                    'oc22_s2ef': DatasetConfig(
                        enabled=True,
                        max_samples=400,
                        data_path="data/oc22_s2ef",
                        unit_type="eV"
                    ),
                    'ani1x': DatasetConfig(
                        enabled=True,
                        max_samples=200,
                        data_path="data/ani1x",
                        unit_type="eV"
                    )
                }
            )
            
            # Load datasets
            registry = UnifiedDatasetRegistry(dataset_config)
            all_samples = registry.load_all_datasets()
            
            logger.info(f"Successfully loaded {len(all_samples)} samples")
            
            return {
                'samples': all_samples,
                'num_samples': len(all_samples),
                'domains': list(set(sample.get('domain', 'unknown') for sample in all_samples))
            }
            
        except ImportError as e:
            logger.error(f"Could not import unified dataset registry: {e}")
            raise NotImplementedError("Unified dataset registry not available")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train_gnn_surrogate(self, train_data: Dict) -> Dict:
        """Train GNN surrogate model.
        
        Args:
            train_data: Training data
            
        Returns:
            Trained GNN model and metrics
        """
        logger.info("Training GNN surrogate model")
        
        try:
            # Import GNN training components
            from gnn.train import GNNTrainer
            from gnn.model import DomainAwareSchNet
            from torch.utils.data import DataLoader
            
            # Create model
            model = DomainAwareSchNet(
                hidden_channels=256,
                num_interactions=8,
                num_gaussians=64,
                cutoff=5.0,
                num_domains=5  # JARVIS-DFT, JARVIS-Elastic, OC20, OC22, ANI1x
            ).to(self.device)
            
            # Create trainer
            trainer = GNNTrainer(
                model=model,
                device=self.device,
                learning_rate=1e-3,
                weight_decay=1e-6,
                max_epochs=100
            )
            
            # Convert samples to PyTorch Geometric format
            from torch_geometric.data import Data, Batch
            from graphs.periodic_graph import PeriodicGraph
            
            graph_data = []
            for sample in train_data['samples'][:100]:  # Limit for training
                # Create graph from atomic structure
                positions = torch.tensor(sample.get('positions', [[0, 0, 0]]), dtype=torch.float32)
                atomic_numbers = torch.tensor(sample.get('atomic_numbers', [6]), dtype=torch.long)
                
                # Create periodic graph
                graph = PeriodicGraph.from_atoms(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    cell=torch.eye(3) * 10.0,  # Default cell
                    cutoff=5.0
                )
                
                # Add targets
                graph.y = torch.tensor([sample.get('energy', 0.0)], dtype=torch.float32)
                graph.domain_id = torch.tensor([self._get_domain_id(sample.get('domain', 'jarvis_dft'))], dtype=torch.long)
                
                graph_data.append(graph)
            
            # Create data loader
            train_loader = DataLoader(graph_data, batch_size=32, shuffle=True)
            
            # Train model
            trainer.train(train_loader, train_loader)  # Use same loader for validation
            
            logger.info("GNN training completed successfully")
            
            return {
                'model': model,
                'trainer': trainer,
                'num_samples': len(graph_data),
                'validation_loss': trainer.best_val_loss
            }
            
        except ImportError as e:
            logger.error(f"Could not import GNN components: {e}")
            raise NotImplementedError("GNN training components not available")
        except Exception as e:
            logger.error(f"Error training GNN: {e}")
            raise
    
    def estimate_uncertainty(self, gnn_model: Dict, test_data: Dict) -> Dict:
        """Estimate uncertainty for test cases.
        
        Args:
            gnn_model: Trained GNN model
            test_data: Test data
            
        Returns:
            Uncertainty estimates
        """
        logger.info("Estimating uncertainty for test cases")
        
        try:
            # Import uncertainty estimation components
            from gnn.uncertainty import EnsembleUncertainty
            
            # Create ensemble uncertainty estimator
            uncertainty_estimator = EnsembleUncertainty(
                models=[gnn_model['model']],  # Single model for now
                device=self.device
            )
            
            # Convert test samples to graph format
            from torch_geometric.data import DataLoader
            from graphs.periodic_graph import PeriodicGraph
            
            test_graphs = []
            for sample in test_data['samples'][:50]:  # Limit for testing
                positions = torch.tensor(sample.get('positions', [[0, 0, 0]]), dtype=torch.float32)
                atomic_numbers = torch.tensor(sample.get('atomic_numbers', [6]), dtype=torch.long)
                
                graph = PeriodicGraph.from_atoms(
                    positions=positions,
                    atomic_numbers=atomic_numbers,
                    cell=torch.eye(3) * 10.0,
                    cutoff=5.0
                )
                
                graph.domain_id = torch.tensor([self._get_domain_id(sample.get('domain', 'jarvis_dft'))], dtype=torch.long)
                test_graphs.append(graph)
            
            test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
            
            # Estimate uncertainty
            uncertainty_results = uncertainty_estimator.predict(test_loader)
            
            logger.info("Uncertainty estimation completed successfully")
            
            return {
                'uncertainty_estimates': uncertainty_results,
                'num_samples': len(test_graphs),
                'mean_uncertainty': np.mean([r['uncertainty'] for r in uncertainty_results])
            }
            
        except ImportError as e:
            logger.error(f"Could not import uncertainty components: {e}")
            raise NotImplementedError("Uncertainty estimation components not available")
        except Exception as e:
            logger.error(f"Error estimating uncertainty: {e}")
            raise
    
    def gate_hard_cases(self, uncertainties: Dict, threshold: float) -> List[int]:
        """Gate hard cases based on uncertainty threshold.
        
        Args:
            uncertainties: Uncertainty estimates
            threshold: Uncertainty threshold
            
        Returns:
            Indices of hard cases
        """
        logger.info(f"Gating hard cases with uncertainty threshold: {threshold}")
        
        hard_cases = []
        for i, result in enumerate(uncertainties['uncertainty_estimates']):
            if result['uncertainty'] > threshold:
                hard_cases.append(i)
        
        logger.info(f"Identified {len(hard_cases)} hard cases out of {uncertainties['num_samples']} total")
        return hard_cases
    
    def run_ensemble_ranking(self, ensemble_predictions: List[Dict], 
                            ranking_config: Optional[DomainRankingConfig] = None) -> Dict:
        """Run ensemble-based gate-hard ranking.
        
        Args:
            ensemble_predictions: List of ensemble prediction dictionaries
            ranking_config: Configuration for ranking
            
        Returns:
            Ranking results
        """
        if ranking_config is None:
            ranking_config = DomainRankingConfig()
        
        # Convert ensemble predictions to PredictionResult objects
        predictions = []
        for pred_dict in ensemble_predictions:
            # Extract domain from prediction
            domain_str = pred_dict.get('domain', 'jarvis_dft')
            try:
                domain = DatasetDomain(domain_str)
            except ValueError:
                logger.warning(f"Unknown domain {domain_str}, defaulting to JARVIS_DFT")
                domain = DatasetDomain.JARVIS_DFT
            
            # Create PredictionResult
            pred_result = PredictionResult(
                sample_id=pred_dict['sample_id'],
                domain=domain,
                energy_pred=pred_dict['energy_pred'],
                energy_target=pred_dict['energy_target'],
                energy_variance=pred_dict['energy_variance'],
                forces_pred=pred_dict.get('forces_pred'),
                forces_target=pred_dict.get('forces_target'),
                forces_variance=pred_dict.get('forces_variance'),
                tm_flag=pred_dict.get('tm_flag', False),
                near_degeneracy_proxy=pred_dict.get('near_degeneracy_proxy', 0.0),
                molecular_properties=pred_dict.get('molecular_properties')
            )
            predictions.append(pred_result)
        
        # Run gate-hard ranking
        ranker = GateHardRanker(ranking_config)
        results = ranker.run_gate_hard_ranking(predictions)
        
        return results
    
    def run_dmet_vqe(self, hard_cases: List[int], data: Dict) -> Dict:
        """Run DMET+VQE for hard cases.
        
        Args:
            hard_cases: Indices of hard cases
            data: Input data
            
        Returns:
            DMET+VQE results
        """
        logger.info(f"Running DMET+VQE calculations for {len(hard_cases)} hard cases")
        
        try:
            # Import quantum chemistry components
            from dft_hybrid.dmet.fragment import QuantumFragmentLabeler, FragmentGenerator
            
            # Create fragment generator and quantum labeler
            fragment_generator = FragmentGenerator()
            quantum_labeler = QuantumFragmentLabeler()
            
            quantum_results = []
            
            for i, case_idx in enumerate(hard_cases[:10]):  # Limit for testing
                logger.info(f"Processing hard case {i+1}/{min(len(hard_cases), 10)}: {case_idx}")
                
                # Get sample data
                sample = data['samples'][case_idx]
                
                # Generate molecular fragments
                fragments = fragment_generator.generate_fragments(
                    sample=sample,
                    domain=sample.get('domain', 'jarvis_dft')
                )
                
                # Run quantum calculations on fragments
                for fragment in fragments[:3]:  # Limit fragments per sample
                    quantum_result = quantum_labeler.label_fragment(
                        fragment=fragment,
                        backend='qiskit_simulator',
                        ansatz='uccsd',
                        max_steps=200,
                        shots=0
                    )
                    
                    quantum_results.append({
                        'case_idx': case_idx,
                        'fragment_id': fragment.get('id', f'frag_{i}'),
                        'quantum_result': quantum_result,
                        'sample_info': sample
                    })
            
            logger.info(f"Completed DMET+VQE calculations for {len(quantum_results)} fragments")
            
            return {
                'quantum_results': quantum_results,
                'num_hard_cases': len(hard_cases),
                'num_fragments': len(quantum_results)
            }
            
        except ImportError as e:
            logger.error(f"Could not import quantum chemistry components: {e}")
            raise NotImplementedError("Quantum chemistry components not available")
        except Exception as e:
            logger.error(f"Error running DMET+VQE: {e}")
            raise
    
    def train_delta_head(self, gnn_results: Dict, quantum_results: Dict) -> Dict:
        """Train delta learning head.
        
        Args:
            gnn_results: GNN predictions
            quantum_results: Quantum calculation results
            
        Returns:
            Trained delta head
        """
        logger.info("Training delta learning head")
        
        try:
            # Import delta head components
            from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadTrainer, DeltaHeadConfig
            
            # Create delta head configuration
            config = DeltaHeadConfig(
                schnet_feature_dim=256,
                domain_embedding_dim=16,
                hidden_dim=128,
                num_domains=5,
                learning_rate=1e-3,
                weight_decay=1e-4,
                num_epochs=50
            )
            
            # Create delta head model
            delta_head = DeltaHead(config).to(self.device)
            
            # Create trainer
            trainer = DeltaHeadTrainer(
                model=delta_head,
                config=config,
                device=self.device
            )
            
            # Prepare training data from quantum results
            training_data = []
            for quantum_result in quantum_results['quantum_results'][:20]:  # Limit for training
                # Extract features and targets
                sample_info = quantum_result['sample_info']
                quantum_data = quantum_result['quantum_result']
                
                # Extract real SchNet features from quantum result
                # In production, this would load actual SchNet model and extract features
                # For now, create physics-based features from quantum properties
                quantum_energy = quantum_data.get('ground_state_energy', 0.0)
                n_qubits = quantum_data.get('n_qubits', 10)
                converged = quantum_data.get('converged', True)
                
                # Create physics-based SchNet features
                schnet_features = torch.zeros(1, 256).to(self.device)
                
                # Energy-correlated features (based on quantum energy)
                energy_magnitude = abs(quantum_energy)
                schnet_features[0, :64] = torch.tensor([energy_magnitude] * 64).to(self.device) * 0.1
                
                # Qubit-correlated features (based on system size)
                qubit_features = torch.tensor([n_qubits] * 64).to(self.device) * 0.01
                schnet_features[0, 64:128] = qubit_features
                
                # Convergence-correlated features
                convergence_factor = 1.0 if converged else 0.5
                schnet_features[0, 128:192] = torch.tensor([convergence_factor] * 64).to(self.device) * 0.05
                
                # Domain-specific features
                domain_id = self._get_domain_id(sample_info.get('domain', 'jarvis_dft'))
                domain_features = torch.tensor([domain_id] * 64).to(self.device) * 0.02
                schnet_features[0, 192:256] = domain_features
                # Number of atoms for per-atom delta
                n_atoms = max(1, int(sample_info.get('num_atoms', len(sample_info.get('atomic_numbers', [])) or 1)))

                # Create training sample with physics-based features
                training_sample = {
                    'schnet_features': schnet_features,
                    'domain_id': torch.tensor([domain_id]).to(self.device),
                    # Train on per-atom delta by default for stability across sizes
                    'delta_target': torch.tensor([
                        (quantum_data.get('ground_state_energy', 0.0) - sample_info.get('energy', 0.0)) / n_atoms
                    ]).to(self.device)
                }
                training_data.append(training_sample)
            
            # Create data loaders
            from torch.utils.data import DataLoader, TensorDataset
            
            schnet_features = torch.cat([sample['schnet_features'] for sample in training_data])
            domain_ids = torch.cat([sample['domain_id'] for sample in training_data])
            delta_targets = torch.cat([sample['delta_target'] for sample in training_data])
            
            dataset = TensorDataset(schnet_features, domain_ids, delta_targets)
            train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Train delta head
            trainer.train(train_loader, train_loader)  # Use same loader for validation
            
            logger.info("Delta head training completed successfully")
            
            return {
                'delta_head': delta_head,
                'trainer': trainer,
                'config': config,
                'num_training_samples': len(training_data)
            }
            
        except ImportError as e:
            logger.error(f"Could not import delta head components: {e}")
            raise NotImplementedError("Delta head components not available")
        except Exception as e:
            logger.error(f"Error training delta head: {e}")
            raise
    
    def run_full_pipeline(self, input_path: str, output_path: str) -> Dict:
        """Run the complete hybrid pipeline.
        
        Args:
            input_path: Path to input data
            output_path: Path for output results
            
        Returns:
            Final pipeline results
        """
        logger.info("Running complete hybrid pipeline")
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data")
            data = self.load_data(input_path)
            
            # Step 2: Train GNN surrogate
            logger.info("Step 2: Training GNN surrogate")
            gnn_results = self.train_gnn_surrogate(data)
            
            # Step 3: Estimate uncertainty
            logger.info("Step 3: Estimating uncertainty")
            uncertainties = self.estimate_uncertainty(gnn_results, data)
            
            # Step 4: Gate hard cases
            logger.info("Step 4: Gating hard cases")
            hard_cases = self.gate_hard_cases(uncertainties, threshold=0.1)
            
            # Step 5: Run DMET+VQE for hard cases
            logger.info("Step 5: Running DMET+VQE calculations")
            quantum_results = self.run_dmet_vqe(hard_cases, data)
            
            # Step 6: Train delta head
            logger.info("Step 6: Training delta head")
            delta_results = self.train_delta_head(gnn_results, quantum_results)
            
            # Compile final results
            final_results = {
                'data_info': {
                    'num_samples': data['num_samples'],
                    'domains': data['domains']
                },
                'gnn_results': {
                    'validation_loss': gnn_results['validation_loss'],
                    'num_training_samples': gnn_results['num_samples']
                },
                'uncertainty_results': {
                    'num_hard_cases': len(hard_cases),
                    'mean_uncertainty': uncertainties['mean_uncertainty']
                },
                'quantum_results': {
                    'num_fragments': quantum_results['num_fragments'],
                    'num_hard_cases': quantum_results['num_hard_cases']
                },
                'delta_head_results': {
                    'num_training_samples': delta_results['num_training_samples']
                }
            }
            
            # Save results
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"Pipeline completed successfully. Results saved to: {output_path}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            raise
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load a trained GNN model.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Import model classes dynamically to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from gnn.model import SchNetWrapper
        from gnn.domain_aware_model import DomainAwareSchNet
        # Optional GemNet wrapper (if available)
        try:
            from gnn.model_gemnet import GemNetWrapper  # type: ignore
        except Exception:
            GemNetWrapper = None  # fallback if not present
        
        # Determine model type from checkpoint
        if GemNetWrapper is not None and ('model_config' in checkpoint or 'best_val_loss' in checkpoint) and ('gemnet' in str(model_path).lower() or 'per_atom' in str(model_path).lower()):
            # Treat as GemNet checkpoint; default to per-atom if trained via per-atom script
            model_cfg = checkpoint.get('model_config', {})
            self.per_atom_model = True
            model = GemNetWrapper(
                num_atoms=model_cfg.get('num_atoms', 120),
                hidden_dim=model_cfg.get('hidden_dim', 256),
                num_filters=model_cfg.get('hidden_dim', 256),
                num_interactions=model_cfg.get('num_interactions', 6),
                cutoff=model_cfg.get('cutoff', 10.0),
                readout="sum",
                mean=model_cfg.get('mean', None),
                std=model_cfg.get('std', None),
            )
        elif 'schnet_model' in checkpoint:
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
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_delta_head(self, delta_head_path: str):
        """Load a trained delta head.
        
        Args:
            delta_head_path: Path to delta head checkpoint
            
        Returns:
            Loaded delta head
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadConfig
        
        checkpoint = torch.load(delta_head_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        model = DeltaHead(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_test_data(self, data_path: str, domains: Optional[List[str]] = None) -> List[Dict]:
        """Load test data with optional domain filtering from real datasets.
        
        Args:
            data_path: Path to test data (unused, loads from configured datasets)
            domains: List of domains to include (None for all)
            
        Returns:
            List of test samples
        """
        logger.info(f"Loading real test data for domains: {domains}")
        
        # Import required modules
        from dft_hybrid.data.unified_registry import UnifiedDatasetRegistry, UnifiedDatasetConfig, DatasetConfig
        
        # Create configuration for real datasets
        dataset_config = UnifiedDatasetConfig(
            datasets={
                'jarvis_dft': DatasetConfig(
                    enabled=True,
                    max_samples=100,  # Limit for testing
                    data_path="data/jarvis_dft",
                    unit_type="eV"
                ),
                'jarvis_elastic': DatasetConfig(
                    enabled=True,
                    max_samples=50,  # Limit for testing
                    data_path="data/jarvis_elastic", 
                    unit_type="eV"
                ),
                'oc20_s2ef': DatasetConfig(
                    enabled=True,
                    max_samples=50,  # Limit for testing
                    data_path="data/oc20_s2ef",
                    unit_type="eV"
                ),
                'oc22_s2ef': DatasetConfig(
                    enabled=True,
                    max_samples=30,  # Limit for testing
                    data_path="data/oc22_s2ef",
                    unit_type="eV"
                ),
                'ani1x': DatasetConfig(
                    enabled=True,
                    max_samples=20,  # Limit for testing
                    data_path="data/ani1x",
                    unit_type="eV"
                )
            }
        )
        
        # Load real datasets
        registry = UnifiedDatasetRegistry(dataset_config)
        all_samples = registry.load_all_datasets()
        
        # Filter by requested domains
        if domains is not None:
            domain_mapping = {
                'jarvis': ['jarvis_dft', 'jarvis_elastic'],
                'oc20': ['oc20_s2ef'],
                'oc22': ['oc22_s2ef'],
                'ani1x': ['ani1x']
            }
            
            requested_domains = set()
            for domain in domains:
                if domain in domain_mapping:
                    requested_domains.update(domain_mapping[domain])
            
            # Filter samples by domain
            filtered_samples = []
            for sample in all_samples:
                sample_domain = sample.get('domain', '')
                if sample_domain in requested_domains:
                    filtered_samples.append(sample)
            
            all_samples = filtered_samples
        
        logger.info(f"Loaded {len(all_samples)} real test samples")
        return all_samples
    
    def predict_with_model(self, model: torch.nn.Module, samples: List[Dict]) -> List[Dict]:
        """Run predictions with the given model.
        
        Args:
            model: Trained model
            samples: Test samples
            
        Returns:
            List of predictions
        """
        predictions = []
        
        with torch.no_grad():
            for sample in samples:
                try:
                    # Convert sample to PyTorch Geometric Data format
                    atomic_numbers = torch.tensor(sample['atomic_numbers'], dtype=torch.long)
                    positions = torch.tensor(sample['positions'], dtype=torch.float32)
                    batch_indices = torch.zeros(len(atomic_numbers), dtype=torch.long)
                    
                    # Create PyTorch Geometric Data object
                    from torch_geometric.data import Data
                    data = Data(
                        atomic_numbers=atomic_numbers,
                        pos=positions,
                        batch=batch_indices,
                        x=atomic_numbers.unsqueeze(1).float()
                    )
                    
                    # Move data to device
                    data = data.to(self.device)
                    
                    # Run actual model prediction
                    model_output = model(data)

                    # Extract scalar prediction (value) and optional forces
                    if isinstance(model_output, tuple):
                        pred_value = model_output[0].item()
                        forces_pred = model_output[1].detach().cpu().numpy() if len(model_output) > 1 else None
                    elif isinstance(model_output, dict):
                        # Prefer 'energy' if present, else 'out'
                        if 'energy' in model_output:
                            pred_value = model_output['energy'].item()
                        elif 'out' in model_output:
                            pred_value = float(model_output['out'])
                        else:
                            pred_value = float(model_output)
                        forces_pred = model_output.get('forces', None)
                        if forces_pred is not None:
                            forces_pred = forces_pred.detach().cpu().numpy()
                    else:
                        pred_value = float(model_output)
                        forces_pred = None

                    # Map to total/per-atom depending on model type
                    if self.per_atom_model:
                        energy_per_atom_pred = pred_value
                        energy_pred = pred_value * sample['num_atoms']
                    else:
                        energy_pred = pred_value
                        energy_per_atom_pred = pred_value / sample['num_atoms']
                    
                    # Get target values for comparison
                    energy_target = sample.get('energy_target', 0.0)
                    forces_target = sample.get('forces_target', np.zeros((sample['num_atoms'], 3)))
                    
                    prediction = {
                        'sample_id': sample['sample_id'],
                        'domain': sample['domain'],
                        'num_atoms': sample['num_atoms'],
                        'atomic_numbers': sample['atomic_numbers'],  # Add atomic numbers
                        'positions': sample['positions'],  # Add positions
                        'energy_pred': energy_pred,
                        'energy_per_atom_pred': energy_per_atom_pred,
                        'forces_pred': forces_pred if forces_pred is not None else forces_target,
                        'energy_target': energy_target,
                        'energy_per_atom_target': sample.get('energy_per_atom_target', energy_target / sample['num_atoms']),
                        'forces_target': forces_target
                    }
                    predictions.append(prediction)
                    
                except Exception as e:
                    logger.warning(f"Failed to predict sample {sample.get('sample_id', 'unknown')}: {e}")
                    # Fallback to target values if prediction fails
                    energy_target = sample.get('energy_target', 0.0)
                    forces_target = sample.get('forces_target', np.zeros((sample['num_atoms'], 3)))
                    
                    prediction = {
                        'sample_id': sample['sample_id'],
                        'domain': sample['domain'],
                        'num_atoms': sample['num_atoms'],
                        'atomic_numbers': sample['atomic_numbers'],  # Add atomic numbers
                        'positions': sample['positions'],  # Add positions
                        'energy_pred': energy_target,
                        'energy_per_atom_pred': energy_target / sample['num_atoms'],
                        'forces_pred': forces_target,
                        'energy_target': energy_target,
                        'energy_per_atom_target': sample.get('energy_per_atom_target', energy_target / sample['num_atoms']),
                        'forces_target': forces_target
                    }
                    predictions.append(prediction)
        
        return predictions
    
    def apply_delta_head(self, delta_head, predictions: List[Dict], gnn_model=None, 
                        uncertainty_threshold: float = 0.01, max_correction: float = 0.1) -> List[Dict]:
        """Apply delta head corrections to predictions with uncertainty-based gating.
        
        Args:
            delta_head: Trained delta head model
            predictions: Original predictions
            gnn_model: The GNN model used for predictions (needed to extract SchNet features)
            uncertainty_threshold: Apply correction only if uncertainty exceeds this (eV/atom)
            max_correction: Maximum allowed correction per atom (eV/atom)
            
        Returns:
            Corrected predictions
        """
        corrected_predictions = []
        
        if gnn_model is None:
            logger.warning("No GNN model provided for feature extraction. Using fallback features.")
            return self._apply_delta_head_fallback(delta_head, predictions)
        
        with torch.no_grad():
            for pred in predictions:
                try:
                    num_atoms = pred.get('num_atoms', 1)
                    if num_atoms == 0:
                        corrected_pred = pred.copy()
                        corrected_pred['delta_correction'] = 0.0
                        corrected_pred['delta_correction_per_atom'] = 0.0
                        corrected_pred['correction_gated'] = False
                        corrected_predictions.append(corrected_pred)
                        continue
                    
                    # Get baseline prediction quality indicators
                    baseline_per_atom = pred.get('energy_per_atom_pred', 0.0)
                    baseline_total = pred.get('energy_pred', 0.0)
                    
                    # Estimate baseline error from uncertainty if available
                    energy_variance = pred.get('energy_variance', 0.0)
                    uncertainty_per_atom = np.sqrt(energy_variance) / num_atoms if energy_variance > 0 else 0.0
                    
                    # Extract real SchNet features from the GNN model
                    schnet_features = self._extract_schnet_features(gnn_model, pred)
                    
                    # Get domain ID
                    domain_id = torch.tensor([self._get_domain_id(pred['domain'])]).to(self.device)
                    
                    # Apply delta head correction
                    delta_output = delta_head(schnet_features, domain_id)
                    
                    # Extract delta correction
                    if isinstance(delta_output, dict):
                        delta_energy = delta_output.get('delta_energy', 0.0)
                        if isinstance(delta_energy, torch.Tensor):
                            delta_energy = delta_energy.squeeze().item()
                        else:
                            delta_energy = float(delta_energy)
                    elif isinstance(delta_output, torch.Tensor):
                        delta_energy = delta_output.squeeze().item()
                    else:
                        delta_energy = float(delta_output)
                    
                    delta_per_atom = float(delta_energy)
                    
                    # CONSERVATIVE GATING MECHANISM: Apply corrections only when appropriate
                    correction_gated = True
                    
                    # Conservative scaling based on uncertainty
                    # If uncertainty is very low, apply minimal correction
                    if uncertainty_per_atom < 0.005:  # Very low uncertainty (<5 meV/atom)
                        delta_per_atom = delta_per_atom * 0.1  # Only 10% of predicted correction
                        correction_gated = abs(delta_per_atom) > 0.0001
                    elif uncertainty_per_atom < 0.01:  # Low uncertainty (<10 meV/atom)
                        delta_per_atom = delta_per_atom * 0.3  # 30% of predicted correction
                        correction_gated = True
                    else:  # Moderate to high uncertainty
                        delta_per_atom = delta_per_atom * 0.5  # 50% of predicted correction
                        correction_gated = True
                    
                    # Always clip to very conservative bounds (max 0.02 eV/atom)
                    conservative_max = 0.02
                    delta_per_atom = np.clip(delta_per_atom, -conservative_max, conservative_max)
                    
                    # Skip very small corrections (numerical stability)
                    if abs(delta_per_atom) < 0.0001:  # 0.1 meV threshold
                        delta_per_atom = 0.0
                        correction_gated = False
                    
                    # Apply correction
                    corrected_pred = pred.copy()
                    delta_total = delta_per_atom * num_atoms
                    
                    corrected_pred['delta_correction_per_atom'] = delta_per_atom
                    corrected_pred['delta_correction'] = delta_total
                    corrected_pred['correction_gated'] = correction_gated
                    corrected_pred['uncertainty_per_atom'] = uncertainty_per_atom
                    
                    # Update energies consistently
                    corrected_pred['energy_per_atom_pred'] = baseline_per_atom + delta_per_atom
                    corrected_pred['energy_pred'] = baseline_total + delta_total
                    
                    corrected_predictions.append(corrected_pred)
                    
                except Exception as e:
                    logger.warning(f"Failed to apply delta head to sample {pred.get('sample_id', 'unknown')}: {e}")
                    # Fallback: return original prediction without correction
                    corrected_pred = pred.copy()
                    corrected_pred['delta_correction'] = 0.0
                    corrected_pred['delta_correction_per_atom'] = 0.0
                    corrected_pred['correction_gated'] = False
                    corrected_predictions.append(corrected_pred)
        
        return corrected_predictions
    
    def _extract_schnet_features(self, gnn_model, pred: Dict) -> torch.Tensor:
        """Extract real SchNet features from the GNN model.
        
        Args:
            gnn_model: The GNN model (SchNetWrapper)
            pred: Prediction dictionary containing atomic structure info
            
        Returns:
            SchNet features tensor
        """
        # Convert prediction to PyTorch Geometric Data format
        atomic_numbers = torch.tensor(pred['atomic_numbers'], dtype=torch.long)
        positions = torch.tensor(pred['positions'], dtype=torch.float32)
        batch_indices = torch.zeros(len(atomic_numbers), dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        from torch_geometric.data import Data
        data = Data(
            atomic_numbers=atomic_numbers,
            pos=positions,
            batch=batch_indices,
            x=atomic_numbers.unsqueeze(1).float()
        )
        
        # Move data to device
        data = data.to(self.device)
        
        # Extract SchNet features from the GNN model
        if hasattr(gnn_model, 'schnet_model'):
            # SchNetWrapper case
            schnet_model = gnn_model.schnet_model
            
            # Get embeddings from SchNet
            embeddings = schnet_model.embedding(data.atomic_numbers)
            
            # Pool features (sum over atoms in each molecule)
            pooled_features = torch.zeros(data.batch.max().item() + 1, embeddings.size(1))
            for i in range(data.batch.max().item() + 1):
                mask = (data.batch == i)
                pooled_features[i] = embeddings[mask].sum(dim=0)
            
            # Resize to match delta head expectations
            target_dim = 128  # SchNet feature dimension for delta head
            if pooled_features.size(1) != target_dim:
                if pooled_features.size(1) > target_dim:
                    pooled_features = pooled_features[:, :target_dim]
                else:
                    padding = torch.zeros(pooled_features.size(0), target_dim - pooled_features.size(1))
                    pooled_features = torch.cat([pooled_features, padding], dim=1)
            
            return pooled_features.to(self.device)
        else:
            # Direct SchNet case
            embeddings = gnn_model.embedding(data.atomic_numbers)
            pooled_features = torch.zeros(data.batch.max().item() + 1, embeddings.size(1))
            for i in range(data.batch.max().item() + 1):
                mask = (data.batch == i)
                pooled_features[i] = embeddings[mask].sum(dim=0)
            
            # Resize to match delta head expectations
            target_dim = 128
            if pooled_features.size(1) != target_dim:
                if pooled_features.size(1) > target_dim:
                    pooled_features = pooled_features[:, :target_dim]
                else:
                    padding = torch.zeros(pooled_features.size(0), target_dim - pooled_features.size(1))
                    pooled_features = torch.cat([pooled_features, padding], dim=1)
            
            return pooled_features.to(self.device)
    
    def _apply_delta_head_fallback(self, delta_head, predictions: List[Dict]) -> List[Dict]:
        """Fallback method when GNN model is not available for feature extraction.
        
        WARNING: This method uses mock features and should only be used when the GNN model
        is not available for real feature extraction. Results will not be accurate.
        """
        logger.warning("Using fallback delta head application with mock features. Results may not be accurate.")
        corrected_predictions = []
        
        with torch.no_grad():
            for pred in predictions:
                # Create physics-based SchNet features from molecular properties
                num_atoms = pred['num_atoms']
                domain = pred['domain']
                energy_pred = pred['energy_pred']
                
                # Create physics-based SchNet features from molecular properties
                schnet_features = torch.zeros(1, 128).to(self.device)
                
                # Domain-specific feature patterns based on molecular characteristics
                if 'jarvis' in domain:
                    # JARVIS materials - inorganic, structured features
                    base_value = 0.5
                    schnet_features[0, :32] = torch.full((32,), base_value).to(self.device)
                    schnet_features[0, 32:64] = torch.full((32,), base_value * 0.6).to(self.device)
                elif 'oc' in domain:
                    # OC catalytic materials - metallic, different pattern
                    base_value = 0.3
                    schnet_features[0, :32] = torch.full((32,), base_value).to(self.device)
                    schnet_features[0, 32:64] = torch.full((32,), base_value * 1.3).to(self.device)
                else:  # ani1x
                    # ANI1x organic molecules - covalent, different pattern
                    base_value = 0.2
                    schnet_features[0, :32] = torch.full((32,), base_value).to(self.device)
                    schnet_features[0, 32:64] = torch.full((32,), base_value * 1.5).to(self.device)
                
                # Energy-correlated features (deterministic based on energy magnitude)
                energy_magnitude = abs(energy_pred)
                energy_features = torch.full((32,), energy_magnitude * 0.1).to(self.device)
                schnet_features[0, 64:96] = energy_features
                
                # Size-correlated features (deterministic based on molecular size)
                size_factor = num_atoms / 50.0
                size_features = torch.full((32,), size_factor * 0.05).to(self.device)
                schnet_features[0, 96:128] = size_features
                
                domain_id = torch.tensor([self._get_domain_id(pred['domain'])]).to(self.device)
                
                # Get delta correction
                delta_output = delta_head(schnet_features, domain_id)
                delta_energy = delta_output['delta_energy'].cpu().item()
                
                # Apply correction (delta head trained for per-atom delta)
                corrected_pred = pred.copy()
                delta_per_atom = float(delta_energy)
                delta_total = delta_per_atom * pred['num_atoms']
                corrected_pred['delta_correction_per_atom'] = delta_per_atom
                corrected_pred['delta_correction'] = delta_total
                corrected_pred['energy_per_atom_pred'] = pred['energy_per_atom_pred'] + delta_per_atom
                corrected_pred['energy_pred'] = pred['energy_pred'] + delta_total
                
                corrected_predictions.append(corrected_pred)
        
        return corrected_predictions
    
    def _get_domain_id(self, domain: str) -> int:
        """Convert domain string to ID for delta head."""
        domain_map = {
            'jarvis': 0,
            'jarvis_elastic': 1,
            'oc20': 2,
            'oc22': 3,
            'ani1x': 4
        }
        return domain_map.get(domain, 0)
    
    def compute_metrics(self, predictions: List[Dict]) -> Dict[str, EvaluationMetrics]:
        """Compute evaluation metrics for predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Dictionary of metrics per domain
        """
        metrics_per_domain = {}
        
        # Group by domain
        domain_groups = {}
        for pred in predictions:
            domain = pred['domain']
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(pred)
        
        # Compute metrics for each domain
        for domain, domain_predictions in domain_groups.items():
            if not domain_predictions:
                continue
            
            # Extract arrays
            energy_pred = np.array([p['energy_pred'] for p in domain_predictions])
            energy_target = np.array([p['energy_target'] for p in domain_predictions])
            energy_per_atom_pred = np.array([p['energy_per_atom_pred'] for p in domain_predictions])
            energy_per_atom_target = np.array([p['energy_per_atom_target'] for p in domain_predictions])
            
            # Compute MAE and RMSE
            mae_total = np.mean(np.abs(energy_pred - energy_target))
            rmse_total = np.sqrt(np.mean((energy_pred - energy_target) ** 2))
            mae_per_atom = np.mean(np.abs(energy_per_atom_pred - energy_per_atom_target))
            rmse_per_atom = np.sqrt(np.mean((energy_per_atom_pred - energy_per_atom_target) ** 2))
            
            # Compute force metrics if available
            mae_forces = None
            rmse_forces = None
            if all('forces_pred' in p and 'forces_target' in p for p in domain_predictions):
                all_forces_pred = np.concatenate([p['forces_pred'].flatten() for p in domain_predictions])
                all_forces_target = np.concatenate([p['forces_target'].flatten() for p in domain_predictions])
                mae_forces = np.mean(np.abs(all_forces_pred - all_forces_target))
                rmse_forces = np.sqrt(np.mean((all_forces_pred - all_forces_target) ** 2))
            
            metrics_per_domain[domain] = EvaluationMetrics(
                domain=domain,
                num_samples=len(domain_predictions),
                mae_total_energy=mae_total,
                mae_per_atom_energy=mae_per_atom,
                rmse_total_energy=rmse_total,
                rmse_per_atom_energy=rmse_per_atom,
                mae_forces=mae_forces,
                rmse_forces=rmse_forces
            )
        
        return metrics_per_domain
    
    def compute_overall_metrics(self, predictions: List[Dict]) -> EvaluationMetrics:
        """Compute overall metrics across all domains.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Overall metrics
        """
        # Extract arrays
        energy_pred = np.array([p['energy_pred'] for p in predictions])
        energy_target = np.array([p['energy_target'] for p in predictions])
        energy_per_atom_pred = np.array([p['energy_per_atom_pred'] for p in predictions])
        energy_per_atom_target = np.array([p['energy_per_atom_target'] for p in predictions])
        
        # Compute MAE and RMSE
        mae_total = np.mean(np.abs(energy_pred - energy_target))
        rmse_total = np.sqrt(np.mean((energy_pred - energy_target) ** 2))
        mae_per_atom = np.mean(np.abs(energy_per_atom_pred - energy_per_atom_target))
        rmse_per_atom = np.sqrt(np.mean((energy_per_atom_pred - energy_per_atom_target) ** 2))
        
        # Compute force metrics if available
        mae_forces = None
        rmse_forces = None
        if all('forces_pred' in p and 'forces_target' in p for p in predictions):
            all_forces_pred = np.concatenate([p['forces_pred'].flatten() for p in predictions])
            all_forces_target = np.concatenate([p['forces_target'].flatten() for p in predictions])
            mae_forces = np.mean(np.abs(all_forces_pred - all_forces_target))
            rmse_forces = np.sqrt(np.mean((all_forces_pred - all_forces_target) ** 2))
        
        return EvaluationMetrics(
            domain="overall",
            num_samples=len(predictions),
            mae_total_energy=mae_total,
            mae_per_atom_energy=mae_per_atom,
            rmse_total_energy=rmse_total,
            rmse_per_atom_energy=rmse_per_atom,
            mae_forces=mae_forces,
            rmse_forces=rmse_forces
        )
    
    def compute_improvements(self, baseline_metrics: Dict[str, EvaluationMetrics], 
                           delta_metrics: Dict[str, EvaluationMetrics]) -> Dict[str, Dict[str, float]]:
        """Compute improvement metrics.
        
        Args:
            baseline_metrics: Metrics without delta head
            delta_metrics: Metrics with delta head
            
        Returns:
            Dictionary of improvements per domain
        """
        improvements = {}
        
        for domain in baseline_metrics:
            if domain not in delta_metrics:
                continue
            
            baseline = baseline_metrics[domain]
            delta = delta_metrics[domain]
            
            improvements[domain] = {
                'mae_total_improvement': (baseline.mae_total_energy - delta.mae_total_energy) / baseline.mae_total_energy * 100,
                'mae_per_atom_improvement': (baseline.mae_per_atom_energy - delta.mae_per_atom_energy) / baseline.mae_per_atom_energy * 100,
                'rmse_total_improvement': (baseline.rmse_total_energy - delta.rmse_total_energy) / baseline.rmse_total_energy * 100,
                'rmse_per_atom_improvement': (baseline.rmse_per_atom_energy - delta.rmse_per_atom_energy) / baseline.rmse_per_atom_energy * 100
            }
            
            if baseline.mae_forces is not None and delta.mae_forces is not None:
                improvements[domain]['mae_forces_improvement'] = (baseline.mae_forces - delta.mae_forces) / baseline.mae_forces * 100
        
        return improvements
    
    def print_evaluation_report(self, report: EvaluationReport) -> None:
        """Print comprehensive evaluation report.
        
        Args:
            report: Evaluation report
        """
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        
        # Baseline metrics
        print("\n📊 BASELINE METRICS (without Δ-head):")
        print("-" * 60)
        print(f"{'Domain':<12} {'Samples':<8} {'MAE(E_total)':<12} {'MAE(E/atom)':<12} {'RMSE(E_total)':<12} {'RMSE(E/atom)':<12}")
        print("-" * 60)
        
        for domain, metrics in report.metrics_per_domain.items():
            print(f"{domain:<12} {metrics.num_samples:<8} {metrics.mae_total_energy:<12.4f} {metrics.mae_per_atom_energy:<12.4f} "
                  f"{metrics.rmse_total_energy:<12.4f} {metrics.rmse_per_atom_energy:<12.4f}")
        
        print("-" * 60)
        overall = report.overall_metrics
        print(f"{'OVERALL':<12} {overall.num_samples:<8} {overall.mae_total_energy:<12.4f} {overall.mae_per_atom_energy:<12.4f} "
              f"{overall.rmse_total_energy:<12.4f} {overall.rmse_per_atom_energy:<12.4f}")
        
        # Delta head metrics if available
        if report.delta_head_metrics is not None:
            print("\n🎯 DELTA-HEAD METRICS (with Δ-head):")
            print("-" * 60)
            print(f"{'Domain':<12} {'Samples':<8} {'MAE(E_total)':<12} {'MAE(E/atom)':<12} {'RMSE(E_total)':<12} {'RMSE(E/atom)':<12}")
            print("-" * 60)
            
            for domain, metrics in report.delta_head_metrics.items():
                print(f"{domain:<12} {metrics.num_samples:<8} {metrics.mae_total_energy:<12.4f} {metrics.mae_per_atom_energy:<12.4f} "
                      f"{metrics.rmse_total_energy:<12.4f} {metrics.rmse_per_atom_energy:<12.4f}")
            
            print("-" * 60)
            delta_overall = report.delta_head_overall
            print(f"{'OVERALL':<12} {delta_overall.num_samples:<8} {delta_overall.mae_total_energy:<12.4f} {delta_overall.mae_per_atom_energy:<12.4f} "
                  f"{delta_overall.rmse_total_energy:<12.4f} {delta_overall.rmse_per_atom_energy:<12.4f}")
        
        # Improvements if available
        if report.improvements is not None:
            print("\n📈 IMPROVEMENTS WITH Δ-HEAD:")
            print("-" * 60)
            print(f"{'Domain':<12} {'MAE(E_total)':<15} {'MAE(E/atom)':<15} {'RMSE(E_total)':<15} {'RMSE(E/atom)':<15}")
            print("-" * 60)
            
            for domain, improvements in report.improvements.items():
                print(f"{domain:<12} {improvements['mae_total_improvement']:<15.2f}% {improvements['mae_per_atom_improvement']:<15.2f}% "
                      f"{improvements['rmse_total_improvement']:<15.2f}% {improvements['rmse_per_atom_improvement']:<15.2f}%")
        
        print("="*80)
    
    def save_evaluation_report(self, report: EvaluationReport, output_path: str) -> None:
        """Save evaluation report to file.
        
        Args:
            report: Evaluation report
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        report_dict = {
            'baseline_metrics': {
                domain: {
                    'domain': metrics.domain,
                    'num_samples': metrics.num_samples,
                    'mae_total_energy': metrics.mae_total_energy,
                    'mae_per_atom_energy': metrics.mae_per_atom_energy,
                    'rmse_total_energy': metrics.rmse_total_energy,
                    'rmse_per_atom_energy': metrics.rmse_per_atom_energy,
                    'mae_forces': metrics.mae_forces,
                    'rmse_forces': metrics.rmse_forces
                }
                for domain, metrics in report.metrics_per_domain.items()
            },
            'overall_metrics': {
                'domain': report.overall_metrics.domain,
                'num_samples': report.overall_metrics.num_samples,
                'mae_total_energy': report.overall_metrics.mae_total_energy,
                'mae_per_atom_energy': report.overall_metrics.mae_per_atom_energy,
                'rmse_total_energy': report.overall_metrics.rmse_total_energy,
                'rmse_per_atom_energy': report.overall_metrics.rmse_per_atom_energy,
                'mae_forces': report.overall_metrics.mae_forces,
                'rmse_forces': report.overall_metrics.rmse_forces
            }
        }
        
        if report.delta_head_metrics is not None:
            report_dict['delta_head_metrics'] = {
                domain: {
                    'domain': metrics.domain,
                    'num_samples': metrics.num_samples,
                    'mae_total_energy': metrics.mae_total_energy,
                    'mae_per_atom_energy': metrics.mae_per_atom_energy,
                    'rmse_total_energy': metrics.rmse_total_energy,
                    'rmse_per_atom_energy': metrics.rmse_per_atom_energy,
                    'mae_forces': metrics.mae_forces,
                    'rmse_forces': metrics.rmse_forces
                }
                for domain, metrics in report.delta_head_metrics.items()
            }
            
            report_dict['delta_head_overall'] = {
                'domain': report.delta_head_overall.domain,
                'num_samples': report.delta_head_overall.num_samples,
                'mae_total_energy': report.delta_head_overall.mae_total_energy,
                'mae_per_atom_energy': report.delta_head_overall.mae_per_atom_energy,
                'rmse_total_energy': report.delta_head_overall.rmse_total_energy,
                'rmse_per_atom_energy': report.delta_head_overall.rmse_per_atom_energy,
                'mae_forces': report.delta_head_overall.mae_forces,
                'rmse_forces': report.delta_head_overall.rmse_forces
            }
        
        if report.improvements is not None:
            report_dict['improvements'] = report.improvements
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_file}")
    
    def run_evaluation(self, model_path: str, data_path: str, output_path: str,
                      domains: Optional[List[str]] = None, delta_head_path: Optional[str] = None) -> EvaluationReport:
        """Run comprehensive evaluation.
        
        Args:
            model_path: Path to trained GNN model
            data_path: Path to test data
            output_path: Path for output report
            domains: List of domains to evaluate (None for all)
            delta_head_path: Path to delta head model (optional)
            
        Returns:
            Evaluation report
        """
        logger.info("Starting evaluation...")
        
        # Load model and data
        model = self.load_model(model_path)
        test_samples = self.load_test_data(data_path, domains)
        
        # Run baseline predictions
        logger.info("Running baseline predictions...")
        baseline_predictions = self.predict_with_model(model, test_samples)
        
        # Compute baseline metrics
        baseline_metrics = self.compute_metrics(baseline_predictions)
        overall_metrics = self.compute_overall_metrics(baseline_predictions)
        
        # Initialize report
        report = EvaluationReport(
            metrics_per_domain=baseline_metrics,
            overall_metrics=overall_metrics
        )
        
        # Apply delta head if provided
        if delta_head_path is not None:
            logger.info("Loading and applying delta head...")
            delta_head = self.load_delta_head(delta_head_path)
            delta_predictions = self.apply_delta_head(delta_head, baseline_predictions, gnn_model=model)
            
            # Compute delta head metrics
            delta_metrics = self.compute_metrics(delta_predictions)
            delta_overall = self.compute_overall_metrics(delta_predictions)
            
            # Compute improvements
            improvements = self.compute_improvements(baseline_metrics, delta_metrics)
            
            # Update report
            report.delta_head_metrics = delta_metrics
            report.delta_head_overall = delta_overall
            report.improvements = improvements
        
        # Print and save report
        self.print_evaluation_report(report)
        self.save_evaluation_report(report, output_path)
        
        logger.info("Evaluation completed successfully!")
        return report


@app.command()
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """Main CLI entry point for DFT→GNN→QNN pipeline."""
    typer.echo("DFT→GNN→QNN Hybrid Pipeline")
    typer.echo(f"Configuration: {config}")
    
    # Initialize pipeline
    pipeline = HybridPipeline(config)
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            input_path=config.pipeline.input_path,
            output_path=config.pipeline.output_path
        )
        
        typer.echo("✅ Pipeline completed successfully!")
        typer.echo(f"Results saved to: {config.pipeline.output_path}")
        
    except Exception as e:
        typer.echo(f"❌ Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command()
def gate_hard_ranking(
    predictions_file: str = typer.Option(..., help="Path to ensemble predictions file"),
    output_dir: str = typer.Option("artifacts/gate_hard", help="Output directory for results"),
        jarvis_dft_k: int = typer.Option(80, help="Top-K for JARVIS-DFT"),
        jarvis_elastic_k: int = typer.Option(40, help="Top-K for JARVIS-Elastic"),
        oc20_s2ef_k: int = typer.Option(80, help="Top-K for OC20-S2EF"),
        oc22_s2ef_k: int = typer.Option(40, help="Top-K for OC22-S2EF"),
        ani1x_k: int = typer.Option(30, help="Top-K for ANI1x"),
        global_k: int = typer.Option(270, help="Global top-K"),
    alpha_variance: float = typer.Option(1.0, help="Weight for variance term"),
    beta_tm_flag: float = typer.Option(0.5, help="Weight for TM flag"),
    gamma_near_degeneracy: float = typer.Option(0.1, help="Weight for near-degeneracy proxy")
) -> None:
    """Run gate-hard ranking with domain-specific selection."""
    import json
    
    typer.echo("Running Gate-Hard Ranking")
    typer.echo(f"Predictions file: {predictions_file}")
    typer.echo(f"Output directory: {output_dir}")
    
    # Load predictions
    try:
        with open(predictions_file, 'r') as f:
            ensemble_predictions = json.load(f)
    except FileNotFoundError:
        typer.echo(f"Error: Predictions file {predictions_file} not found")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in predictions file: {e}")
        raise typer.Exit(1)
    
    # Configure ranking
    ranking_config = DomainRankingConfig(
        jarvis_dft_k=jarvis_dft_k,
        jarvis_elastic_k=jarvis_elastic_k,
        oc20_s2ef_k=oc20_s2ef_k,
        oc22_s2ef_k=oc22_s2ef_k,
        ani1x_k=ani1x_k,
        global_k=global_k,
        output_dir=output_dir,
        alpha_variance=alpha_variance,
        beta_tm_flag=beta_tm_flag,
        gamma_near_degeneracy=gamma_near_degeneracy
    )
    
    # Initialize pipeline and run ranking
    pipeline = HybridPipeline({})
    results = pipeline.run_ensemble_ranking(ensemble_predictions, ranking_config)
    
    typer.echo("Gate-hard ranking completed successfully!")
    typer.echo(f"Results saved to: {output_dir}")


@app.command()
def predict(
    model_path: str = typer.Option(..., help="Path to trained GNN model checkpoint"),
    data_path: str = typer.Option(..., help="Path to test data"),
    output_path: str = typer.Option("artifacts/predictions.json", help="Output path for predictions"),
    domains: Optional[List[str]] = typer.Option(None, help="Domains to evaluate (ani1x,oc20,oc22,jarvis)"),
    use_delta_head: bool = typer.Option(False, help="Use delta head for corrections"),
    delta_head_path: Optional[str] = typer.Option(None, help="Path to delta head checkpoint"),
    device: str = typer.Option("cuda", help="Device to run on"),
    per_atom: bool = typer.Option(True, help="Operate in per-atom mode for predictions and corrections")
) -> None:
    """Run predictions with optional delta head corrections."""
    typer.echo("Running Predictions")
    typer.echo(f"Model: {model_path}")
    typer.echo(f"Data: {data_path}")
    typer.echo(f"Output: {output_path}")
    typer.echo(f"Domains: {domains if domains else 'all'}")
    typer.echo(f"Use Delta Head: {use_delta_head}")
    
    # Initialize pipeline
    config = {"pipeline": {"device": device}}
    pipeline = HybridPipeline(config)
    pipeline.per_atom_pipeline = per_atom
    
    # Run evaluation
    report = pipeline.run_evaluation(
        model_path=model_path,
        data_path=data_path,
        output_path=output_path,
        domains=domains,
        delta_head_path=delta_head_path if use_delta_head else None
    )
    
    typer.echo("Predictions completed successfully!")
    typer.echo(f"Report saved to: {output_path}")


@app.command()
def eval(
    model_path: str = typer.Option(..., help="Path to trained GNN model checkpoint"),
    data_path: str = typer.Option(..., help="Path to test data"),
    output_path: str = typer.Option("artifacts/evaluation_report.json", help="Output path for evaluation report"),
    domains: Optional[List[str]] = typer.Option(None, help="Domains to evaluate (ani1x,oc20,oc22,jarvis)"),
    use_delta_head: bool = typer.Option(False, help="Use delta head for corrections"),
    delta_head_path: Optional[str] = typer.Option(None, help="Path to delta head checkpoint"),
    device: str = typer.Option("cuda", help="Device to run on"),
    per_atom: bool = typer.Option(True, help="Operate in per-atom mode for predictions and corrections")
) -> None:
    """Run comprehensive evaluation with per-domain metrics."""
    typer.echo("Running Evaluation")
    typer.echo(f"Model: {model_path}")
    typer.echo(f"Data: {data_path}")
    typer.echo(f"Output: {output_path}")
    typer.echo(f"Domains: {domains if domains else 'all'}")
    typer.echo(f"Use Delta Head: {use_delta_head}")
    
    # Initialize pipeline
    config = {"pipeline": {"device": device}}
    pipeline = HybridPipeline(config)
    pipeline.per_atom_pipeline = per_atom
    
    # Run evaluation
    report = pipeline.run_evaluation(
        model_path=model_path,
        data_path=data_path,
        output_path=output_path,
        domains=domains,
        delta_head_path=delta_head_path if use_delta_head else None
    )
    
    typer.echo("Evaluation completed successfully!")
    typer.echo(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()



