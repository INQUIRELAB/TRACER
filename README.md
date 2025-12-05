# TRACER (Transparent and Reliable Accuracy, Confidence, and Error Ranking): A Reliability-First GemNet Baseline for Trustworthy
Computational Materials Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art materials property prediction pipeline using Graph Neural Networks (GNNs) with domain-aware adaptation and novel gate-hard ranking for identifying challenging cases. Achieves **MAE of 0.037 eV/atom** on JARVIS-DFT dataset, outperforming ALIGNN by 25.8% in MAE and 29.3% in RMSE.

## 🎯 Key Results

- **MAE**: 0.037025 eV/atom (25.8% better than ALIGNN) - **single model comparison (fair)**
- **RMSE**: 0.079944 eV/atom (29.3% better than ALIGNN) - **single model comparison (fair)**
- **R²**: 0.993856 (99.4% variance explained)
- **Transfer Learning**: R²=0.964 on Matbench Perovskites after fine-tuning

**Fair Comparison Guarantee**: All main results use **single best models** (not ensemble). Ensemble (3 models) is used only for uncertainty quantification, ensuring fair comparison with ALIGNN baseline.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Gourab562/TRACER.git
cd TRACER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Train GemNet model
python scripts/train_gemnet_per_atom.py train \
    --num-epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --output-dir models/gemnet_per_atom_fixed

# Evaluate model
python scripts/evaluate_gemnet_film.py \
    --model-path models/gemnet_per_atom_fixed/best_model.pt \
    --test-data data/preprocessed_full_unified/test_data.json \
    --batch-size 32 \
    --device cuda

# Run gate-hard ranking
python scripts/run_gate_hard_with_gemnet.py \
    --predictions artifacts/gemnet_predictions/ensemble_predictions.json \
    --output-dir artifacts/gate_hard_gemnet \
    --top-k 270
```

## 📋 Features

### Core Components

1. **GemNet Architecture**: Directional message passing GNN with geometric information
   - Hidden dimension: 256
   - 6 interaction blocks
   - 10.0 Å cutoff radius
   - Sum pooling readout

2. **FiLM Domain Adaptation**: Feature-wise linear modulation for multi-domain support
   - Domain embedding dimension: 16
   - Supports 5 domains (JARVIS-DFT, JARVIS-Elastic, OC20, OC22, ANI1x)

3. **Ensemble Uncertainty Quantification**: Multiple models for confidence-aware predictions
   - Ensemble size: 3 models
   - Uncertainty via ensemble variance

4. **Gate-Hard Ranking System** **(Negative Result)**
   - Multi-factor scoring: uncertainty + chemistry + physics
   - Designed to identify hard cases requiring additional computation
   - In practice, **variance-only ranking outperforms Gate-Hard by ≈10.1%** on this dataset (we report this as a negative result)

5. **Comprehensive Error Analysis**: Systematic breakdown by composition, energy range, and structure size

## 📊 Performance

### JARVIS-DFT Test Set (3,604 samples)

| Model | MAE (eV/atom) | RMSE (eV/atom) | R² |
|-------|---------------|----------------|-----|
| **Our Pipeline** | **0.037025** | **0.079944** | **0.993856** |
| ALIGNN | 0.049761 | 0.112840 | 0.987759 |
| **Improvement** | **-25.8%** | **-29.3%** | **+0.6%** |

### Error Analysis by Category

**By Composition Complexity**:
- Ternary: 2,233 samples, MAE = 0.033 eV/atom (best)
- Binary: 850 samples, MAE = 0.036 eV/atom
- Quaternary: 377 samples, MAE = 0.042 eV/atom
- Elemental: 68 samples, MAE = 0.139 eV/atom

**By Formation Energy Range**:
- 0 to 1 eV/atom: 2,492 samples, MAE = 0.031 eV/atom (best)
- -1 to 0 eV/atom: 581 samples, MAE = 0.045 eV/atom
- < -2 eV/atom: 186 samples, MAE = 0.061 eV/atom

**By Structure Size**:
- < 10 atoms: 2,288 samples, MAE = 0.032 eV/atom (best)
- 10-20 atoms: 840 samples, MAE = 0.036 eV/atom
- 20-50 atoms: 466 samples, MAE = 0.057 eV/atom

### Transfer Learning: Matbench Perovskites

| Metric | Before Fine-Tuning | After Fine-Tuning | Improvement |
|--------|-------------------|-------------------|-------------|
| **MAE** | 0.581 eV/atom | **0.104 eV/atom** | **82% better** |
| **RMSE** | 0.723 eV/atom | **0.139 eV/atom** | **81% better** |
| **R²** | 0.008 | **0.964** | **Excellent fit** |

## 📁 Project Structure

```
TRACER/
├── src/                              # Source code
│   ├── gnn/
│   │   ├── model_gemnet.py          # GemNet + FiLM implementation
│   │   ├── uncertainty.py           # Ensemble uncertainty quantification
│   │   └── train.py                 # Training utilities
│   ├── pipeline/
│   │   ├── run.py                   # Main pipeline orchestrator
│   │   └── gate_hard_ranking.py     # Gate-hard ranking system 
│   ├── data/
│   │   ├── preprocessing.py         # Data preprocessing
│   │   └── io.py                    # Data loading/saving
│   ├── graphs/                       # Graph construction utilities
│   ├── dft_hybrid/                   # DFT-GNN hybrid modules
│   └── config/                       # Configuration files
│
├── scripts/                          # Core training/evaluation scripts
│   ├── train_gemnet_per_atom.py     # Main TRACER training script
│   ├── train_gemnet_baseline.py     # Baseline (no FiLM)
│   ├── train_gemnet_domain_only.py  # Domain-embedding ablation
│   ├── evaluate_gemnet_film.py      # Main evaluation script
│   ├── generate_ensemble_predictions.py # Ensemble predictions for UQ
│   ├── evaluate_uncertainty.py      # Uncertainty-quality evaluation
│   ├── run_gate_hard_with_gemnet.py # Gate-hard / variance-only ranking
│   ├── analyze_errors_by_category.py# Error analysis
│   ├── robustness_ablations.py      # Hyperparameter sensitivity
│   ├── run_ablations.py             # Ablation driver
│   ├── fine_tune_matbench.py        # Transfer learning
│   ├── evaluate_matbench.py         # Matbench evaluation
│   ├── train_alignn_fixed.py        # ALIGNN baseline training
│   └── evaluate_alignn_fixed.py     # ALIGNN baseline evaluation
│
│   # Additional experimental and legacy scripts (not needed for main results)
│   # are kept in `scripts/archive/` for completeness.
│
├── docs/                             # Documentation
│   ├── FULL_TECHNICAL_REPORT.md     # Complete technical report
│   ├── DATA_VERIFICATION_REPORT.md  # Data verification
│   ├── HONESTY_AND_REPRODUCIBILITY.md # Reproducibility statement
│   └── REPRODUCIBILITY_CHECKLIST.md # Reproducibility checklist
│
├── tests/                            # Unit tests
│   ├── test_gnn_stub.py
│   └── test_graphs.py
│
├── figures/                          # Publication-quality figures (PDF + TIFF)
│   ├── figure1_parity_plot.pdf
│   ├── figure2_gate_hard_diagram.pdf
│   └── README.md                    # Figure documentation
│
├── pyproject.toml                   # Package configuration & dependencies
├── Makefile                         # Development commands
├── LICENSE                          # MIT License
├── README.md                        # This file
├── QUICK_START.md                   # Quick start guide
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
├── DATASET_INFORMATION.md           # Dataset download instructions
└── ids_train_val_test.json         # Data split IDs (reproducibility)
```

**Note**: Model checkpoints and large data files are excluded (see `.gitignore`). Users train their own models using the provided scripts.

## 🔬 Methodology

### Architecture

1. **Graph Construction**: Convert atomic structures to PyTorch Geometric graphs
   - Edge connectivity via radius graph (10.0 Å cutoff)
   - Node features: atomic numbers
   - Edge features: distances and normalized vectors

2. **GemNet Model**: Directional message passing
   - Embedding layer: atomic numbers → 256-dim features
   - 6 interaction blocks with geometric information
   - FiLM modulation for domain adaptation (optional)
   - Graph-level sum pooling

3. **Ensemble Training**: Train 3 models with different random seeds
   - Uncertainty quantification via ensemble variance
   - Enables confidence-aware predictions

4. **Gate-Hard Ranking**: Multi-factor scoring
   ```
   Score = α·variance + β·TM_flag + γ·near_degeneracy_proxy
   ```
   - α = 1.0 (uncertainty weight)
   - β = 0.5 (transition metal boost)
   - γ = 0.1 (near-degeneracy boost)

### Training Configuration

- **Learning rate**: 1e-4
- **Batch size**: 16
- **Optimizer**: Adam
- **Weight decay**: 1e-5
- **Epochs**: 50 (with early stopping patience: 10)
- **Loss**: MSE on normalized per-atom energies

## 📚 Dataset

**JARVIS-DFT Dataset**:
- **Total samples**: 36,029 (after cleaning from 37,099)
- **Training**: 28,823 samples (80%)
- **Validation**: 3,602 samples (10%)
- **Test**: 3,604 samples (10%)
- **Source**: NIST JARVIS-DFT database
- **Format**: JSON with atomic structures and formation energies
- **Normalization**: Mean = 0.002190 eV/atom, Std = 1.000787 eV/atom

## 🛠️ Requirements

### Core Dependencies
- Python >= 3.10
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- ASE >= 3.22.0
- PyMatGen >= 2023.12.0

### Optional Dependencies
- **Development tools**: `pytest`, `ruff`, `mypy` - Install with `pip install -e ".[dev]"`
- **Experiment tracking**: `wandb` - Install with `pip install -e ".[wandb]"`

See `pyproject.toml` for complete dependency list.

## 📖 Documentation

- **Full Technical Report**: `docs/FULL_TECHNICAL_REPORT.md`
- **Gate-Hard Ranking**: `GATE_HARD_RANKING.md`
- **Data Verification**: `docs/DATA_VERIFICATION_REPORT.md`

## 🧪 Running Experiments

### Training

```bash
# Train baseline model (no FiLM)
python scripts/train_gemnet_baseline.py train \
    --num-epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --output-dir models/gemnet_baseline

# Train full model (with FiLM)
python scripts/train_gemnet_per_atom.py train \
    --num-epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --output-dir models/gemnet_per_atom_fixed
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate_gemnet_film.py \
    --model-path models/gemnet_per_atom_fixed/best_model.pt \
    --test-data data/preprocessed_full_unified/test_data.json \
    --batch-size 32 \
    --device cuda

# Run ablation study
python scripts/run_ablations.py \
    --test-data data/preprocessed_full_unified/test_data.json \
    --device cuda

# Error analysis by category
python scripts/analyze_errors_by_category.py \
    --model-path models/gemnet_baseline/best_model.pt \
    --test-data data/preprocessed_full_unified/test_data.json \
    --output-dir artifacts/error_analysis \
    --device cuda
```

### Transfer Learning

```bash
# Fine-tune on Matbench Perovskites
python scripts/fine_tune_matbench.py \
    --pretrained-model models/gemnet_baseline/best_model.pt \
    --task perovskites \
    --num-epochs 20 \
    --learning-rate 1e-5 \
    --batch-size 16 \
    --output-dir models/matbench_perovskites_finetuned \
    --device cuda
```

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- JARVIS-DFT dataset from NIST
- PyTorch Geometric for graph neural network framework
- ALIGNN for baseline comparison

## ⚠️ Notes

- **Model Checkpoints**: Pre-trained models are not included due to size. Users should train their own models using the provided scripts.
- **Data**: The JARVIS-DFT dataset should be downloaded separately. See `DATASET_INFORMATION.md` for download instructions.
- **Focus**: This repository focuses on the GNN-based approach with Gate-Hard Ranking. Quantum corrections were explored separately but are not included as they did not improve performance.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [gourab.datta-1@ou.edu].

