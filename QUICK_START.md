# Quick Start Guide

Get started with the Domain-Aware GNN pipeline in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

## Installation

```bash
# Clone repository
git clone https://github.com/Gourab562/TRACER.git
cd TRACER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[dev]"
```

## Download Data

The JARVIS-DFT dataset should be downloaded separately. See dataset documentation for details.

## Basic Usage

### 1. Train a Model

```bash
python scripts/train_gemnet_per_atom.py train \
    --num-epochs 50 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --device cuda \
    --output-dir models/my_model
```

### 2. Evaluate

```bash
python scripts/evaluate_gemnet_film.py \
    --model-path models/my_model/best_model.pt \
    --test-data data/preprocessed_full_unified/test_data.json \
    --device cuda
```

### 3. Run Gate-Hard Ranking

```bash
python scripts/run_gate_hard_with_gemnet.py \
    --predictions artifacts/predictions.json \
    --output-dir artifacts/gate_hard \
    --top-k 270
```

## Expected Results

With default settings, you should achieve:
- **MAE**: ~0.037 eV/atom
- **RMSE**: ~0.080 eV/atom
- **R²**: ~0.994

## Troubleshooting

**Out of Memory?**
- Reduce batch size: `--batch-size 8`
- Use CPU: `--device cpu` (slower)

**Import Errors?**
- Ensure virtual environment is activated
- Reinstall: `pip install -e ".[dev]"`

**Data Not Found?**
- Download JARVIS-DFT dataset
- Update paths in scripts

For more details, see [README.md](README.md) or [docs/FULL_TECHNICAL_REPORT.md](docs/FULL_TECHNICAL_REPORT.md).

