# Dataset Information

## ⚠️ Dataset NOT Included in Repository

**Important**: The JARVIS-DFT dataset is **NOT** included in this repository due to its large size.

## 📥 How to Obtain the Dataset

### Option 1: Download from JARVIS-DFT

1. Visit the JARVIS-DFT database: https://jarvis.nist.gov/jarvisdft/
2. Download the formation energy dataset
3. Follow the preprocessing steps in `docs/FULL_TECHNICAL_REPORT.md`

### Option 2: Use Preprocessed Data

If you have access to preprocessed data:
- Place it in `data/preprocessed_full_unified/`
- Expected files:
  - `train_data.json`
  - `val_data.json`
  - `test_data.json`

## 📋 What IS Included

### Data Split Information
- ✅ `ids_train_val_test.json` - Contains the sample IDs for train/val/test splits
  - This file is **small** (few KB) and is included
  - Allows exact reproduction of data splits
  - Format: `{"id_train": [...], "id_val": [...], "id_test": [...]}`

### Data Preprocessing Scripts
- ✅ `src/data/preprocessing.py` - Data preprocessing code
- ✅ `scripts/preprocess_full_dataset.py` - Preprocessing script
- ✅ `scripts/load_and_preprocess_full_dataset.py` - Data loading script

## 🔧 Setup Instructions

1. **Download JARVIS-DFT dataset** (separately, not in repo)

2. **Preprocess the data**:
   ```bash
   python scripts/preprocess_full_dataset.py
   ```

3. **Verify data splits**:
   ```bash
   python -c "import json; data = json.load(open('ids_train_val_test.json')); print(f'Train: {len(data[\"id_train\"])}, Val: {len(data[\"id_val\"])}, Test: {len(data[\"id_test\"])}')"
   ```

## 📊 Dataset Statistics

**JARVIS-DFT Dataset**:
- Total samples: 36,029 (after cleaning from 37,099)
- Train: 28,823 samples (80%)
- Validation: 3,602 samples (10%)
- Test: 3,604 samples (10%)

**Data Format**:
- JSON files with atomic structures
- Each sample contains:
  - `atomic_numbers`: List of atomic numbers
  - `positions`: Atomic positions (Cartesian coordinates)
  - `formation_energy_per_atom`: Target value (eV/atom)
  - `domain`: Dataset domain identifier

## ⚠️ Why Dataset is Excluded

1. **Size**: Dataset files are very large (hundreds of MB to GB)
2. **GitHub Limits**: GitHub has file size limits (100 MB per file, 1 GB for free repos)
3. **Best Practice**: Large datasets should be hosted separately
4. **Reproducibility**: Data split IDs (`ids_train_val_test.json`) ensure exact reproducibility

## ✅ Reproducibility Guarantee

Even without the dataset in the repo, results are fully reproducible:
- ✅ Data split IDs included (`ids_train_val_test.json`)
- ✅ Preprocessing scripts included
- ✅ Exact split information documented
- ✅ Random seed fixed (42) for reproducibility

## 📚 Additional Resources

- JARVIS-DFT Database: https://jarvis.nist.gov/jarvisdft/
- Dataset Documentation: See `docs/FULL_TECHNICAL_REPORT.md` Section 2
- Preprocessing Guide: See `docs/FULL_TECHNICAL_REPORT.md` Section 3

