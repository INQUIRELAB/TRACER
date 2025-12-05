# Reproducibility Checklist for npj Submission

This checklist ensures all results are reproducible and honestly reported.

## ✅ Data and Splits

- [x] **Fixed random seed**: 42 (for all random operations)
- [x] **Same data splits**: Train (28,823), Val (3,602), Test (3,604)
- [x] **Test set isolation**: Never used during training or hyperparameter tuning
- [x] **Normalization**: Stats computed only on training set
- [x] **Data splits saved**: `ids_train_val_test.json` for reproducibility

## ✅ Model Training

- [x] **Hyperparameters documented**: Learning rate (1e-4), batch size (16), epochs (50)
- [x] **Early stopping**: Based on validation set only (patience: 10)
- [x] **Best model selection**: Based on validation loss, not test set
- [x] **Random seeds fixed**: PyTorch, NumPy, data splits all use seed 42
- [x] **Training scripts available**: `scripts/train_gemnet_per_atom.py`

## ✅ Evaluation

- [x] **Single model results**: Main metrics use single best model (not ensemble)
- [x] **Ensemble usage**: Only for uncertainty quantification, not main metrics
- [x] **Evaluation scripts available**: `scripts/evaluate_gemnet_film.py`
- [x] **Proper denormalization**: Applied consistently
- [x] **Metrics computed identically**: Same formula for all models

## ✅ Fair Comparison

- [x] **ALIGNN comparison**: Single model vs single model (fair)
- [x] **Same test set**: Identical 3,604 samples for both models
- [x] **Same training data**: JARVIS-DFT, 36,029 samples
- [x] **Same target format**: Per-atom formation energy (eV/atom)
- [x] **Same evaluation methodology**: Proper denormalization, identical metrics

## ✅ Honest Reporting

- [x] **Negative results reported**: FiLM minimal benefit (<0.01%)
- [x] **Negative results reported**: Quantum corrections degrade performance
- [x] **Complete ablation**: All variants reported (baseline, domain-only, full)
- [x] **Limitations stated**: Higher computational cost, more complex architecture
- [x] **No cherry-picking**: All results from same test set, all variants included

## ✅ Verification

- [x] **All metrics verified**: From evaluation logs
- [x] **Data verification**: Test set size confirmed (3,604 samples)
- [x] **Model checkpoints**: Paths documented, results verified
- [x] **Ablation results**: All three variants verified
- [x] **Error analysis**: All categories verified

## ✅ Reproducibility Information

- [x] **Seeds documented**: Random seed 42
- [x] **Hyperparameters documented**: All training parameters
- [x] **Evaluation procedure documented**: Step-by-step process
- [x] **Code available**: All scripts in repository
- [x] **Documentation complete**: Full technical report

## ⚠️ Known Limitations

- [ ] **Pre-trained models**: Not included (users must train their own)
- [ ] **Large data files**: Not included (users must download JARVIS-DFT)
- [ ] **Computational requirements**: GPU recommended (RTX 4090 or better)

## 📝 For Reviewers

All results can be reproduced by:
1. Downloading JARVIS-DFT dataset
2. Running `scripts/train_gemnet_per_atom.py` with seed 42
3. Running `scripts/evaluate_gemnet_film.py` on test set
4. Comparing with ALIGNN using same test set

All metrics are computed identically and verified from evaluation logs.

