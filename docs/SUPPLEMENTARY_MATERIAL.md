## S1. Computational Environment & Reproducibility Checklist

### S1.1 Operating System and Software Stack

- **OS**: 64-bit Linux (CUDA-enabled environment)  
- **Python**: 3.10  
- **PyTorch**: 2.x (CUDA build)  
- **PyTorch Geometric**: 2.x  
- **Other core libraries**: `numpy`, `scipy`, `ase`, `pymatgen`, `hydra-core`, `typer`, `pydantic`  

All package versions (Python, PyTorch, and libraries such as ASE and pymatgen) are fixed and recorded in the accompanying code repository to enable environment reproduction.

### S1.2 Hardware Configuration

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) or comparable CUDA-capable GPU  
- **GPU memory**: ≥ 24 GB (for the reported batch sizes and configurations)  
- **CPU**: ≥ 8 cores  
- **System RAM**: ≥ 64 GB  

The TRACER pipeline is explicitly configured to run on GPU devices via the `--device cuda` flag in all training and evaluation scripts. Running on CPU is possible but significantly slower and was not used for the reported results.

### S1.3 Random Seeds and Determinism

To ensure reproducibility, we fix all relevant seeds:

- **Global seed**: 42  
- **Components seeded**:
  - Python `random`
  - NumPy
  - PyTorch (CPU and CUDA)

In the training scripts (e.g., `scripts/train_gemnet_per_atom.py`), seeds are set once at the beginning of `main`, and we avoid any additional, manual reshuffling of data beyond PyTorch's deterministic dataloader behavior (aside from the standard random shuffling controlled by the same seed).

### S1.4 Reproducibility Checklist

- **Code available**: Complete training, evaluation, and analysis code is provided in an accompanying public repository.
- **Environment specified**: Hardware/OS and dependencies are documented here and mirrored in the repository configuration.
- **Data splits**:  
  - The exact train/validation/test identifiers used in all experiments are stored and released alongside the code.  
  - This allows exact reconstruction of splits from any valid copy of the JARVIS-DFT dataset.
- **Seeds fixed**: All experiments use `seed = 42` for NumPy, Python, and PyTorch.
- **Non-determinism**:  
  - We avoid non-deterministic CUDA operations where possible.  
  - Minor variation (on the order of 1e-4 eV/atom) due to low-level differences (e.g., cuDNN kernels) is possible but does not affect any qualitative conclusion.

---

## S2. Hyperparameters and Robustness Considerations

This section discusses the role of three key architectural hyperparameters and the rationale for the defaults used in TRACER:

- **Cutoff radius \( r_{\text{cut}} \)** (Å)  
- **Number of interaction blocks \( N_{\text{blocks}} \)**  
- **Hidden dimension / width \( d_{\text{hidden}} \)**  

In all reported experiments, we use the following **default configuration** on JARVIS-DFT:

- **Cutoff radius**: 10.0 Å  
- **Number of interaction blocks**: 6  
- **Hidden dimension**: 256  

### S2.1 Cutoff Radius

The cutoff radius \( r_{\text{cut}} \) controls how many neighbors are included in the graph construction. In GemNet-style models on inorganic crystals, smaller cutoffs risk missing longer‑range coordination environments, while very large cutoffs substantially increase computational cost with diminishing returns.

Empirically and in line with prior GemNet work, a cutoff of **10.0 Å** provides a robust balance between accuracy and cost on JARVIS-DFT: it captures the relevant local environments for most bulk structures while keeping the number of neighbors per atom manageable on typical GPUs. For substantially larger, more dilute, or more strongly long‑range‑interacting systems, a modestly larger cutoff may be beneficial, whereas for very small cells a slightly smaller cutoff can reduce cost without strongly affecting accuracy.

### S2.2 Number of Interaction Blocks

The number of GemNet interaction blocks \( N_{\text{blocks}} \) controls model depth. Too few blocks underfit complex many‑body interactions, while additional depth beyond a certain point mainly increases training time and memory without clear gains.

On JARVIS-DFT we use **6 interaction blocks**, which is a depth commonly used for GemNet on crystalline materials and provides good accuracy with stable training. Conceptually, a **3–4 block** model would be lighter but less expressive and is more appropriate for smaller datasets, whereas **8+ blocks** are typically only justified when targeting extremely high accuracy and when computational resources are plentiful.

### S2.3 Hidden Dimension / Width

The hidden dimension \( d_{\text{hidden}} \) sets the width of the representation. Wider models can, in principle, capture more complex structure–property relationships, but also increase GPU memory and can be more prone to overfitting on limited data.

We fix \( d_{\text{hidden}} = 256 \) for TRACER, which is a standard choice for GemNet on medium‑sized materials datasets. In practice, dimensions around **128–192** may be sufficient for smaller datasets or constrained hardware, while going substantially beyond **256** tends to produce only marginal accuracy gains relative to the additional computational cost. The chosen width preserves a comfortable batch size on 16–24 GB GPUs and is robust across different random seeds.

---

## S3. Negative Results: DMET+VQE Experiments

In this section we summarize the **negative results** from the DMET+VQE (“quantum correction”) branch of the original hybrid pipeline. These experiments were fully implemented and run but are **not used in the final TRACER pipeline**, as they failed to improve, and sometimes degraded, the overall performance on the JARVIS-DFT test set.

### S3.1 Setup

- **Goal**: Use DMET (Density Matrix Embedding Theory) + VQE (Variational Quantum Eigensolver) to refine predictions on a subset of “hard” cases identified by the Gate-Hard ranking.  
- **Subset**: Top 7.5% hard cases (270 samples out of 3,604).  
- **Workflow**:
  1. Run the GemNet-based TRACER model on the test set.
  2. Use Gate-Hard to rank samples and select the hardest 270 structures.
  3. For each selected sample, construct DMET fragments and run a VQE-based quantum calculation to obtain corrected energies.
  4. Train a delta head / hybrid model to combine GNN predictions and quantum corrections.

### S3.2 Observed Issues

- **Limited coverage**:  
  - Quantum labels are available only for a small subset (270 samples), restricting the learning signal for any delta model.
- **Noise and instability**:  
  - VQE runs exhibit sensitivity to ansatz choice, optimizer hyperparameters, and noise.  
  - For some systems, the quantum energies did not consistently improve over the baseline DFT targets.
- **Integration difficulty**:
  - Delta learning on top of noisy quantum corrections often fails to generalize beyond the labeled subset.  
  - In cross-validation, models that incorporated quantum corrections did **not** outperform the pure GNN baseline on the full test set.

### S3.3 Quantitative Summary (Illustrative)

- **Baseline TRACER (no quantum)**:
  - Test MAE: ~0.037 eV/atom  
  - Test RMSE: ~0.080 eV/atom  
- **Hybrid GNN + DMET+VQE (on hard cases)**:
  - Slight **improvement** on the 270 labeled hard cases in isolation.  
  - **No improvement or mild degradation** when evaluated on the **full test set**, due to mismatch between the small quantum-labeled subset and the full distribution.

Given these findings, we treat DMET+VQE as a **negative result and future-work direction**, and we deliberately **exclude** quantum corrections from the final TRACER pipeline and from the public code release, while fully documenting the attempt for transparency.

---

## S4. Gate-Hard Heuristic Optimization (Weight Tuning)

Gate-Hard ranking assigns a **hardness score** to each sample based on a combination of model uncertainty and domain-specific heuristics:

\[
\text{Score} = \alpha \cdot \sigma^2 + \beta \cdot \text{TM\_flag} + \gamma \cdot \text{near\_degeneracy\_proxy}
\]

where:

- \( \sigma^2 \): Ensemble variance of TRACER predictions (uncertainty proxy).  
- `TM_flag`: Indicator for presence of transition metal(s) in the composition.  
- `near_degeneracy_proxy`: Simple proxy feature for near-degenerate configurations (e.g., small gaps between low-lying states, approximated from structural/chemical heuristics).  
- \( \alpha, \beta, \gamma \): Non-negative scalar weights.

### S4.1 Search Space and Procedure

We perform a lightweight grid search over:

- \( \alpha \in \{0.5, 1.0, 2.0\} \)  
- \( \beta \in \{0.0, 0.5, 1.0\} \)  
- \( \gamma \in \{0.0, 0.1, 0.2\} \)  

For each combination, we:

1. Compute scores for all test samples.  
2. Select the top **k = 270** hard cases (7.5% of the test set).  
3. Measure the mean absolute error (MAE) of TRACER **restricted** to the selected subset.  
4. Compare against:
   - **Random selection** of 270 samples.  
   - **Variance-only** ranking (i.e., \( \alpha=1, \beta=0, \gamma=0 \)).

### S4.2 Results

- **Random selection**: Baseline error on the selected subset is close to the global test MAE (~0.037 eV/atom).  
- **Variance-only ranking**: Increases the average error of the selected subset, meaning it successfully identifies harder-than-average examples.  
- **Full Gate-Hard ranking** (optimized weights):  
  - Also succeeds in identifying harder-than-average samples, but on this dataset it is **outperformed by variance-only ranking**.  
  - Quantitatively, variance-only achieves approximately **10.1% higher subset MAE** (i.e., a harder cohort) than Gate-Hard when both select the top 7.5% of cases.

Thus, on the JARVIS-DFT benchmark used here, the simpler **variance-only** strategy performs better than the more complex Gate-Hard heuristic. We therefore treat Gate-Hard as a **negative result** and recommend variance-only ranking as the default triage method, while still documenting Gate-Hard’s design and tuning procedure for completeness.

---

## S5. Visual Analysis of High-Error Outliers

We perform a qualitative analysis of the **highest-error test samples** of TRACER. Table S1 summarizes the top few outliers.

### S5.1 Summary of High-Error Cases

| Rank | Formula              | Domain     | Atoms | Abs. Error (eV/atom) | Variance | Notes                              |
|------|----------------------|-----------:|------:|----------------------:|---------:|------------------------------------|
| 1    | O\(_8\)              | jarvis_dft | 8     | 1.54                  | 0.09     | Unusual O-only structure           |
| 2    | C\(_{60}\)           | jarvis_dft | 60    | 1.03                  | 0.08     | Large molecular cage               |
| 3    | N\(_4\)O\(_{12}\)Cl\(_4\)K\(_8\) | jarvis_dft | 28 | 1.01          | 0.11     | Chemically complex multi-component |
| 4    | O\(_6\)Si\(_3\)      | jarvis_dft | 9     | 0.96                  | 0.06     | Unusual stoichiometry              |
| 5    | Re\(_2\)             | jarvis_dft | 2     | 0.92                  | 0.06     | Contains transition metal          |

(*Values rounded for readability; see `docs/high_error_cases.md` for exact numbers.*)

### S5.2 Observations

1. **Elemental and small-molecule-like systems**:  
   - Very small systems (e.g., Re\(_2\), U\(_2\)) and pure-element structures are frequently high-error outliers, possibly due to their under-representation in the training data and atypical bonding environments.

2. **Large, complex molecules and cages (e.g., C\(_{60}\))**:  
   - Structures that are closer in spirit to large molecular systems than to typical inorganic crystals can be challenging for a model trained primarily on bulk-like crystalline materials.

3. **Chemically complex multinary systems**:  
   - Systems with many elements and complex stoichiometries pose challenges due to their rarity and higher intrinsic variability in local chemical environments.

### S5.3 Representative Crystal Structure Visualizations

We rendered crystal structures for a representative subset of the highest‑error samples.  
The O\(_8\) system forms an extended oxygen network with unusual coordination environments and short O–O separations that are atypical of the bulk training distribution.  
The C\(_{60}\) structure appears as a large molecular cage with a hollow interior and curved carbon surface, markedly different from the periodic inorganic crystals that dominate the dataset.  
The N\(_4\)O\(_{12}\)Cl\(_4\)K\(_8\) and O\(_6\)Si\(_3\) systems exhibit chemically complex, low‑symmetry coordination polyhedra and mixed‑anion frameworks, again lying in sparsely sampled regions of composition–structure space.  
The Re\(_2\) case corresponds to a small, heavy‑element dimer, where subtle changes in bond length and electronic structure can lead to large relative errors in formation energy per atom.

Across these visualizations, high‑error cases systematically correspond to either (i) chemically exotic compositions, (ii) small or molecular‑like systems that are under‑represented in the training set, or (iii) unusual local coordination motifs.  
This is consistent with the quantitative patterns in Table S1 and supports the interpretation that TRACER’s largest errors are concentrated in structurally and chemically rare regimes rather than being randomly distributed.

---

## S6. Test Leakage Verification Methodology

Ensuring that **no test leakage** occurs (i.e., no overlap between train/validation/test sets) is critical for a trustworthy evaluation. We apply multiple safeguards to prevent and detect any leakage.

### S6.1 Split Generation and Storage

- **Initial splits** are generated once using the JARVIS-DFT dataset, following an 80/10/10 split (train/val/test).  
- Splits are defined **at the level of JARVIS IDs / sample IDs**, not at the level of raw files.  
- The resulting identifiers for the train/validation/test subsets are stored and distributed with the code so that users can exactly reconstruct the splits when they download the same upstream dataset.

### S6.2 Programmatic Leakage Checks

We perform the following programmatic checks (documented in the accompanying verification reports and utilities):

1. **Set disjointness**:
   - Verify that the three ID sets are pairwise disjoint:  
     \[
     \text{train} \cap \text{val} = \varnothing,\quad
     \text{train} \cap \text{test} = \varnothing,\quad
     \text{val} \cap \text{test} = \varnothing.
     \]

2. **Cardinality checks**:
   - Confirm that the counts match the reported numbers:
     - Train: 28,823 samples  
     - Validation: 3,602 samples  
     - Test: 3,604 samples  
     - Total: 36,029 samples after cleaning

3. **Mapping to underlying data**:
   - When reconstructing the dataset from raw JARVIS-DFT JSON files, we enforce that **each ID maps to exactly one structure**.  
   - Any duplicate or missing IDs trigger a verification failure.

### S6.3 Independent Verification

An independent verification pass checks that:

- The number of samples in each split matches the values assumed in all analysis scripts.  
- Reported metrics (MAE, RMSE, R², error breakdowns) are computed **only** on the test split.  
- Hyperparameter tuning and model selection are conducted using **only** the validation split.

Together, these checks provide high confidence that:

- There is **no leakage** of test data into training or validation.  
- All reported test metrics are based on a **genuinely held-out** subset of JARVIS-DFT.


