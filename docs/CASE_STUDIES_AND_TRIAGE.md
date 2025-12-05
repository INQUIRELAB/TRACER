# Case Studies and Variance-Based Triage Analysis

This document summarizes high-error case studies from the JARVIS-DFT test set and evaluates the effectiveness of variance-based triage compared to random escalation.

## High-Error Case Studies

We inspect the top high-error predictions from the ensemble to contextualize failure modes. Variance (ensemble predictive variance) serves as our uncertainty proxy.

| Rank | Sample ID | Formula | Domain | Atoms | Abs. Error (eV/atom) | Variance | Std Dev | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | unknown | O8 | jarvis_dft | 8 | 1.5449 | 0.0870 | 0.2950 | - |
| 2 | unknown | C60 | jarvis_dft | 60 | 1.0312 | 0.0848 | 0.2912 | - |
| 3 | unknown | N4O12Cl4K8 | jarvis_dft | 28 | 1.0142 | 0.1133 | 0.3367 | - |
| 4 | unknown | O6Si3 | jarvis_dft | 9 | 0.9644 | 0.0629 | 0.2508 | - |
| 5 | unknown | Re2 | jarvis_dft | 2 | 0.9193 | 0.0603 | 0.2457 | Contains transition metal |
| 6 | unknown | U2 | jarvis_dft | 2 | 0.7996 | 0.0613 | 0.2475 | - |


## Variance-Based Escalation vs Random Sampling

We compare a variance-prioritized selection strategy against random sampling for different escalation budgets and error thresholds. Yield is defined as the fraction of selected samples whose absolute error exceeds the specified threshold (i.e., truly challenging cases). Relative improvement is computed against the mean random yield.

Random trials: 2000

| Budget | Error Threshold (eV/atom) | Variance Yield | Random Yield (mean ± std) | Relative Improvement |
| --- | --- | --- | --- | --- |
| 50 | 0.10 | 0.340 | 0.076 ± 0.036 | +346.9% |
| 50 | 0.15 | 0.260 | 0.042 ± 0.028 | +516.8% |
| 50 | 0.20 | 0.080 | 0.025 ± 0.022 | +222.1% |
| 100 | 0.10 | 0.250 | 0.078 ± 0.026 | +220.3% |
| 100 | 0.15 | 0.170 | 0.044 ± 0.020 | +287.0% |
| 100 | 0.20 | 0.060 | 0.026 ± 0.016 | +130.1% |
| 150 | 0.10 | 0.220 | 0.078 ± 0.021 | +182.8% |
| 150 | 0.15 | 0.133 | 0.044 ± 0.016 | +201.7% |
| 150 | 0.20 | 0.053 | 0.026 ± 0.013 | +101.4% |
| 200 | 0.10 | 0.200 | 0.078 ± 0.019 | +157.4% |
| 200 | 0.15 | 0.110 | 0.043 ± 0.014 | +153.6% |
| 200 | 0.20 | 0.040 | 0.026 ± 0.011 | +52.2% |
| 220 | 0.10 | 0.200 | 0.077 ± 0.018 | +158.3% |
| 220 | 0.15 | 0.105 | 0.044 ± 0.014 | +139.4% |
| 220 | 0.20 | 0.036 | 0.026 ± 0.011 | +40.1% |
| 250 | 0.10 | 0.200 | 0.078 ± 0.016 | +155.7% |
| 250 | 0.15 | 0.104 | 0.044 ± 0.013 | +138.4% |
| 250 | 0.20 | 0.036 | 0.026 ± 0.010 | +37.2% |
| 280 | 0.10 | 0.193 | 0.078 ± 0.016 | +147.5% |
| 280 | 0.15 | 0.100 | 0.044 ± 0.012 | +126.7% |
| 280 | 0.20 | 0.036 | 0.026 ± 0.009 | +35.3% |
| 300 | 0.10 | 0.200 | 0.078 ± 0.015 | +157.3% |
| 300 | 0.15 | 0.100 | 0.043 ± 0.011 | +130.6% |
| 300 | 0.20 | 0.037 | 0.026 ± 0.009 | +41.5% |


Notes:
- Variance-based selection consistently outperforms random sampling across tested budgets and thresholds.
- Random yields are averaged over 2000 trials to provide a stable baseline.
