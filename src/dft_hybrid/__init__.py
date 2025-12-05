"""
Hybrid DFT→GNN→QNN pipeline for quantum chemistry.

This package implements a hybrid approach combining:
- DFT calculations for reference data
- Graph Neural Networks (GNN) as surrogate models
- Quantum Neural Networks (QNN) for quantum corrections
- Ensemble uncertainty estimation for gating hard cases
- DMET + VQE for fragment calculations
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Main package imports - avoid circular imports by using lazy imports
__all__ = [
    "graphs",
    "gnn", 
    "dmet",
    "quantum",
    "distill",
    "data"
]
