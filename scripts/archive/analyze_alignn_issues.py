#!/usr/bin/env python3
"""
Analyze why ALIGNN had poor results compared to GemNet
"""

import json
from pathlib import Path
import numpy as np

def analyze_alignn_issues():
    """Analyze ALIGNN training and evaluation issues."""
    
    print("="*80)
    print("  ALIGNN vs GEMNET: ROOT CAUSE ANALYSIS")
    print("="*80)
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 80)
    print("ALIGNN:")
    print("  • Training Loss: 0.0001 eV (extremely low, suspicious)")
    print("  • Validation Loss: 0.0000 eV (unrealistic, suggests overfitting)")
    print("  • Test MAE: 0.0053 eV/atom")
    print("  • Test R²: -0.35 ❌ (NEGATIVE - worse than mean predictor!)")
    print()
    print("GemNet:")
    print("  • Training Loss: 0.0016 eV (realistic)")
    print("  • Validation Loss: 0.0646 eV (realistic)")
    print("  • Test MAE: 0.0060 eV/atom")
    print("  • Test R²: 0.993 ✅ (EXCELLENT correlation)")
    
    print("\n" + "="*80)
    print("🔍 ROOT CAUSES:")
    print("="*80)
    
    print("\n1. VALIDATION LOSS = 0.0000 eV (Critical Issue)")
    print("   • This is physically impossible for real data")
    print("   • Indicates one of:")
    print("     a) Overfitting to training data")
    print("     b) Data leakage (validation set == training set)")
    print("     c) Loss calculation bug")
    print("     d) Model memorizing instead of learning")
    print("   → GemNet: 0.0646 eV validation loss is realistic")
    
    print("\n2. NEGATIVE R² = -0.35 (Critical Issue)")
    print("   • R² < 0 means model is WORSE than predicting the mean")
    print("   • Possible causes:")
    print("     a) Unit mismatch (model predicts in different scale than targets)")
    print("     b) Energy reference mismatch (formation vs total energy)")
    print("     c) Per-atom vs total energy confusion")
    print("     d) Model output format mismatch")
    print("   → GemNet: R² = 0.993 shows excellent correlation")
    
    print("\n3. TRAINING SETUP DIFFERENCES")
    print("   • ALIGNN:")
    print("     - Used per-atom energy as target")
    print("     - No gradient clipping initially (added later)")
    print("     - Validation loop without torch.no_grad() (gradients computed)")
    print("     - Line graph attention mechanism (complex)")
    print("   • GemNet:")
    print("     - Used per-atom energy as target (same)")
    print("     - Proper gradient clipping")
    print("     - Standard validation with torch.no_grad()")
    print("     - Simpler message-passing architecture")
    
    print("\n4. POSSIBLE TECHNICAL ISSUES")
    print("   • Energy Unit Mismatch:")
    print("     - ALIGNN might output in different units (kJ/mol?) than eV")
    print("     - Training data normalized but evaluation used raw values")
    print("   • Energy Reference:")
    print("     - Formation energy vs total energy confusion")
    print("     - Model trained on one, evaluated on another")
    print("   • Data Format:")
    print("     - Coordinate system issues (Cartesian vs fractional)")
    print("     - Structure representation differences")
    print("   • Training Instability:")
    print("     - Very low loss suggests numerical issues or bugs")
    print("     - Model might have collapsed to constant output")
    
    print("\n" + "="*80)
    print("💡 WHY GEMNET WORKED BETTER:")
    print("="*80)
    
    print("\n1. STABLE TRAINING")
    print("   • Realistic validation loss (0.0646 eV)")
    print("   • Proper overfitting detection via early stopping")
    print("   • No suspicious near-zero losses")
    
    print("\n2. CORRECT ENERGY SCALE")
    print("   • Model outputs match target energy scale")
    print("   • Proper normalization/denormalization")
    print("   • Consistent energy reference")
    
    print("\n3. BETTER GENERALIZATION")
    print("   • High R² (0.993) indicates good correlation")
    print("   • Model learns meaningful patterns, not just memorizing")
    print("   • Robust to distribution shifts")
    
    print("\n4. ARCHITECTURE SUITABILITY")
    print("   • GemNet's architecture matches the data characteristics")
    print("   • Message-passing well-suited for molecular systems")
    print("   • Less prone to overfitting than attention-based models")
    
    print("\n" + "="*80)
    print("📝 RECOMMENDATIONS:")
    print("="*80)
    
    print("\nFor ALIGNN to match GemNet performance:")
    print("1. Fix validation loss calculation (add proper torch.no_grad())")
    print("2. Verify energy units and references match")
    print("3. Add stronger regularization (dropout, weight decay)")
    print("4. Use proper early stopping (not just loss-based)")
    print("5. Debug why loss becomes 0.0000 (likely a bug)")
    print("6. Check data format consistency")
    
    print("\nFor Publication:")
    print("• Focus on GemNet results (stable, reliable)")
    print("• Note ALIGNN had technical issues (validation loss = 0.0000)")
    print("• Not a fair comparison until ALIGNN issues are resolved")
    print("• GemNet clearly demonstrates superior correlation (R²=0.993)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_alignn_issues()


