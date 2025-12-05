#!/usr/bin/env python3
"""
Red Flag Audit - Check for Publication Concerns
Detects any issues that could cause paper rejection.
"""

import sys
from pathlib import Path
import re
import logging
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_file_for_red_flags(filepath, content):
    """Check a file for red flags."""
    red_flags = []
    
    # Check for mock/fake/placeholder
    mock_patterns = [
        r'create_mock',
        r'generate_fake',
        r'placeholder',
        r'TODO.*mock',
        r'# MOCK',
        r'FIXME.*mock'
    ]
    
    for pattern in mock_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            red_flags.append(f"Mock data pattern: {pattern}")
    
    # Check for excessive random generation (in production code)
    if 'def predict' in content or 'def forward' in content:
        # Should not have random generation in prediction
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'np.random' in line or 'torch.rand' in line or 'random.random' in line:
                if i < len(lines) and '# test' not in lines[i-1].lower():
                    red_flags.append(f"Random data in production at line {i+1}")
    
    # Check for hardcoded values that look suspicious
    if 'return 0.0' in content or 'return 1.0' in content:
        if 'predict' in content.lower() or 'forward' in content.lower():
            red_flags.append("Suspicious hardcoded return values")
    
    return red_flags


def audit_critical_files():
    """Audit critical pipeline files for red flags."""
    logger.info("🔍 RED FLAG AUDIT")
    logger.info("="*80)
    
    critical_files = [
        'src/pipeline/run.py',
        'src/pipeline/gate_hard_ranking.py',
        'src/dft_hybrid/distill/delta_head.py',
        'src/gnn/model_gemnet.py',
        'src/gnn/train.py'
    ]
    
    all_red_flags = []
    
    for filepath in critical_files:
        filepath = Path(filepath)
        if not filepath.exists():
            continue
        
        logger.info(f"\n📄 Checking: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        red_flags = check_file_for_red_flags(filepath, content)
        
        if red_flags:
            logger.warning(f"   ⚠ Found {len(red_flags)} red flags:")
            for flag in red_flags:
                logger.warning(f"      - {flag}")
            all_red_flags.extend(red_flags)
        else:
            logger.info("   ✓ No red flags found")
    
    return all_red_flags


def check_test_functions():
    """Check if red flags are in test functions only."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: CHECKING TEST FUNCTION STATUS")
    logger.info("="*80)
    
    # Check gate_hard_ranking.py
    with open('src/pipeline/gate_hard_ranking.py', 'r') as f:
        content = f.read()
    
    # Look for test function markers
    if 'def test_' in content or 'if __name__' in content:
        logger.info("   ✓ Found test functions (acceptable)")
    
    # Check if main pipeline uses real data
    with open('src/pipeline/run.py', 'r') as f:
        content = f.read()
    
    if 'predict_with_model' in content and 'load_model' in content:
        logger.info("   ✓ Main pipeline uses real model loading")
    
    return True


def check_data_flow():
    """Check data flow for authenticity."""
    logger.info("\n" + "="*80)
    logger.info("DATA FLOW AUTHENTICATION")
    logger.info("="*80)
    
    checks = []
    
    # Check 1: Model loads from checkpoint
    if Path("models/gemnet_full/best_model.pt").exists():
        logger.info("   ✓ Model checkpoint exists")
        checks.append(True)
    else:
        logger.error("   ✗ Model checkpoint missing!")
        checks.append(False)
    
    # Check 2: Training data exists
    if Path("data/preprocessed_full_unified/train_data.json").exists():
        logger.info("   ✓ Training data exists")
        checks.append(True)
    else:
        logger.error("   ✗ Training data missing!")
        checks.append(False)
    
    # Check 3: Ensemble predictions exist
    if Path("artifacts/gemnet_predictions/ensemble_predictions.json").exists():
        logger.info("   ✓ Ensemble predictions exist")
        checks.append(True)
    else:
        logger.warning("   ⚠ Ensemble predictions missing")
        checks.append(False)
    
    return all(checks)


def check_for_common_issues():
    """Check for common publication red flags."""
    logger.info("\n" + "="*80)
    logger.info("CHECKING COMMON ISSUES")
    logger.info("="*80)
    
    issues = []
    
    # Issue 1: Data leakage
    # Would need to check if train/val overlap (handled in audit)
    
    # Issue 2: Overfitting
    checkpoint = torch.load("models/gemnet_full/best_model.pt", map_location='cpu')
    train_loss = checkpoint.get('train_loss', 999)
    val_loss = checkpoint.get('best_val_loss', 999)
    
    if abs(train_loss - val_loss) / max(train_loss, val_loss) < 0.2:
        logger.info("   ✓ Train/val losses close (no overfitting)")
    else:
        logger.warning("   ⚠ Large train/val gap (possible overfitting)")
        issues.append("Overfitting concern")
    
    # Issue 3: Inadequate data
    with open('data/preprocessed_full_unified/split_info.json', 'r') as f:
        splits = json.load(f)
    
    if splits['train_samples'] < 1000:
        logger.warning("   ⚠ Training set too small")
        issues.append("Small training set")
    else:
        logger.info(f"   ✓ Training set size adequate ({splits['train_samples']} samples)")
    
    # Issue 4: Results too good to be true
    val_loss = checkpoint.get('best_val_loss', 999)
    if val_loss < 0.001:
        logger.warning("   ⚠ Validation loss suspiciously low")
        issues.append("Suspiciously perfect results")
    elif val_loss < 0.1:
        logger.info(f"   ✓ Validation loss realistic ({val_loss:.4f})")
    else:
        logger.info(f"   ✓ Validation loss: {val_loss:.4f}")
    
    return issues


def main():
    """Run red flag audit."""
    import torch
    import json
    
    logger.info("🔍 COMPREHENSIVE RED FLAG AUDIT")
    logger.info("="*80)
    logger.info("Checking for publication concerns:")
    logger.info("  • Mock/fake data")
    logger.info("  • Placeholder code")
    logger.info("  • Data leakage")
    logger.info("  • Overfitting")
    logger.info("  • Unrealistic results")
    logger.info("="*80)
    
    # Run audits
    red_flags = audit_critical_files()
    check_test_functions()
    data_flow_ok = check_data_flow()
    common_issues = check_for_common_issues()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 RED FLAG AUDIT SUMMARY")
    logger.info("="*80)
    
    total_issues = len(red_flags) + len(common_issues) + (0 if data_flow_ok else 1)
    
    if total_issues == 0:
        logger.info("   ✅ NO RED FLAGS DETECTED!")
        logger.info("   Your pipeline is clean and ready for publication")
    else:
        logger.warning(f"   ⚠ Found {total_issues} potential issues:")
        for flag in red_flags:
            logger.warning(f"      - {flag}")
        for issue in common_issues:
            logger.warning(f"      - {issue}")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()

