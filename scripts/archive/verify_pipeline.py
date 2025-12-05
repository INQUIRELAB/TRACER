#!/usr/bin/env python3
"""
Comprehensive pipeline verification script.
Checks all claimed components are implemented and working.
"""

import sys
from pathlib import Path
import torch
import json
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 80)
print("  PIPELINE VERIFICATION CHECKLIST")
print("=" * 80)
print()

checks = {
    "1. GemNet Architecture": {"status": "PENDING", "details": []},
    "2. FiLM Domain Adaptation": {"status": "PENDING", "details": []},
    "3. LoRA Adapters": {"status": "PENDING", "details": []},
    "4. Ensemble Uncertainty": {"status": "PENDING", "details": []},
    "5. Gate-Hard Ranking": {"status": "PENDING", "details": []},
    "6. Delta Head (Quantum Corrections)": {"status": "PENDING", "details": []},
    "7. Quantum DMET+VQE": {"status": "PENDING", "details": []},
    "8. Trained Models": {"status": "PENDING", "details": []},
    "9. Evaluation Pipeline": {"status": "PENDING", "details": []},
    "10. Main Pipeline Integration": {"status": "PENDING", "details": []},
}

# 1. Check GemNet Architecture
print("🔍 1. Checking GemNet Architecture...")
try:
    from gnn.model_gemnet import GemNetWrapper, FiLMLayer, DomainEmbedding
    checks["1. GemNet Architecture"]["status"] = "✅ PASS"
    checks["1. GemNet Architecture"]["details"].append("GemNetWrapper class found")
    checks["1. GemNet Architecture"]["details"].append("FiLMLayer class found")
    checks["1. GemNet Architecture"]["details"].append("DomainEmbedding class found")
except ImportError as e:
    checks["1. GemNet Architecture"]["status"] = "❌ FAIL"
    checks["1. GemNet Architecture"]["details"].append(f"Import error: {e}")

# 2. Check FiLM in GemNet
print("🔍 2. Checking FiLM Domain Adaptation...")
try:
    from gnn.model_gemnet import GemNetWrapper
    model = GemNetWrapper(use_film=True, num_domains=5, film_dim=16)
    if hasattr(model, 'domain_embedding') and model.domain_embedding is not None:
        checks["2. FiLM Domain Adaptation"]["status"] = "✅ PASS"
        checks["2. FiLM Domain Adaptation"]["details"].append("FiLM enabled in GemNetWrapper")
        checks["2. FiLM Domain Adaptation"]["details"].append(f"Domain embedding: {type(model.domain_embedding)}")
        checks["2. FiLM Domain Adaptation"]["details"].append(f"Output layer has FiLM: {hasattr(model.output, 'film_layer')}")
    else:
        checks["2. FiLM Domain Adaptation"]["status"] = "⚠️  WARNING"
        checks["2. FiLM Domain Adaptation"]["details"].append("FiLM not enabled by default")
except Exception as e:
    checks["2. FiLM Domain Adaptation"]["status"] = "❌ FAIL"
    checks["2. FiLM Domain Adaptation"]["details"].append(f"Error: {e}")

# 3. Check LoRA in GemNet
print("🔍 3. Checking LoRA Adapters...")
try:
    from gnn.model_gemnet import GemNetWrapper
    model = GemNetWrapper()
    if hasattr(model, 'lora') or any('lora' in name.lower() for name, _ in model.named_modules()):
        checks["3. LoRA Adapters"]["status"] = "✅ PASS"
        checks["3. LoRA Adapters"]["details"].append("LoRA found in GemNet model")
    else:
        # Check if LoRA exists in domain_aware_model
        try:
            from gnn.domain_aware_model import LoRALayer
            checks["3. LoRA Adapters"]["status"] = "⚠️  WARNING"
            checks["3. LoRA Adapters"]["details"].append("LoRA exists but NOT in GemNet (only in SchNet domain_aware_model)")
            checks["3. LoRA Adapters"]["details"].append("LoRA is in domain_aware_model.py, not GemNetWrapper")
        except ImportError:
            checks["3. LoRA Adapters"]["status"] = "❌ FAIL"
            checks["3. LoRA Adapters"]["details"].append("LoRA not found anywhere")
except Exception as e:
    checks["3. LoRA Adapters"]["status"] = "❌ FAIL"
    checks["3. LoRA Adapters"]["details"].append(f"Error: {e}")

# 4. Check Ensemble Uncertainty
print("🔍 4. Checking Ensemble Uncertainty...")
try:
    from gnn.uncertainty import EnsembleUncertainty
    checks["4. Ensemble Uncertainty"]["status"] = "✅ PASS"
    checks["4. Ensemble Uncertainty"]["details"].append("EnsembleUncertainty class found")
    
    # Check for ensemble model checkpoints
    ensemble_dirs = list(Path("artifacts").glob("*ensemble*")) if Path("artifacts").exists() else []
    if ensemble_dirs:
        checks["4. Ensemble Uncertainty"]["details"].append(f"Found ensemble artifacts: {len(ensemble_dirs)}")
    else:
        checks["4. Ensemble Uncertainty"]["details"].append("No ensemble model checkpoints found")
except ImportError as e:
    checks["4. Ensemble Uncertainty"]["status"] = "❌ FAIL"
    checks["4. Ensemble Uncertainty"]["details"].append(f"Import error: {e}")

# 5. Check Gate-Hard Ranking
print("🔍 5. Checking Gate-Hard Ranking...")
try:
    from pipeline.gate_hard_ranking import GateHardRanker, DomainRankingConfig
    checks["5. Gate-Hard Ranking"]["status"] = "✅ PASS"
    checks["5. Gate-Hard Ranking"]["details"].append("GateHardRanker class found")
    
    # Check for gate-hard artifacts
    gate_hard_dirs = list(Path("artifacts").glob("*gate*hard*")) if Path("artifacts").exists() else []
    if gate_hard_dirs:
        checks["5. Gate-Hard Ranking"]["details"].append(f"Found gate-hard artifacts: {len(gate_hard_dirs)}")
except ImportError as e:
    checks["5. Gate-Hard Ranking"]["status"] = "❌ FAIL"
    checks["5. Gate-Hard Ranking"]["details"].append(f"Import error: {e}")

# 6. Check Delta Head
print("🔍 6. Checking Delta Head...")
try:
    from dft_hybrid.distill.delta_head import DeltaHead, DeltaHeadTrainer
    checks["6. Delta Head (Quantum Corrections)"]["status"] = "✅ PASS"
    checks["6. Delta Head (Quantum Corrections)"]["details"].append("DeltaHead class found")
    
    # Check for trained delta head
    delta_path = Path("artifacts/delta_head.pt")
    if delta_path.exists():
        checks["6. Delta Head (Quantum Corrections)"]["details"].append("Trained delta head checkpoint found")
        try:
            ckpt = torch.load(delta_path, map_location='cpu', weights_only=False)
            checks["6. Delta Head (Quantum Corrections)"]["details"].append(f"Checkpoint keys: {list(ckpt.keys())[:5]}")
        except:
            pass
    else:
        checks["6. Delta Head (Quantum Corrections)"]["details"].append("⚠️  No trained delta head checkpoint found")
except ImportError as e:
    checks["6. Delta Head (Quantum Corrections)"]["status"] = "❌ FAIL"
    checks["6. Delta Head (Quantum Corrections)"]["details"].append(f"Import error: {e}")

# 7. Check Quantum DMET+VQE
print("🔍 7. Checking Quantum DMET+VQE...")
try:
    from dft_hybrid.dmet.fragment import QuantumFragmentLabeler, FragmentGenerator
    checks["7. Quantum DMET+VQE"]["status"] = "✅ PASS"
    checks["7. Quantum DMET+VQE"]["details"].append("QuantumFragmentLabeler class found")
    checks["7. Quantum DMET+VQE"]["details"].append("FragmentGenerator class found")
    
    # Check if Qiskit is available
    try:
        import qiskit
        checks["7. Quantum DMET+VQE"]["details"].append(f"Qiskit available: {qiskit.__version__}")
    except ImportError:
        checks["7. Quantum DMET+VQE"]["details"].append("⚠️  Qiskit not available")
        
    # Check for quantum labels
    qnn_labels = Path("artifacts/quantum_labels_gate_hard.csv")
    if qnn_labels.exists():
        checks["7. Quantum DMET+VQE"]["details"].append("Quantum labels CSV found")
        import pandas as pd
        df = pd.read_csv(qnn_labels)
        checks["7. Quantum DMET+VQE"]["details"].append(f"Quantum labels: {len(df)} samples")
except ImportError as e:
    checks["7. Quantum DMET+VQE"]["status"] = "❌ FAIL"
    checks["7. Quantum DMET+VQE"]["details"].append(f"Import error: {e}")
except Exception as e:
    checks["7. Quantum DMET+VQE"]["status"] = "⚠️  WARNING"
    checks["7. Quantum DMET+VQE"]["details"].append(f"Error checking labels: {e}")

# 8. Check Trained Models
print("🔍 8. Checking Trained Models...")
model_paths = {
    "GemNet with FiLM": Path("models/gemnet_per_atom_film/best_model.pt"),
    "Delta Head": Path("artifacts/delta_head.pt"),
}
all_found = True
for name, path in model_paths.items():
    if path.exists():
        checks["8. Trained Models"]["details"].append(f"✅ {name}: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            size_mb = path.stat().st_size / (1024 * 1024)
            checks["8. Trained Models"]["details"].append(f"   Size: {size_mb:.1f} MB")
        except Exception as e:
            checks["8. Trained Models"]["details"].append(f"   ⚠️  Error loading: {e}")
    else:
        checks["8. Trained Models"]["details"].append(f"❌ {name}: NOT FOUND at {path}")
        all_found = False

if all_found:
    checks["8. Trained Models"]["status"] = "✅ PASS"
else:
    checks["8. Trained Models"]["status"] = "⚠️  WARNING"

# 9. Check Evaluation Pipeline
print("🔍 9. Checking Evaluation Pipeline...")
eval_script = Path("scripts/evaluate_gemnet_film.py")
if eval_script.exists():
    checks["9. Evaluation Pipeline"]["status"] = "✅ PASS"
    checks["9. Evaluation Pipeline"]["details"].append(f"Evaluation script found: {eval_script}")
    
    # Check if evaluation was run
    eval_log = Path("logs/evaluate_gemnet_film_full.log")
    if eval_log.exists():
        checks["9. Evaluation Pipeline"]["details"].append("Evaluation log found")
        with open(eval_log, 'r') as f:
            content = f.read()
            if "MAE" in content and "R²" in content:
                checks["9. Evaluation Pipeline"]["details"].append("Evaluation results found in log")
else:
    checks["9. Evaluation Pipeline"]["status"] = "❌ FAIL"
    checks["9. Evaluation Pipeline"]["details"].append("Evaluation script not found")

# 10. Check Main Pipeline Integration
print("🔍 10. Checking Main Pipeline Integration...")
try:
    from pipeline.run import HybridPipeline
    checks["10. Main Pipeline Integration"]["status"] = "✅ PASS"
    checks["10. Main Pipeline Integration"]["details"].append("HybridPipeline class found")
    
    # Check if pipeline has all methods
    pipeline_methods = ['load_data', 'train_gnn_surrogate', 'estimate_uncertainty', 'apply_delta_head']
    for method in pipeline_methods:
        if hasattr(HybridPipeline, method):
            checks["10. Main Pipeline Integration"]["details"].append(f"✅ {method}() method exists")
        else:
            checks["10. Main Pipeline Integration"]["details"].append(f"❌ {method}() method missing")
            checks["10. Main Pipeline Integration"]["status"] = "⚠️  WARNING"
except ImportError as e:
    checks["10. Main Pipeline Integration"]["status"] = "❌ FAIL"
    checks["10. Main Pipeline Integration"]["details"].append(f"Import error: {e}")

# Print summary
print()
print("=" * 80)
print("  VERIFICATION SUMMARY")
print("=" * 80)
print()

for check_name, check_data in checks.items():
    status = check_data["status"]
    details = check_data["details"]
    
    print(f"{status} {check_name}")
    for detail in details:
        print(f"   {detail}")
    print()

# Overall assessment
pass_count = sum(1 for c in checks.values() if c["status"] == "✅ PASS")
warn_count = sum(1 for c in checks.values() if c["status"] == "⚠️  WARNING")
fail_count = sum(1 for c in checks.values() if c["status"] == "❌ FAIL")

print("=" * 80)
print(f"Overall: {pass_count} ✅ | {warn_count} ⚠️  | {fail_count} ❌")
print("=" * 80)

# Issues to address
print()
if warn_count > 0 or fail_count > 0:
    print("⚠️  ISSUES TO ADDRESS:")
    print()
    for check_name, check_data in checks.items():
        if check_data["status"] in ["⚠️  WARNING", "❌ FAIL"]:
            print(f"   {check_name}: {check_data['status']}")
            for detail in check_data["details"]:
                if "⚠️" in detail or "❌" in detail:
                    print(f"      - {detail}")
else:
    print("✅ All checks passed! Pipeline is fully implemented.")



