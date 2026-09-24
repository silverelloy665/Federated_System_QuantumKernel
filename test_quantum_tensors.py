"""
Quantum Tensor Verification Test.
Verifies 12-qubit feature dimensionality and Pauli rotation bounds [-pi, pi] on centralized datasets.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def verify_quantum_tensors(file_path: Path, branch_name: str) -> bool:
    if not file_path.exists():
        print(f"[FAIL] {branch_name}: File not found at {file_path}!")
        return False
    
    data = np.load(file_path)
    X_train, X_test = data['X_train'], data['X_test']
    
    print(f"\n--- Verifying {branch_name} ---")
    all_ok = True
    
    # Check 1: 12-Qubit Dimensionality
    if X_train.shape[1] == 12 and X_test.shape[1] == 12:
        print(f"  [PASS] Dimensionality: Exactly 12 features (Ready for 12 qubits). Shapes: Train={X_train.shape}, Test={X_test.shape}")
    else:
        print(f"  [FAIL] Dimensionality Failed: Train has {X_train.shape[1]}, Test has {X_test.shape[1]} features.")
        all_ok = False

    # Check 2: Pauli Phase Angle Bounds [-pi, pi]
    train_min, train_max = np.min(X_train), np.max(X_train)
    test_min, test_max = np.min(X_test), np.max(X_test)
    
    tol = 1e-4
    if train_min >= -np.pi - tol and train_max <= np.pi + tol:
        print(f"  [PASS] Train Bounds: [{train_min:.4f}, {train_max:.4f}] (Valid Pauli Rotations).")
    else:
        print(f"  [FAIL] Train Bounds Failed: [{train_min:.4f}, {train_max:.4f}].")
        all_ok = False
        
    if test_min >= -np.pi - tol and test_max <= np.pi + tol:
        print(f"  [PASS] Test Bounds: [{test_min:.4f}, {test_max:.4f}] (Valid Pauli Rotations).")
    else:
        print(f"  [FAIL] Test Bounds Failed: [{test_min:.4f}, {test_max:.4f}].")
        all_ok = False

    return all_ok

def run_tests():
    config = PipelineConfig()
    centralized_dir = config.centralized_dir
    
    phys_file = centralized_dir / "physical_branch_train_test.npz"
    net_file = centralized_dir / "network_branch_train_test.npz"

    if not phys_file.exists():
        phys_file = centralized_dir / "physical_branch.npz"
    if not net_file.exists():
        net_file = centralized_dir / "network_branch.npz"

    p_ok = verify_quantum_tensors(phys_file, "Physical Branch")
    n_ok = verify_quantum_tensors(net_file, "Network Branch")

    if p_ok and n_ok:
        print("\n[SUCCESS] All Quantum Tensor verification checks passed!")
    else:
        print("\n[ERROR] Some Quantum Tensor verification checks failed.")

if __name__ == "__main__":
    run_tests()
