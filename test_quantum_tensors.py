import numpy as np
import os
import sys

# Ensure UTF-8 output
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Define paths to the generated tensors
physical_path = r"C:\Users\Aarush\OneDrive\Desktop\MERGED_CSV\centralized\physical_branch_train_test.npz"
network_path = r"C:\Users\Aarush\OneDrive\Desktop\MERGED_CSV\centralized\network_branch_train_test.npz"

def verify_quantum_tensors(file_path, branch_name):
    if not os.path.exists(file_path):
        print(f"❌ {branch_name}: File not found!")
        return
    
    data = np.load(file_path)
    X_train, X_test = data['X_train'], data['X_test']
    
    print(f"--- Verifying {branch_name} ---")
    
    # Check 1: 12-Qubit Dimensionality
    if X_train.shape[1] == 12 and X_test.shape[1] == 12:
        print("✅ Dimensionality: Exactly 12 features (Ready for 12 qubits).")
    else:
        print(f"❌ Dimensionality Failed: Train has {X_train.shape[1]}, Test has {X_test.shape[1]} features.")

    # Check 2: Pauli Phase Angle Bounds [-π, π]
    train_min, train_max = np.min(X_train), np.max(X_train)
    test_min, test_max = np.min(X_test), np.max(X_test)
    
    # Using np.pi with a tiny tolerance for floating point rounding
    if train_min >= -np.pi - 1e-5 and train_max <= np.pi + 1e-5:
        print(f"✅ Train Bounds: [{train_min:.4f}, {train_max:.4f}] (Valid Pauli Rotations).")
    else:
        print(f"❌ Train Bounds Failed: [{train_min:.4f}, {train_max:.4f}].")
        
    if test_min >= -np.pi - 1e-5 and test_max <= np.pi + 1e-5:
        print(f"✅ Test Bounds: [{test_min:.4f}, {test_max:.4f}] (Valid Pauli Rotations).")
    else:
        print(f"❌ Test Bounds Failed: [{test_min:.4f}, {test_max:.4f}].")

verify_quantum_tensors(physical_path, "Physical Branch")
verify_quantum_tensors(network_path, "Network Branch")

