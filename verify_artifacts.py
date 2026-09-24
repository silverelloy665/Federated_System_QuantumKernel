"""
Verification and Validation Suite for SwarmGuard Preprocessed Artifacts.
Checks integrity, bounds [-pi, pi], dimensions (12 qubits), and generated reports.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig

def verify_all_artifacts(config: PipelineConfig = None) -> bool:
    if config is None:
        config = PipelineConfig()

    print("\n" + "="*75)
    print("      SWARMGUARD ARTIFACT INTEGRITY & QUANTUM BOUNDS VERIFICATION")
    print("="*75)
    print(f"Target Directory: {config.output_dir}")

    out_dir = config.output_dir
    all_passed = True

    # 1. Expected Files
    expected_files = [
        config.reports_dir / "dataset_inventory.csv",
        config.reports_dir / "deduplication_report.csv",
        config.reports_dir / "data_leakage_report.csv",
        config.reports_dir / "feature_mapping.csv",
        config.reports_dir / "label_mapping.csv",
        config.reports_dir / "skewness_report.csv",
        config.centralized_dir / "network_branch_train_test.npz",
        config.centralized_dir / "physical_branch_train_test.npz",
        config.preprocessing_dir / "network_pca_12.pkl",
        config.preprocessing_dir / "physical_pca_12.pkl",
        config.preprocessing_dir / "label_mappings.json",
        out_dir / "SWARMGUARD_DATA_REPORT.html"
    ]

    for c in range(1, config.num_federated_clients + 1):
        expected_files.append(config.federated_network_dir / f"client_{c}.npz")
        expected_files.append(config.federated_physical_dir / f"client_{c}.npz")

    print("\n[1] File Existence Audit:")
    for f in expected_files:
        if f.exists():
            try:
                rel = f.relative_to(out_dir)
            except ValueError:
                rel = f.name
            print(f"  [PASS] Found: {rel}")
        else:
            print(f"  [FAIL] Missing file: {f}")
            all_passed = False

    # 2. Centralized .npz Validation
    print("\n[2] Centralized Datasets & 12-Qubit Quantum Rotation Validation:")
    for branch in ["physical", "network"]:
        npz_file = config.centralized_dir / f"{branch}_branch_train_test.npz"
        if not npz_file.exists():
            npz_file = config.centralized_dir / f"{branch}_branch.npz"
        if not npz_file.exists():
            print(f"  [FAIL] Centralized {branch} .npz not found")
            all_passed = False
            continue

        data = np.load(npz_file)
        keys = ["X_train", "X_test", "y_train", "y_test", "y_train_binary", "y_test_binary"]
        for k in keys:
            if k in data:
                print(f"  [PASS] {branch} .npz contains key '{k}' with shape {data[k].shape}")
            else:
                print(f"  [FAIL] {branch} .npz missing key '{k}'")
                all_passed = False

        X_tr, X_te = data["X_train"], data["X_test"]
        # Dimension == 12
        if X_tr.shape[1] == config.num_qubits and X_te.shape[1] == config.num_qubits:
            print(f"  [PASS] {branch} exact {config.num_qubits}-qubit feature dimension verified.")
        else:
            print(f"  [FAIL] {branch} dimension mismatch: {X_tr.shape[1]} != {config.num_qubits}")
            all_passed = False

        # Range [-pi, pi]
        tr_min, tr_max = float(np.min(X_tr)), float(np.max(X_tr))
        te_min, te_max = float(np.min(X_te)), float(np.max(X_te))
        tol = 1e-4
        if tr_min >= -np.pi - tol and tr_max <= np.pi + tol and te_min >= -np.pi - tol and te_max <= np.pi + tol:
            print(f"  [PASS] {branch} Pauli angle bounds [-pi, pi] strictly satisfied:")
            print(f"         Train: [{tr_min:.4f}, {tr_max:.4f}] | Test: [{te_min:.4f}, {te_max:.4f}]")
        else:
            print(f"  [FAIL] {branch} angle bounds violated: Train=[{tr_min:.4f}, {tr_max:.4f}], Test=[{te_min:.4f}, {te_max:.4f}]")
            all_passed = False

    # 3. Federated Client Partitions
    print(f"\n[3] Federated Client Partitions (Dirichlet alpha={config.dirichlet_alpha}):")
    for b_name, b_dir in [("Physical", config.federated_physical_dir), ("Network", config.federated_network_dir)]:
        for c in range(1, config.num_federated_clients + 1):
            c_file = b_dir / f"client_{c}.npz"
            if not c_file.exists():
                print(f"  [FAIL] Missing client file: {c_file}")
                all_passed = False
                continue
            c_data = np.load(c_file)
            X_c = c_data["X_train"]
            c_min, c_max = float(np.min(X_c)), float(np.max(X_c))
            if X_c.shape[1] == config.num_qubits and c_min >= -np.pi - 1e-4 and c_max <= np.pi + 1e-4:
                print(f"  [PASS] {b_name} Client {c}: {len(X_c):,} samples | shape: {X_c.shape} | range: [{c_min:.4f}, {c_max:.4f}]")
            else:
                print(f"  [FAIL] {b_name} Client {c} failed validation")
                all_passed = False

    # 4. Preprocessing Objects
    print("\n[4] Serialized Preprocessing Transformers:")
    for b in ["physical", "network"]:
        scaler_p = config.preprocessing_dir / f"{b}_scaler.pkl"
        if not scaler_p.exists():
            scaler_p = config.scalers_dir / f"{b}_scaler.pkl"
        if scaler_p.exists():
            with open(scaler_p, "rb") as f:
                sc = pickle.load(f)
                print(f"  [PASS] Loaded {b}_scaler.pkl (n_features_in_ = {sc.n_features_in_})")
        else:
            print(f"  [FAIL] Missing {b}_scaler.pkl")
            all_passed = False

        pca_p = config.preprocessing_dir / f"{b}_pca_12.pkl"
        if pca_p.exists():
            with open(pca_p, "rb") as f:
                pca = pickle.load(f)
                print(f"  [PASS] Loaded {b}_pca_12.pkl (n_components = {pca.n_components_})")
        else:
            print(f"  [FAIL] Missing {b}_pca_12.pkl")
            all_passed = False

    # 5. Audit Reports
    print("\n[5] Audit Reports:")
    for report_name in ["dataset_inventory.csv", "deduplication_report.csv", "feature_mapping.csv", "label_mapping.csv", "data_leakage_report.csv"]:
        rp = config.reports_dir / report_name
        if rp.exists():
            df_rep = pd.read_csv(rp)
            print(f"  [PASS] {report_name}: {len(df_rep)} rows")
        else:
            print(f"  [FAIL] Missing report: {report_name}")
            all_passed = False

    print("\n" + "="*75)
    if all_passed:
        print(">> VERIFICATION SUCCESS: ALL ARTIFACT CHECKS PASSED PERFECTLY")
    else:
        print(">> VERIFICATION FAILED: SOME ARTIFACT CHECKS FAILED")
    print("="*75 + "\n")

    return all_passed

if __name__ == "__main__":
    success = verify_all_artifacts()
    sys.exit(0 if success else 1)
