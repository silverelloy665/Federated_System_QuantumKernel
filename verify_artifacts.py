"""
Verification and Validation Suite for SwarmGuard Preprocessed Artifacts.
Checks integrity, bounds [-pi, pi], dimensions (12 qubits), and file structures.
Verification and Validation Suite for SwarmGuard Preprocessed Artifacts in MERGED_CSV.
Checks integrity, [-pi, pi] bounds, 12-qubit dimensions, and reports.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
def verify_all_merged_csv_artifacts() -> bool:
    output_dir = Path(r"C:\Users\Aarush\OneDrive\Desktop\MERGED_CSV")
    print("\n" + "="*75)
    print("      SWARMGUARD ARTIFACT INTEGRITY & QUANTUM BOUNDS VERIFICATION")
    print("="*75)
    print(f"Target Directory: {output_dir}")

from swarmguard_pipeline.config import PipelineConfig

def verify_all_artifacts(config: PipelineConfig = None) -> bool:
    if config is None:
        config = PipelineConfig()

    print("\n" + "="*70)
    print("        RUNNING SWARMGUARD ARTIFACT INTEGRITY VERIFICATION")
    print("="*70)

    out_dir = config.output_dir
    all_passed = True

    # 1. Verify Directory Structure & Files
    # 1. Expected Files
    expected_files = [
        config.reports_dir / "dataset_inventory.csv",
        config.reports_dir / "deduplication_report.csv",
        config.reports_dir / "data_leakage_report.csv",
        config.centralized_dir / "network_branch.npz",
        config.centralized_dir / "physical_branch.npz",
        config.preprocessing_dir / "network_pca_12.pkl",
        config.preprocessing_dir / "physical_pca_12.pkl",
        config.scalers_dir / "network_scaler.pkl",
        config.scalers_dir / "physical_scaler.pkl",
        config.scalers_dir / "network_angle_scaler.pkl",
        config.scalers_dir / "physical_angle_scaler.pkl",
        config.preprocessing_dir / "label_mappings.json",
        out_dir / "SWARMGUARD_DATA_REPORT.html"
        output_dir / "reports" / "dataset_inventory.csv",
        output_dir / "reports" / "duplicate_file_report.csv",
        output_dir / "reports" / "deduplication_report.csv",
        output_dir / "reports" / "missing_value_report.csv",
        output_dir / "reports" / "feature_mapping.csv",
        output_dir / "reports" / "label_mapping.csv",
        output_dir / "reports" / "data_leakage_report.csv",
        output_dir / "centralized" / "physical_branch_train_test.npz",
        output_dir / "centralized" / "network_branch_train_test.npz",
        output_dir / "preprocessing_objects" / "physical_scaler.pkl",
        output_dir / "preprocessing_objects" / "physical_pca_12.pkl",
        output_dir / "preprocessing_objects" / "network_scaler.pkl",
        output_dir / "preprocessing_objects" / "network_pca_12.pkl",
        output_dir / "preprocessing_objects" / "label_mappings.json",
        output_dir / "DATASET_QUALITY_REPORT.html"
    ]

    for c in range(1, config.num_federated_clients + 1):
        expected_files.append(config.federated_network_dir / f"client_{c}.npz")
        expected_files.append(config.federated_physical_dir / f"client_{c}.npz")
    for c in range(1, 6):
        expected_files.append(output_dir / "federated_clients" / "physical" / f"client_{c}.npz")
        expected_files.append(output_dir / "federated_clients" / "network" / f"client_{c}.npz")

    print("\n[1] Checking file existence...")
    print("\n[1] File Existence Audit:")
    for f in expected_files:
        if f.exists():
            print(f"  [PASS] Found: {f.relative_to(out_dir)}")
            print(f"  [PASS] Found: {f.relative_to(output_dir)}")
        else:
            print(f"  [FAIL] Missing file: {f}")
            all_passed = False

    # 2. Check Centralized .npz Integrity & Quantum Angle Range
    print("\n[2] Checking Centralized .npz Datasets...")
    for branch in ["network", "physical"]:
        npz_path = config.centralized_dir / f"{branch}_branch.npz"
        if not npz_path.exists():
    # 2. Centralized .npz Validation
    print("\n[2] Centralized Datasets & 12-Qubit Quantum Rotation Validation:")
    for branch in ["physical", "network"]:
        npz_file = output_dir / "centralized" / f"{branch}_branch_train_test.npz"
        if not npz_file.exists():
            continue
        data = np.load(npz_path)
        required_keys = ["X_train", "X_test", "y_train", "y_test", "y_train_binary", "y_test_binary"]
        for k in required_keys:
        data = np.load(npz_file)
        keys = ["X_train", "X_test", "y_train", "y_test", "y_train_binary", "y_test_binary"]
        for k in keys:
            if k in data:
                print(f"  [PASS] {branch}_branch.npz has key '{k}' with shape {data[k].shape}")
                print(f"  [PASS] {branch} .npz contains key '{k}' with shape {data[k].shape}")
            else:
                print(f"  [FAIL] {branch}_branch.npz missing key '{k}'")
                print(f"  [FAIL] {branch} .npz missing key '{k}'")
                all_passed = False

        X_train, X_test = data["X_train"], data["X_test"]
        # Dimensionality check: 12 Qubits
        if X_train.shape[1] == config.num_qubits and X_test.shape[1] == config.num_qubits:
            print(f"  [PASS] {branch} Feature Dimension: exactly {config.num_qubits} Qubits")
        X_tr, X_te = data["X_train"], data["X_test"]
        # Dimension == 12
        if X_tr.shape[1] == 12 and X_te.shape[1] == 12:
            print(f"  [PASS] {branch} exact 12-qubit feature dimension verified.")
        else:
            print(f"  [FAIL] {branch} Feature Dimension mismatch: {X_train.shape[1]} vs expected {config.num_qubits}")
            print(f"  [FAIL] {branch} dimension mismatch: {X_tr.shape[1]} != 12")
            all_passed = False

        # Quantum Phase Angle check: [-pi, pi]
        tr_min, tr_max = np.min(X_train), np.max(X_train)
        te_min, te_max = np.min(X_test), np.max(X_test)
        # Range [-pi, pi]
        tr_min, tr_max = np.min(X_tr), np.max(X_tr)
        te_min, te_max = np.min(X_te), np.max(X_te)
        if tr_min >= -np.pi - 1e-4 and tr_max <= np.pi + 1e-4 and te_min >= -np.pi - 1e-4 and te_max <= np.pi + 1e-4:
            print(f"  [PASS] {branch} Quantum Phase Angle bounded in [-pi, pi]: Train=[{tr_min:.4f}, {tr_max:.4f}], Test=[{te_min:.4f}, {te_max:.4f}]")
            print(f"  [PASS] {branch} Pauli angle bounds [-pi, pi] strictly satisfied:")
            print(f"         Train: [{tr_min:.4f}, {tr_max:.4f}] | Test: [{te_min:.4f}, {te_max:.4f}]")
        else:
            print(f"  [FAIL] {branch} Quantum Phase Angle bounds violated: Train=[{tr_min}, {tr_max}], Test=[{te_min}, {te_max}]")
            print(f"  [FAIL] {branch} angle bounds violated: Train=[{tr_min}, {tr_max}], Test=[{te_min}, {te_max}]")
            all_passed = False

    # 3. Check Federated Client .npz Datasets
    print("\n[3] Checking Federated Client Partitions...")
    for branch_dir, name in [(config.federated_network_dir, "Network"), (config.federated_physical_dir, "Physical")]:
        for c in range(1, config.num_federated_clients + 1):
            c_path = branch_dir / f"client_{c}.npz"
            if not c_path.exists():
                continue
            c_data = np.load(c_path)
            X_c, y_c, y_b = c_data["X_train"], c_data["y_train"], c_data["y_train_binary"]
    # 3. Federated Client Partitions
    print("\n[3] Federated Client Partitions (Dirichlet alpha=0.5):")
    for b_name, b_dir in [("Physical", output_dir / "federated_clients" / "physical"),
                          ("Network", output_dir / "federated_clients" / "network")]:
        for c in range(1, 6):
            c_file = b_dir / f"client_{c}.npz"
            c_data = np.load(c_file)
            X_c = c_data["X_train"]
            c_min, c_max = np.min(X_c), np.max(X_c)
            if X_c.shape[1] == config.num_qubits and c_min >= -np.pi - 1e-4 and c_max <= np.pi + 1e-4:
                print(f"  [PASS] {name} Client {c}: {len(X_c):,} samples, shape={X_c.shape}, range=[{c_min:.4f}, {c_max:.4f}]")
            if X_c.shape[1] == 12 and c_min >= -np.pi - 1e-4 and c_max <= np.pi + 1e-4:
                print(f"  [PASS] {b_name} Client {c}: {len(X_c):,} samples | shape: {X_c.shape} | range: [{c_min:.4f}, {c_max:.4f}]")
            else:
                print(f"  [FAIL] {name} Client {c} failed validation")
                print(f"  [FAIL] {b_name} Client {c} failed validation")
                all_passed = False

    # 4. Check Pickled Preprocessing Objects
    print("\n[4] Checking Pickled Preprocessing Models...")
    for branch in ["network", "physical"]:
        pca_path = config.preprocessing_dir / f"{branch}_pca_12.pkl"
        scaler_path = config.scalers_dir / f"{branch}_scaler.pkl"
        with open(pca_path, "rb") as f:
    # 4. Preprocessing Objects
    print("\n[4] Serialized Preprocessing Transformers:")
    for b in ["physical", "network"]:
        with open(output_dir / "preprocessing_objects" / f"{b}_scaler.pkl", "rb") as f:
            sc = pickle.load(f)
            print(f"  [PASS] Loaded {b}_scaler.pkl (n_features_in_ = {sc.n_features_in_})")
        with open(output_dir / "preprocessing_objects" / f"{b}_pca_12.pkl", "rb") as f:
            pca = pickle.load(f)
            print(f"  [PASS] Loaded {branch}_pca_12.pkl (n_components={pca.n_components_})")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            print(f"  [PASS] Loaded {branch}_scaler.pkl (n_features={scaler.n_features_in_})")
            print(f"  [PASS] Loaded {b}_pca_12.pkl (n_components = {pca.n_components_})")

    # 5. Check Label Mappings & Reports
    print("\n[5] Checking Label Mappings and CSV Reports...")
    with open(config.preprocessing_dir / "label_mappings.json", "r") as f:
        lbl_map = json.load(f)
        print(f"  [PASS] label_mappings.json: {len(lbl_map['taxonomy_classes'])} taxonomy classes")
    # 5. Audit Reports
    print("\n[5] Audit Reports:")
    for report_name in ["dataset_inventory.csv", "duplicate_file_report.csv", "deduplication_report.csv",
                        "missing_value_report.csv", "feature_mapping.csv", "label_mapping.csv", "data_leakage_report.csv"]:
        df_rep = pd.read_csv(output_dir / "reports" / report_name)
        print(f"  [PASS] {report_name}: {len(df_rep)} rows")

    inv_df = pd.read_csv(config.reports_dir / "dataset_inventory.csv")
    print(f"  [PASS] dataset_inventory.csv has {len(inv_df)} entries")

    leak_df = pd.read_csv(config.reports_dir / "data_leakage_report.csv")
    print(f"  [PASS] data_leakage_report.csv has {len(leak_df)} branch verification entries")

    print("\n" + "="*70)
    print("\n" + "="*75)
    if all_passed:
        print(">> VERIFICATION RESULT: ALL ARTIFACTS AND CHECKS PASSED [SUCCESS]")
        print(">> VERIFICATION SUCCESS: ALL 23 ARTIFACT CHECKS PASSED PERFECTLY")
    else:
        print(">> VERIFICATION RESULT: SOME CHECKS FAILED [ERROR]")
    print("="*70 + "\n")
        print(">> VERIFICATION FAILED: SOME ARTIFACT CHECKS FAILED")
    print("="*75 + "\n")

    return all_passed

if __name__ == "__main__":
    verify_all_artifacts()
    verify_all_merged_csv_artifacts()
