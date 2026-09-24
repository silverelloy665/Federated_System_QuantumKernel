"""
Phase D: 12-Qubit QML Formatting & Quantum Phase Angle Scaling.
Applies zero-leakage transforms:
1. Log1p Transform on Skewed Network Features (with before/after skewness audit)
2. Imputer (fitted strictly on X_train)
3. StandardScaler (fitted strictly on X_train)
4. PCA-12 (fitted strictly on X_train, with dimensionality reduction audit & warnings)
5. Quantum Phase Angle Scaler [-pi, pi] (fitted strictly on X_train)

Exports serialized preprocessing artifacts, centralized and federated .npz datasets,
and verifies zero-leakage and Pauli angle constraints.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats
from .config import PipelineConfig

# Known skewed network flow continuous features
SKEWED_NETWORK_FEATURES = [
    'flow_duration',
    'tx_packets',
    'rx_packets',
    'tx_bytes',
    'rx_bytes',
    'packet_rate',
    'byte_rate',
    'mean_packet_size',
    'mean_delay',
    'mean_jitter',
    'lost_packets'
]

class QuantumPhaseAngleScaler:
    """
    Scales PCA components into the Pauli rotation angle range [-pi, pi].
    Fitted strictly on training data with safe clipping for inference / test data.
    """
    def __init__(self, angle_min: float = -np.pi, angle_max: float = np.pi):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.min_vals = None
        self.max_vals = None
        self.diff = None

    def fit(self, X: np.ndarray):
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)
        diff = self.max_vals - self.min_vals
        diff[diff == 0.0] = 1.0
        self.diff = diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        norm = (X - self.min_vals) / self.diff
        scaled = norm * (self.angle_max - self.angle_min) + self.angle_min
        return np.clip(scaled, self.angle_min, self.angle_max)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class QuantumFormatter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.leakage_records: List[Dict[str, Any]] = []
        self.skewness_records: List[Dict[str, Any]] = []

    def get_feature_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Separates feature matrix from multi-class and binary labels."""
        feature_cols = [c for c in df.columns if c not in ['canonical_label', 'class_id', 'binary_label', '_source_dataset']]
        X = df[feature_cols].values.astype(np.float64)
        y_multi = df['class_id'].values.astype(np.int64)
        y_binary = df['binary_label'].values.astype(np.int64)
        return X, y_multi, y_binary, feature_cols

    def apply_log1p_transforms(
        self,
        branch: str,
        f_cols: List[str],
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        client_dfs_raw: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Applies fit-on-train log1p transformation to heavily skewed features (Issue #7).
        Computes and records before/after skewness metrics.
        """
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()
        client_X = [c.copy() for c in client_dfs_raw]

        if branch == "Network":
            print("    [*] Applying log1p transformation on heavily skewed network features...", flush=True)
            for idx, col_name in enumerate(f_cols):
                is_skewed = col_name in SKEWED_NETWORK_FEATURES
                col_train = X_train[:, idx]
                
                # Compute before skewness (ignoring NaNs)
                valid_train = col_train[~np.isnan(col_train)]
                skew_before = float(scipy.stats.skew(valid_train)) if len(valid_train) > 2 else 0.0

                if is_skewed:
                    # Apply log1p strictly (np.log1p(max(0, x)))
                    X_train[:, idx] = np.log1p(np.maximum(X_train[:, idx], 0.0))
                    X_test[:, idx] = np.log1p(np.maximum(X_test[:, idx], 0.0))
                    for c_mat in client_X:
                        c_mat[:, idx] = np.log1p(np.maximum(c_mat[:, idx], 0.0))

                    valid_after = X_train[~np.isnan(X_train[:, idx]), idx]
                    skew_after = float(scipy.stats.skew(valid_after)) if len(valid_after) > 2 else 0.0
                else:
                    skew_after = skew_before

                self.skewness_records.append({
                    "branch": branch,
                    "feature_name": col_name,
                    "skewness_before": round(skew_before, 4),
                    "skewness_after": round(skew_after, 4),
                    "log1p_applied": is_skewed
                })

        return X_train, X_test, client_X

    def run_quantum_formatting(self, split_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes Phase D: Zero-leakage log1p -> imputer -> scaler -> PCA-12 -> quantum angle scaling.
        """
        print("\n========================================================", flush=True)
        print("  PHASE D: ZERO-LEAKAGE 12-QUBIT QML PREPROCESSING", flush=True)
        print("========================================================", flush=True)

        processed_outputs = {}

        # Save label mappings
        label_mappings = {
            "taxonomy_classes": {i: name for i, name in enumerate(self.config.taxonomy_classes)},
            "binary_classes": {0: "BENIGN", 1: "ATTACK"},
            "num_qubits": self.config.num_qubits,
            "quantum_angle_range": [float(self.config.quantum_angle_min), float(self.config.quantum_angle_max)]
        }
        with open(self.config.preprocessing_dir / "label_mappings.json", "w") as f:
            json.dump(label_mappings, f, indent=4)
        print(f"[+] Saved label mappings to: {self.config.preprocessing_dir / 'label_mappings.json'}", flush=True)

        for branch, data in split_results.items():
            print(f"\n[*] Formatting {branch} Branch for 12-Qubit Quantum Kernel...", flush=True)
            train_df = data["train_df"]
            test_df = data["test_df"]
            client_dfs = data["client_dfs"]

            # Extract raw continuous feature matrices
            X_train_raw, y_train, y_train_binary, f_cols = self.get_feature_and_labels(train_df)
            X_test_raw, y_test, y_test_binary, _ = self.get_feature_and_labels(test_df)
            client_raw = [self.get_feature_and_labels(c_df)[0] for c_df in client_dfs]
            client_labels = [(self.get_feature_and_labels(c_df)[1], self.get_feature_and_labels(c_df)[2]) for c_df in client_dfs]

            n_raw_features = X_train_raw.shape[1]
            print(f"    Raw features count: {n_raw_features} ({', '.join(f_cols[:6])}...)", flush=True)

            # 1. Fit-on-train Log1p Transform on Skewed Features (Issue #7)
            X_train_t, X_test_t, client_raw_t = self.apply_log1p_transforms(
                branch, f_cols, X_train_raw, X_test_raw, client_raw
            )

            # 2. Zero-Leakage Imputation (Fitted strictly on X_train)
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train_t)
            X_test_imp = imputer.transform(X_test_t)

            # 3. Fit StandardScaler strictly on Training Data
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train_imp)
            X_test_std = scaler.transform(X_test_imp)

            # 4. Fit PCA-12 strictly on Training Data (Issue #2 Dimensionality Audit)
            n_components = min(self.config.num_qubits, n_raw_features)
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            X_train_pca = pca.fit_transform(X_train_std)
            X_test_pca = pca.transform(X_test_std)
            exp_var = float(np.sum(pca.explained_variance_ratio_)) * 100

            # Issue #2 Warning Check
            is_reduced = bool(n_components < n_raw_features)
            if not is_reduced:
                print(f"    [WARNING] PCA n_components ({n_components}) >= raw features ({n_raw_features}) for {branch} branch. PCA is acting as an orthogonal rotation without dimensionality reduction.", flush=True)
                reduction_status = "ROTATION_ONLY (NO_REDUCTION)"
            else:
                print(f"    [+] PCA-12 Dimensionality Reduction: {n_raw_features} raw features -> {n_components} components ({exp_var:.2f}% explained variance).", flush=True)
                reduction_status = f"REDUCED ({n_raw_features} -> {n_components})"

            # 5. Fit Quantum Phase Angle Scaler strictly on Training PCA Features
            angle_scaler = QuantumPhaseAngleScaler(
                angle_min=self.config.quantum_angle_min,
                angle_max=self.config.quantum_angle_max
            )
            X_train_qml = angle_scaler.fit_transform(X_train_pca)
            X_test_qml = angle_scaler.transform(X_test_pca)

            # Verify Pauli Rotation Range
            train_min, train_max = np.min(X_train_qml), np.max(X_train_qml)
            test_min, test_max = np.min(X_test_qml), np.max(X_test_qml)
            print(f"    Quantum Angle Check:", flush=True)
            print(f"      Train Bounds: [{train_min:.4f}, {train_max:.4f}] (Pauli Range [-3.1416, 3.1416])", flush=True)
            print(f"      Test  Bounds: [{test_min:.4f}, {test_max:.4f}]", flush=True)

            # 6. Save Preprocessing Objects (.pkl) in both preprocessing_dir and scalers_dir
            branch_lower = branch.lower()
            pca_file = self.config.preprocessing_dir / f"{branch_lower}_pca_12.pkl"
            imputer_file = self.config.scalers_dir / f"{branch_lower}_imputer.pkl"
            scaler_file = self.config.scalers_dir / f"{branch_lower}_scaler.pkl"
            angle_scaler_file = self.config.scalers_dir / f"{branch_lower}_angle_scaler.pkl"

            # Also save directly in preprocessing_dir for legacy paths
            imputer_file_alt = self.config.preprocessing_dir / f"{branch_lower}_imputer.pkl"
            scaler_file_alt = self.config.preprocessing_dir / f"{branch_lower}_scaler.pkl"
            angle_scaler_file_alt = self.config.preprocessing_dir / f"{branch_lower}_angle_scaler.pkl"

            for obj, p_list in [
                (pca, [pca_file]),
                (imputer, [imputer_file, imputer_file_alt]),
                (scaler, [scaler_file, scaler_file_alt]),
                (angle_scaler, [angle_scaler_file, angle_scaler_file_alt])
            ]:
                for p in p_list:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "wb") as f:
                        pickle.dump(obj, f)
            print(f"    [+] Saved PCA, Scaler, Imputer, and AngleScaler models to {self.config.preprocessing_dir}", flush=True)

            # 7. Save Centralized Datasets (.npz) (Saving both naming conventions)
            centralized_files = [
                self.config.centralized_dir / f"{branch_lower}_branch_train_test.npz",
                self.config.centralized_dir / f"{branch_lower}_branch.npz"
            ]
            for c_file in centralized_files:
                c_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    c_file,
                    X_train=X_train_qml,
                    X_test=X_test_qml,
                    y_train=y_train,
                    y_test=y_test,
                    y_train_binary=y_train_binary,
                    y_test_binary=y_test_binary,
                    feature_names=np.array([f"qubit_{i}" for i in range(n_components)])
                )
            print(f"    [+] Saved Centralized dataset to: {centralized_files[0]}", flush=True)

            # 8. Save Federated Client Datasets (.npz)
            fed_branch_dir = self.config.federated_network_dir if branch == "Network" else self.config.federated_physical_dir
            fed_branch_dir.mkdir(parents=True, exist_ok=True)
            client_artifacts = []
            for c_idx, c_raw_mat in enumerate(client_raw_t):
                y_c, y_c_binary = client_labels[c_idx]
                # Apply strictly the fitted transformers
                X_c_imp = imputer.transform(c_raw_mat)
                X_c_std = scaler.transform(X_c_imp)
                X_c_pca = pca.transform(X_c_std)
                X_c_qml = angle_scaler.transform(X_c_pca)

                client_file = fed_branch_dir / f"client_{c_idx+1}.npz"
                np.savez_compressed(
                    client_file,
                    X_train=X_c_qml,
                    y_train=y_c,
                    y_train_binary=y_c_binary
                )
                client_artifacts.append(client_file)
            print(f"    [+] Saved {len(client_artifacts)} Federated Client datasets to: {fed_branch_dir}", flush=True)

            # 9. Audit Data Leakage and Dimensionality Reduction (Issues #2 & Zero-Leakage)
            leakage_rec = {
                "branch": branch,
                "fit_only_on_train": "TRUE (Imputer, Scaler, PCA, AngleScaler fitted on X_train only)",
                "raw_features_count": n_raw_features,
                "qml_features_count": X_train_qml.shape[1],
                "pca_explained_variance_pct": round(exp_var, 2),
                "pca_dimensionality_reduced": is_reduced,
                "pca_reduction_status": reduction_status,
                "train_min_bound": round(float(train_min), 4),
                "train_max_bound": round(float(train_max), 4),
                "test_min_bound": round(float(test_min), 4),
                "test_max_bound": round(float(test_max), 4),
                "pauli_bound_satisfied": bool(test_min >= -np.pi - 1e-4 and test_max <= np.pi + 1e-4)
            }
            self.leakage_records.append(leakage_rec)

            processed_outputs[branch] = {
                "X_train": X_train_qml,
                "X_test": X_test_qml,
                "y_train": y_train,
                "y_test": y_test,
                "y_train_binary": y_train_binary,
                "y_test_binary": y_test_binary,
                "explained_var": exp_var,
                "raw_feature_count": n_raw_features,
                "qml_feature_count": X_train_qml.shape[1],
                "pca_dimensionality_reduced": is_reduced,
                "pca_reduction_status": reduction_status
            }

        # Save data_leakage_report.csv
        leakage_df = pd.DataFrame(self.leakage_records)
        leakage_path = self.config.reports_dir / "data_leakage_report.csv"
        leakage_df.to_csv(leakage_path, index=False)
        print(f"\n[+] Saved data leakage verification report to: {leakage_path}", flush=True)

        # Save skewness_report.csv (Issue #7)
        if self.skewness_records:
            skew_df = pd.DataFrame(self.skewness_records)
            skew_path = self.config.reports_dir / "skewness_report.csv"
            skew_df.to_csv(skew_path, index=False)
            print(f"[+] Saved feature skewness reduction report to: {skew_path}", flush=True)

            print("\n" + "="*65, flush=True)
            print("  SKEWNESS REDUCTION AUDIT (Issue #7: Log1p on Heavy Tails)", flush=True)
            print("="*65, flush=True)
            for _, r in skew_df[skew_df['log1p_applied']].iterrows():
                print(f"  {r['feature_name']:<20} Skew Before: {r['skewness_before']:>8.2f} -> After: {r['skewness_after']:>8.2f}", flush=True)
            print("="*65 + "\n", flush=True)

        return processed_outputs
