"""
Phase D: 12-Qubit QML Formatting & Quantum Phase Angle Scaling.
Applies zero-leakage transforms:
1. Imputer (fitted strictly on X_train)
2. StandardScaler (fitted strictly on X_train)
3. PCA-12 (fitted strictly on X_train)
4. Quantum Phase Angle Scaler [-pi, pi] (fitted strictly on X_train)

Exports serialized preprocessing artifacts, .npz dataset files, and audits zero leakage.
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
from .config import PipelineConfig

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

    def get_feature_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Separates feature matrix from multi-class and binary labels."""
        feature_cols = [c for c in df.columns if c not in ['canonical_label', 'class_id', 'binary_label', '_source_dataset']]
        X = df[feature_cols].values.astype(np.float64)
        y_multi = df['class_id'].values.astype(np.int64)
        y_binary = df['binary_label'].values.astype(np.int64)
        return X, y_multi, y_binary, feature_cols

    def run_quantum_formatting(self, split_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes Phase D: Zero-leakage imputer -> scaler -> PCA-12 -> quantum angle scaling.
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

            print(f"    Raw features count: {X_train_raw.shape[1]} ({', '.join(f_cols[:6])}...)", flush=True)

            # 1. Zero-Leakage Imputation (Fitted strictly on X_train)
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)

            # 2. Fit StandardScaler strictly on Training Data
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train_imp)
            X_test_std = scaler.transform(X_test_imp)

            # 3. Fit PCA-12 strictly on Training Data
            n_components = min(self.config.num_qubits, X_train_raw.shape[1])
            pca = PCA(n_components=n_components, random_state=self.config.random_state)
            X_train_pca = pca.fit_transform(X_train_std)
            X_test_pca = pca.transform(X_test_std)
            exp_var = float(np.sum(pca.explained_variance_ratio_)) * 100
            print(f"    PCA-12 Fitted on Train. Total Explained Variance: {exp_var:.2f}%", flush=True)

            # 4. Fit Quantum Phase Angle Scaler strictly on Training PCA Features
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

            # 5. Save Preprocessing Objects (.pkl)
            branch_lower = branch.lower()
            pca_file = self.config.preprocessing_dir / f"{branch_lower}_pca_12.pkl"
            imputer_file = self.config.scalers_dir / f"{branch_lower}_imputer.pkl"
            scaler_file = self.config.scalers_dir / f"{branch_lower}_scaler.pkl"
            angle_scaler_file = self.config.scalers_dir / f"{branch_lower}_angle_scaler.pkl"

            with open(pca_file, "wb") as f:
                pickle.dump(pca, f)
            with open(imputer_file, "wb") as f:
                pickle.dump(imputer, f)
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            with open(angle_scaler_file, "wb") as f:
                pickle.dump(angle_scaler, f)
            print(f"    [+] Saved PCA and Scaler models to preprocessing_objects/", flush=True)

            # 6. Save Centralized Datasets (.npz)
            centralized_file = self.config.centralized_dir / f"{branch_lower}_branch.npz"
            np.savez_compressed(
                centralized_file,
                X_train=X_train_qml,
                X_test=X_test_qml,
                y_train=y_train,
                y_test=y_test,
                y_train_binary=y_train_binary,
                y_test_binary=y_test_binary,
                feature_names=np.array([f"qubit_{i}" for i in range(n_components)])
            )
            print(f"    [+] Saved Centralized dataset to: {centralized_file}", flush=True)

            # 7. Save Federated Client Datasets (.npz)
            fed_branch_dir = self.config.federated_network_dir if branch == "Network" else self.config.federated_physical_dir
            client_artifacts = []
            for c_idx, c_df in enumerate(client_dfs):
                X_c_raw, y_c, y_c_binary, _ = self.get_feature_and_labels(c_df)
                # Apply strictly the fitted transformers
                X_c_imp = imputer.transform(X_c_raw)
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

            # 8. Audit Data Leakage
            leakage_rec = {
                "branch": branch,
                "fit_only_on_train": "TRUE (Imputer, Scaler, PCA, AngleScaler fitted on X_train only)",
                "raw_features_count": X_train_raw.shape[1],
                "qml_features_count": X_train_qml.shape[1],
                "pca_explained_variance_pct": round(exp_var, 2),
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
                "raw_feature_count": X_train_raw.shape[1],
                "qml_feature_count": X_train_qml.shape[1]
            }

        # Save data_leakage_report.csv
        leakage_df = pd.DataFrame(self.leakage_records)
        leakage_path = self.config.reports_dir / "data_leakage_report.csv"
        leakage_df.to_csv(leakage_path, index=False)
        print(f"\n[+] Saved data leakage verification report to: {leakage_path}", flush=True)

        return processed_outputs
