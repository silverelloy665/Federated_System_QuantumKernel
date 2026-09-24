"""
Phase 2: Four-Arm Capacity-Matched Control Matrix (~36 Parameters).
Implements and benchmarks:
(a) Proposed: 12-Qubit QCNN + Re-upload (36 parameters)
(b) Classical Capacity-Matched MLP (36 parameters)
(c) Classical Matrix Product State / Tensor Network (36 parameters)
(d) Standard Hardware-Efficient VQC (Barren-Plateau Control, 36 parameters)

Evaluates on centralized Network and Physical datasets (8 total training runs).
Records train/test accuracy, macro F1, and training wall-clock time.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import expit

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

from .qcnn_ansatz import QCNNModel

def binary_cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)))

def spsa_fit(predict_proba, weights: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
             iterations: int = 60, sample_size: int = 32, lr: float = 0.15, c0: float = 0.15,
             clip: float = 1.0, log_every: int = 10, label: str = "") -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """
    Shared SPSA trainer for the quantum arms (identical budget for QCNN and HE-VQC).
    Returns the trained weights and the full-train-set loss recorded every `log_every` iterations.
    """
    w = weights.copy()
    history = [(0, binary_cross_entropy(predict_proba(X_train, w), y_train))]
    print(f"      [{label}] iter {0:>3} train loss {history[-1][1]:.4f}", flush=True)
    for it in range(iterations):
        idx = np.random.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
        xb, yb = X_train[idx], y_train[idx]

        delta = np.random.choice([-1.0, 1.0], size=len(w))
        c_k = c0 / (it + 1) ** 0.101
        a_k = lr / (it + 1) ** 0.602

        loss_plus = binary_cross_entropy(predict_proba(xb, w + c_k * delta), yb)
        loss_minus = binary_cross_entropy(predict_proba(xb, w - c_k * delta), yb)
        ghat = (loss_plus - loss_minus) / (2.0 * c_k) * delta
        w -= a_k * np.clip(ghat, -clip, clip)

        if (it + 1) % log_every == 0 or it + 1 == iterations:
            history.append((it + 1, binary_cross_entropy(predict_proba(X_train, w), y_train)))
            print(f"      [{label}] iter {it + 1:>3} train loss {history[-1][1]:.4f}", flush=True)
    return w, history

# =========================================================================
# Model (b): Capacity-Matched Classical MLP (36 Parameters)
# Architecture: 12 -> 2 -> 2 -> 1
# Layer 1: 12 * 2 + 2 = 26
# Layer 2: 2 * 2 + 2 = 6
# Layer 3: 2 * 1 + 1 + 1 (gain) = 4
# Total Parameters = 26 + 6 + 4 = 36
# =========================================================================
class CapacityMatchedMLP:
    def __init__(self, input_dim: int = 12):
        self.input_dim = input_dim
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, 2) * 0.15  # 24 params
        self.b1 = np.zeros(2)                          # 2 params
        self.W2 = np.random.randn(2, 2) * 0.15          # 4 params
        self.b2 = np.zeros(2)                          # 2 params
        self.W3 = np.random.randn(2, 1) * 0.15          # 2 params
        self.b3 = np.zeros(1)                          # 1 param
        self.gain = np.ones(1)                         # 1 param
        self.num_params = 36

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)
        logits = (np.dot(h2, self.W3) + self.b3) * self.gain
        return expit(logits.flatten())

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 60, lr: float = 0.08, batch_size: int = 64):
        n_samples = len(X_train)
        self.loss_history = [(0, binary_cross_entropy(self.forward(X_train), y_train))]
        for epoch in range(epochs):
            self._train_epoch(X_train, y_train, n_samples, lr, batch_size)
            self.loss_history.append((epoch + 1, binary_cross_entropy(self.forward(X_train), y_train)))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      [MLP] epoch {epoch + 1:>3} train loss {self.loss_history[-1][1]:.4f}", flush=True)

    def _train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, n_samples: int, lr: float, batch_size: int):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            z1 = np.dot(xb, self.W1) + self.b1
            h1 = np.maximum(0, z1)
            z2 = np.dot(h1, self.W2) + self.b2
            h2 = np.maximum(0, z2)
            logits = (np.dot(h2, self.W3) + self.b3) * self.gain
            probs = expit(logits.flatten())

            dloss = (probs - yb)[:, None] / len(xb)
            dlogits = dloss * self.gain
            dgain = np.sum(dloss * (np.dot(h2, self.W3) + self.b3))

            dW3 = np.dot(h2.T, dlogits)
            db3 = np.sum(dlogits, axis=0)

            dh2 = np.dot(dlogits, self.W3.T)
            dz2 = dh2 * (z2 > 0)
            dW2 = np.dot(h1.T, dz2)
            db2 = np.sum(dz2, axis=0)

            dh1 = np.dot(dz2, self.W2.T)
            dz1 = dh1 * (z1 > 0)
            dW1 = np.dot(xb.T, dz1)
            db1 = np.sum(dz1, axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.gain -= lr * 0.01 * dgain


# =========================================================================
# Model (c): Capacity-Matched Matrix Product State (MPS) (36 Parameters)
# =========================================================================
class CapacityMatchedMPS:
    def __init__(self, num_sites: int = 12):
        self.num_sites = num_sites
        np.random.seed(42)
        self.params = np.random.randn(36) * 0.15
        self.num_params = 36

    def contract_sample(self, x: np.ndarray, p: np.ndarray) -> float:
        vec = np.array([np.cos(x[0] / 2.0), np.sin(x[0] / 2.0)])
        M0 = p[0:4].reshape(2, 2)
        state = np.dot(vec, M0)

        p_idx = 4
        for i in range(1, 10):
            w = p[p_idx:p_idx+3]
            p_idx += 3
            mat = np.array([[w[0], w[1]], [w[1], w[2]]])
            local_vec = np.array([np.cos(x[i] / 2.0), np.sin(x[i] / 2.0)])
            state = np.dot(state, mat) * local_vec[0] + np.dot(state, mat.T) * local_vec[1]
            state = state / (np.linalg.norm(state) + 1e-6)

        M10 = p[31:35].reshape(2, 2)
        end_vec = np.array([np.cos(x[10] / 2.0), np.sin(x[10] / 2.0)])
        final_state = np.dot(state, M10)
        logit = np.dot(final_state, end_vec) * np.cos(x[11] / 2.0) + p[35]
        return float(logit)

    def forward(self, X: np.ndarray) -> np.ndarray:
        logits = np.array([self.contract_sample(x, self.params) for x in X])
        return expit(logits)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 15, lr: float = 0.08, batch_size: int = 64):
        n_samples = len(X_train)
        self.loss_history = [(0, binary_cross_entropy(self.forward(X_train), y_train))]
        for epoch in range(epochs):
            self._train_epoch(X_train, y_train, n_samples, lr, batch_size)
            self.loss_history.append((epoch + 1, binary_cross_entropy(self.forward(X_train), y_train)))
            print(f"      [MPS] epoch {epoch + 1:>3} train loss {self.loss_history[-1][1]:.4f}", flush=True)

    def _train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, n_samples: int, lr: float, batch_size: int):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            eps = 1e-4
            preds = self.forward(xb)
            base_loss = -np.mean(yb * np.log(preds + 1e-8) + (1 - yb) * np.log(1 - preds + 1e-8))

            sample_p = np.random.choice(len(self.params), size=min(6, len(self.params)), replace=False)
            grad = np.zeros_like(self.params)
            for pi in sample_p:
                p_pert = self.params.copy()
                p_pert[pi] += eps
                p_preds = expit(np.array([self.contract_sample(x, p_pert) for x in xb]))
                pert_loss = -np.mean(yb * np.log(p_preds + 1e-8) + (1 - yb) * np.log(1 - p_preds + 1e-8))
                grad[pi] = (pert_loss - base_loss) / eps

            self.params -= lr * grad


# =========================================================================
# Model (d): Standard Hardware-Efficient VQC (Barren-Plateau Control, 36 Params)
# =========================================================================
class HardwareEfficientVQC:
    def __init__(self, num_qubits: int = 12):
        self.num_qubits = num_qubits
        self.num_params = 36
        np.random.seed(42)
        self.weights = np.random.uniform(-np.pi/4, np.pi/4, size=36)
        
        self.x_params = ParameterVector('x', 12)
        self.theta_params = ParameterVector('theta', 36)
        
        qc = QuantumCircuit(12, name="HE_VQC_12Q")
        for i in range(12):
            qc.ry(self.x_params[i], i)
        
        p_idx = 0
        for layer in range(3):
            for i in range(12):
                qc.ry(self.theta_params[p_idx], i)
                p_idx += 1
            for i in range(11):
                qc.cz(i, i+1)
            qc.cz(11, 0)
        
        self.circuit = qc
        obs_str = "I" * 11 + "Z"
        self.observable = SparsePauliOp.from_list([(obs_str, 1.0)])

    def compute_expectation(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        w = self.weights if weights is None else weights
        theta_dict = {p: w[i] for i, p in enumerate(self.theta_params)}
        x_dict = {self.x_params[j]: x[j] for j in range(self.num_qubits)}
        bound = self.circuit.assign_parameters({**theta_dict, **x_dict})
        sv = Statevector.from_instruction(bound)
        return float(sv.expectation_value(self.observable).real)

    def predict_proba(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        w = self.weights if weights is None else weights
        exp_vals = np.array([self.compute_expectation(x, w) for x in X])
        return expit(2.0 * exp_vals)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **spsa_kwargs):
        self.weights, self.loss_history = spsa_fit(self.predict_proba, self.weights, X_train, y_train,
                                                   label="HE-VQC", **spsa_kwargs)


# =========================================================================
# Control Matrix Benchmark Runner
# =========================================================================
QUANTUM_ARM_SPSA = dict(iterations=60, sample_size=32, lr=0.15, c0=0.15, clip=1.0, log_every=10)
CENTRALIZED_RESULTS_FILE = "centralized_training_results.json"

def _measured_notes(history: List[Tuple[int, float]], y_te_pred: np.ndarray, y_te: np.ndarray, unit: str) -> str:
    """Notes column built only from what this run measured."""
    return (f"Train loss {history[0][1]:.4f} -> {history[-1][1]:.4f} over {history[-1][0]} {unit}; "
            f"predicts ATTACK for {y_te_pred.mean() * 100:.1f}% of test (true rate {y_te.mean() * 100:.1f}%)")

def run_control_matrix_benchmark() -> List[List[Any]]:
    print("\n" + "="*75, flush=True)
    print("    SWARMGUARD: FOUR-ARM CAPACITY-MATCHED CONTROL MATRIX BENCHMARK", flush=True)
    print("="*75, flush=True)

    import json
    from swarmguard_pipeline.config import PipelineConfig
    config = PipelineConfig()

    results_table = []
    run_record: Dict[str, Any] = {"spsa_budget_quantum_arms": QUANTUM_ARM_SPSA, "branches": {}}

    for branch in ["Network", "Physical"]:
        npz_file = config.centralized_dir / f"{branch.lower()}_branch_train_test.npz"
        data = np.load(npz_file)
        
        X_train, y_train = data["X_train"], data["y_train_binary"]
        X_test, y_test = data["X_test"], data["y_test_binary"]

        print(f"\n[*] Evaluating {branch} Branch Centralized Benchmark (Train: {len(X_train):,}, Test: {len(X_test):,})...", flush=True)

        np.random.seed(42)
        n_train_sub = min(350, len(X_train))
        n_test_sub = min(150, len(X_test))
        idx_tr = np.random.choice(len(X_train), size=n_train_sub, replace=False)
        idx_te = np.random.choice(len(X_test), size=n_test_sub, replace=False)

        x_tr_eval, y_tr_eval = X_train[idx_tr], y_train[idx_tr]
        x_te_eval, y_te_eval = X_test[idx_te], y_test[idx_te]

        branch_record: Dict[str, Any] = {
            "train_eval_indices": idx_tr.tolist(),
            "test_eval_indices": idx_te.tolist(),
            "models": {}
        }

        # Arm (a): Proposed QCNN + Re-upload (36 Params) -- trained with the shared SPSA budget
        print("  -> Running Arm (a): Proposed 12-Qubit QCNN + Re-upload...", flush=True)
        t0 = time.time()
        qcnn = QCNNModel(branch, 12)
        qcnn.weights, qcnn_history = spsa_fit(qcnn.predict_proba, qcnn.weights, x_tr_eval, y_tr_eval,
                                              label="QCNN", **QUANTUM_ARM_SPSA)
        t_qcnn = time.time() - t0
        qcnn_preds_tr = qcnn.predict_proba(x_tr_eval)
        qcnn_preds_te = qcnn.predict_proba(x_te_eval)
        qcnn_y_tr = (qcnn_preds_tr >= 0.5).astype(int)
        qcnn_y_te = (qcnn_preds_te >= 0.5).astype(int)

        acc_tr_qcnn = accuracy_score(y_tr_eval, qcnn_y_tr) * 100.0
        acc_te_qcnn = accuracy_score(y_te_eval, qcnn_y_te) * 100.0
        f1_te_qcnn = f1_score(y_te_eval, qcnn_y_te, average="macro")

        results_table.append([
            "Proposed QCNN + Re-Upload",
            branch,
            qcnn.num_trainable_params,
            round(acc_tr_qcnn, 2),
            round(acc_te_qcnn, 2),
            round(f1_te_qcnn, 4),
            round(t_qcnn, 2),
            _measured_notes(qcnn_history, qcnn_y_te, y_te_eval, "SPSA iterations")
        ])
        branch_record["models"]["QCNN"] = {
            "train_accuracy": acc_tr_qcnn, "test_accuracy": acc_te_qcnn, "test_macro_f1": f1_te_qcnn,
            "loss_history": qcnn_history, "trained_weights": qcnn.weights.tolist()
        }

        # Arm (b): Capacity-Matched MLP (36 Params)
        print("  -> Running Arm (b): Classical Capacity-Matched MLP (36 Params)...", flush=True)
        t0 = time.time()
        mlp = CapacityMatchedMLP(input_dim=12)
        mlp.fit(x_tr_eval, y_tr_eval, epochs=60, lr=0.08)
        t_mlp = time.time() - t0

        mlp_preds_tr = mlp.forward(x_tr_eval)
        mlp_preds_te = mlp.forward(x_te_eval)
        mlp_y_tr = (mlp_preds_tr >= 0.5).astype(int)
        mlp_y_te = (mlp_preds_te >= 0.5).astype(int)

        acc_tr_mlp = accuracy_score(y_tr_eval, mlp_y_tr) * 100.0
        acc_te_mlp = accuracy_score(y_te_eval, mlp_y_te) * 100.0
        f1_te_mlp = f1_score(y_te_eval, mlp_y_te, average="macro")

        results_table.append([
            "Capacity-Matched MLP",
            branch,
            mlp.num_params,
            round(acc_tr_mlp, 2),
            round(acc_te_mlp, 2),
            round(f1_te_mlp, 4),
            round(t_mlp, 2),
            _measured_notes(mlp.loss_history, mlp_y_te, y_te_eval, "epochs")
        ])
        branch_record["models"]["MLP"] = {
            "train_accuracy": acc_tr_mlp, "test_accuracy": acc_te_mlp, "test_macro_f1": f1_te_mlp,
            "loss_history": mlp.loss_history
        }

        # Arm (c): Capacity-Matched Matrix Product State (MPS) (36 Params)
        print("  -> Running Arm (c): Classical Tensor Network / MPS (36 Params)...", flush=True)
        t0 = time.time()
        mps = CapacityMatchedMPS(num_sites=12)
        mps.fit(x_tr_eval, y_tr_eval, epochs=15, lr=0.05)
        t_mps = time.time() - t0

        mps_preds_tr = mps.forward(x_tr_eval)
        mps_preds_te = mps.forward(x_te_eval)
        mps_y_tr = (mps_preds_tr >= 0.5).astype(int)
        mps_y_te = (mps_preds_te >= 0.5).astype(int)

        acc_tr_mps = accuracy_score(y_tr_eval, mps_y_tr) * 100.0
        acc_te_mps = accuracy_score(y_te_eval, mps_y_te) * 100.0
        f1_te_mps = f1_score(y_te_eval, mps_y_te, average="macro")

        results_table.append([
            "Matrix Product State (MPS)",
            branch,
            mps.num_params,
            round(acc_tr_mps, 2),
            round(acc_te_mps, 2),
            round(f1_te_mps, 4),
            round(t_mps, 2),
            _measured_notes(mps.loss_history, mps_y_te, y_te_eval, "epochs")
        ])
        branch_record["models"]["MPS"] = {
            "train_accuracy": acc_tr_mps, "test_accuracy": acc_te_mps, "test_macro_f1": f1_te_mps,
            "loss_history": mps.loss_history
        }

        # Arm (d): Standard Hardware-Efficient VQC (36 Params)
        print("  -> Running Arm (d): Standard Hardware-Efficient VQC (Barren Plateau Control, 36 Params)...", flush=True)
        t0 = time.time()
        vqc = HardwareEfficientVQC(num_qubits=12)
        vqc.fit(x_tr_eval, y_tr_eval, **QUANTUM_ARM_SPSA)
        t_vqc = time.time() - t0

        vqc_preds_tr = vqc.predict_proba(x_tr_eval)
        vqc_preds_te = vqc.predict_proba(x_te_eval)
        vqc_y_tr = (vqc_preds_tr >= 0.5).astype(int)
        vqc_y_te = (vqc_preds_te >= 0.5).astype(int)

        acc_tr_vqc = accuracy_score(y_tr_eval, vqc_y_tr) * 100.0
        acc_te_vqc = accuracy_score(y_te_eval, vqc_y_te) * 100.0
        f1_te_vqc = f1_score(y_te_eval, vqc_y_te, average="macro")

        results_table.append([
            "Hardware-Efficient VQC",
            branch,
            vqc.num_params,
            round(acc_tr_vqc, 2),
            round(acc_te_vqc, 2),
            round(f1_te_vqc, 4),
            round(t_vqc, 2),
            _measured_notes(vqc.loss_history, vqc_y_te, y_te_eval, "SPSA iterations")
        ])
        branch_record["models"]["HE-VQC"] = {
            "train_accuracy": acc_tr_vqc, "test_accuracy": acc_te_vqc, "test_macro_f1": f1_te_vqc,
            "loss_history": vqc.loss_history
        }
        run_record["branches"][branch] = branch_record

    # Persist the real run so downstream steps (federated baseline) load it instead of hardcoding numbers
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.reports_dir / CENTRALIZED_RESULTS_FILE
    with open(out_path, "w") as f:
        json.dump(run_record, f, indent=2, default=float)
    print(f"\n[+] Saved centralized training results to: {out_path}", flush=True)

    print("\n[+] Completed 8-Arm Centralized Control Matrix Benchmark.", flush=True)
    return results_table

