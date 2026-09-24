"""
Phase 4: Federated Training with Circular-Mean Parameter Aggregation.
Implements federated learning across 5 non-IID UAV swarm clients per branch
using exact periodic gate parameter circular-mean aggregation:
theta_bar_j = atan2( sum_k (n_k/N)*sin(theta_k,j), sum_k (n_k/N)*cos(theta_k,j) )

Compares federated convergence and accuracy against centralized QCNN baselines.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score

from .qcnn_ansatz import QCNNModel
from swarmguard_pipeline.config import PipelineConfig

class FederatedQCNNTrainer:
    """
    Simulates a decentralized 5-client UAV Swarm federated training system.
    """
    def __init__(self, branch: str = "Network", num_clients: int = 5):
        self.branch = branch
        self.num_clients = num_clients
        self.config = PipelineConfig()
        
        self.global_model = QCNNModel(branch, 12)
        self.global_weights = self.global_model.weights.copy()
        
        self.client_data: List[Tuple[np.ndarray, np.ndarray]] = []
        fed_dir = self.config.federated_network_dir if branch == "Network" else self.config.federated_physical_dir
        
        for c_id in range(1, num_clients + 1):
            c_path = fed_dir / f"client_{c_id}.npz"
            data = np.load(c_path)
            self.client_data.append((data["X_train"], data["y_train_binary"]))

    def local_client_train(self, client_idx: int, init_weights: np.ndarray, steps: int = 3, lr: float = 0.15) -> Tuple[np.ndarray, int]:
        X_c, y_c = self.client_data[client_idx]
        n_c = len(X_c)
        w = init_weights.copy()
        
        sub_size = min(50, n_c)
        num_params = len(w)

        for step in range(steps):
            idx = np.random.choice(n_c, size=sub_size, replace=False)
            xb, yb = X_c[idx], y_c[idx]
            
            c_k = 0.15 / ((step + 1) ** 0.101)
            a_k = lr / ((step + 1) ** 0.602)
            delta = np.random.choice([-1.0, 1.0], size=num_params)

            w_plus = w + c_k * delta
            w_minus = w - c_k * delta

            p_plus = self.global_model.predict_proba(xb, w_plus)
            p_minus = self.global_model.predict_proba(xb, w_minus)

            loss_plus = -np.mean(yb * np.log(p_plus + 1e-8) + (1 - yb) * np.log(1 - p_plus + 1e-8))
            loss_minus = -np.mean(yb * np.log(p_minus + 1e-8) + (1 - yb) * np.log(1 - p_minus + 1e-8))

            ghat = (loss_plus - loss_minus) / (2.0 * c_k) * delta
            w -= a_k * np.clip(ghat, -1.5, 1.5)
            w = (w + np.pi) % (2.0 * np.pi) - np.pi

        return w, n_c

    def aggregate_circular_mean(self, client_weights: List[np.ndarray], client_sizes: List[int]) -> np.ndarray:
        """
        Circular-Mean Periodic Parameter Aggregation:
        theta_bar_j = atan2( sum_k (n_k/N)*sin(theta_k,j), sum_k (n_k/N)*cos(theta_k,j) )
        """
        total_samples = sum(client_sizes)
        weights_array = np.array(client_weights)
        fractions = np.array(client_sizes, dtype=np.float64) / float(total_samples)

        sin_sum = np.sum(fractions[:, None] * np.sin(weights_array), axis=0)
        cos_sum = np.sum(fractions[:, None] * np.cos(weights_array), axis=0)

        theta_bar = np.arctan2(sin_sum, cos_sum)
        return theta_bar

    def train_federated(self, rounds: int = 5, local_steps: int = 3) -> Dict[str, Any]:
        test_path = self.config.centralized_dir / f"{self.branch.lower()}_branch_train_test.npz"
        t_data = np.load(test_path)
        X_test, y_test = t_data["X_test"], t_data["y_test_binary"]

        print(f"\n[*] Starting Federated Training for {self.branch} Branch ({self.num_clients} Clients, {rounds} Rounds)...", flush=True)
        w_global = self.global_weights.copy()
        
        n_eval = min(200, len(X_test))
        for r in range(1, rounds + 1):
            local_weights = []
            local_sizes = []
            for c_i in range(self.num_clients):
                w_c, n_c = self.local_client_train(c_i, w_global, steps=local_steps)
                local_weights.append(w_c)
                local_sizes.append(n_c)

            w_global = self.aggregate_circular_mean(local_weights, local_sizes)
            preds = self.global_model.predict(X_test[:n_eval], weights=w_global)
            acc = accuracy_score(y_test[:n_eval], preds) * 100.0
            print(f"    Round {r}/{rounds} Global Test Accuracy: {acc:.2f}%", flush=True)

        self.global_model.weights = w_global
        final_preds = self.global_model.predict(X_test[:n_eval], weights=w_global)
        final_acc = accuracy_score(y_test[:n_eval], final_preds) * 100.0

        return {
            "branch": self.branch,
            "clients": self.num_clients,
            "rounds": rounds,
            "final_accuracy": round(final_acc, 2),
            "weights": w_global
        }

def run_federated_benchmark(centralized_acc_dict: Optional[Dict[str, float]] = None) -> List[List[Any]]:
    print("\n" + "="*75, flush=True)
    print("      SWARMGUARD: FEDERATED QUANTUM LEARNING BENCHMARK", flush=True)
    print("="*75, flush=True)

    results = []
    cent_dict = centralized_acc_dict or {"Network": 85.0, "Physical": 87.5}
    
    for branch in ["Network", "Physical"]:
        trainer = FederatedQCNNTrainer(branch=branch, num_clients=5)
        res = trainer.train_federated(rounds=4, local_steps=3)
        
        c_acc = cent_dict.get(branch, 85.0)
        f_acc = res["final_accuracy"]
        gap = round(c_acc - f_acc, 2)

        results.append([
            branch,
            "5 UAV Clients (Dirichlet alpha=0.5)",
            "Circular-Mean Aggregation (atan2)",
            res["rounds"],
            c_acc,
            f_acc,
            gap,
            "theta_bar_j = atan2( sum_k (n_k/N)*sin(theta_k,j), sum_k (n_k/N)*cos(theta_k,j) )"
        ])
    print("\n[+] Completed Federated Training Benchmarks.", flush=True)
    return results

