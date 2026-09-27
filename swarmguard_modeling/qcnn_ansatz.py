"""
Phase 1: 12-Qubit Decoupled Dual-Branch QCNN Circuit Architecture.
Implements separate Network and Physical QCNN circuits with:
- Layer 1: Ry(x_i) Angle Encoding on 12 wires (features sorted by PCA explained-variance rank)
- Layer 2: Convolution 1 (parameterized 2-qubit unitaries on 6 adjacent pairs)
- Layer 3: Data Re-Upload 1 (Ry(x_i) on all 12 wires)
- Layer 4: Asymmetric Pre-Pool (wires 0-7 pooled down to 4 wires; wires 8-11 pass through -> 8 active wires)
- Layers 5-10: 8 -> 4 -> 2 -> 1 hierarchical Convolution and Pooling layers
- Measurement: Observable sigma_z on surviving wire q0
- High-performance execution engine and capacity-matched 36-parameter budget.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

def feature_wire(feature_idx: int, num_qubits: int = 12) -> int:
    """
    Wire carrying PCA feature `feature_idx`. PCA columns arrive in descending explained-variance
    order (feature 0 = strongest), so reversing puts the strongest 4 on pass-through wires 8-11
    and the weakest 8 on wires 0-7, which the asymmetric pre-pool compresses first.
    """
    return num_qubits - 1 - feature_idx

class QCNNCircuitBuilder:
    """
    Constructs the parameterized 12-qubit QCNN QuantumCircuit in Qiskit.
    """
    def __init__(self, num_qubits: int = 12, name: str = "QCNN_12Q"):
        self.num_qubits = num_qubits
        self.name = name

    def conv_block(self, qc: QuantumCircuit, q1: int, q2: int, param1: Parameter, param2: Parameter):
        """2-qubit parameterized convolution block."""
        qc.ry(param1, q1)
        qc.ry(param2, q2)
        qc.cz(q1, q2)

    def pool_block(self, qc: QuantumCircuit, q_source: int, q_target: int, param: Parameter):
        """2-qubit parameterized pooling block (source -> target)."""
        qc.crz(param, q_source, q_target)
        qc.cx(q_target, q_source)

    def build_circuit(self) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """
        Builds the 12-qubit QCNN circuit.
        Returns: (QuantumCircuit, input_params, trainable_params)
        """
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        
        # 12 input features (x_0 .. x_11)
        x_params = ParameterVector('x', self.num_qubits)
        
        # Exactly 36 trainable parameters matching the capacity budget
        theta = ParameterVector('theta', 36)
        
        # -------------------------------------------------------------
        # Layer 1: Feature Encoding Ry(x_i)
        # Wires 0-7: Weakest 8 PCA components (paired for pre-pooling)
        # Wires 8-11: Strongest 4 PCA components (preserved through pre-pool)
        # -------------------------------------------------------------
        for j in range(self.num_qubits):
            qc.ry(x_params[j], feature_wire(j, self.num_qubits))
        qc.barrier(label="Encoding_1")

        # -------------------------------------------------------------
        # Layer 2: Convolution 1 (6 pairs on 12 qubits -> 12 params)
        # -------------------------------------------------------------
        p_idx = 0
        pairs_c1 = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        for (q1, q2) in pairs_c1:
            self.conv_block(qc, q1, q2, theta[p_idx], theta[p_idx + 1])
            p_idx += 2
        qc.barrier(label="Conv_1")

        # -------------------------------------------------------------
        # Layer 3: Data Re-Upload 1 (Ry(x_i) on all 12 wires)
        # -------------------------------------------------------------
        for j in range(self.num_qubits):
            qc.ry(x_params[j], feature_wire(j, self.num_qubits))
        qc.barrier(label="ReUpload_1")

        # -------------------------------------------------------------
        # Layer 4: Asymmetric Pre-Pool (Wires 0-7 pooled down to 4 wires)
        # Wires 8-11 pass through untouched.
        # Active wires after pre-pool: [0, 2, 4, 6, 8, 9, 10, 11] (8 wires) -> 4 params
        # -------------------------------------------------------------
        pre_pool_pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]
        for (q_src, q_tgt) in pre_pool_pairs:
            self.pool_block(qc, q_src, q_tgt, theta[p_idx])
            p_idx += 1
        qc.barrier(label="PrePool_12to8")

        # -------------------------------------------------------------
        # Layer 5: Convolution 2 on 8 active wires (4 pairs -> 8 params)
        # Active wires: [0, 2, 4, 6, 8, 9, 10, 11]
        # -------------------------------------------------------------
        pairs_c2 = [(0, 2), (4, 6), (8, 9), (10, 11)]
        for (q1, q2) in pairs_c2:
            self.conv_block(qc, q1, q2, theta[p_idx], theta[p_idx + 1])
            p_idx += 2
        qc.barrier(label="Conv_2_8Q")

        # -------------------------------------------------------------
        # Layer 6: Pooling 2 (8 -> 4 active wires -> 4 params)
        # Active wires: [0, 4, 8, 10]
        # -------------------------------------------------------------
        pool_pairs_2 = [(2, 0), (6, 4), (9, 8), (11, 10)]
        for (q_src, q_tgt) in pool_pairs_2:
            self.pool_block(qc, q_src, q_tgt, theta[p_idx])
            p_idx += 1
        qc.barrier(label="Pool_8to4")

        # -------------------------------------------------------------
        # Layer 7: Convolution 3 on 4 active wires (2 pairs -> 4 params)
        # Active wires: [0, 4, 8, 10]
        # -------------------------------------------------------------
        pairs_c3 = [(0, 4), (8, 10)]
        for (q1, q2) in pairs_c3:
            self.conv_block(qc, q1, q2, theta[p_idx], theta[p_idx + 1])
            p_idx += 2
        qc.barrier(label="Conv_3_4Q")

        # -------------------------------------------------------------
        # Layer 8: Pooling 3 (4 -> 2 active wires -> 2 params: theta[32..33])
        # Active wires: [0, 8]
        # -------------------------------------------------------------
        pool_pairs_3 = [(4, 0), (10, 8)]
        for (q_src, q_tgt) in pool_pairs_3:
            self.pool_block(qc, q_src, q_tgt, theta[p_idx])
            p_idx += 1
        qc.barrier(label="Pool_4to2")

        # -------------------------------------------------------------
        # Layer 9: Convolution 4 on 2 active wires (1 pair -> 1 param: theta[34])
        # Active wires: [0, 8]
        # -------------------------------------------------------------
        qc.ry(theta[p_idx], 0)
        qc.cz(0, 8)
        p_idx += 1
        qc.barrier(label="Conv_4_2Q")

        # -------------------------------------------------------------
        # Layer 10: Final Pooling 4 (2 -> 1 active wire -> 1 param: theta[35])
        # Surviving wire: q0
        # -------------------------------------------------------------
        self.pool_block(qc, 8, 0, theta[p_idx])
        p_idx += 1
        qc.barrier(label="Measure_q0")

        return qc, x_params, theta


class QCNNModel:
    """
    Decoupled 12-Qubit QCNN Classification Model.
    Computes exact expectation value <Z_0> of the surviving wire q0.
    """
    def __init__(self, branch_name: str = "Network", num_qubits: int = 12):
        self.branch_name = branch_name
        self.num_qubits = num_qubits
        
        builder = QCNNCircuitBuilder(num_qubits=num_qubits, name=f"QCNN_{branch_name}")
        self.circuit, self.x_params, self.theta_params = builder.build_circuit()
        self.num_trainable_params = len(self.theta_params) # Exactly 36 parameters

        # Period of each trainable parameter: Ry(t + 2pi) = -Ry(t) is a global phase, but
        # CRz(t + 2pi) = (Z on control) . CRz(t) is not, so CRz-gated parameters have period 4pi.
        crz_params = {p for inst in self.circuit.data if inst.operation.name == "crz"
                      for p in inst.operation.params[0].parameters}
        self.param_periods = np.array([4.0 * np.pi if p in crz_params else 2.0 * np.pi for p in self.theta_params])

        # Trainable weights (initialized uniformly in [-pi/4, pi/4])
        np.random.seed(42)
        self.weights = np.random.uniform(-np.pi/4, np.pi/4, size=self.num_trainable_params)
        
        # Scaling & bias parameters
        self.scale = 2.0
        self.bias = 0.0

        # Target observable: Z on wire 0, Identity on wires 1..11
        obs_str = "I" * (self.num_qubits - 1) + "Z"
        self.observable = SparsePauliOp.from_list([(obs_str, 1.0)])

    def get_circuit_metadata(self) -> Dict[str, Any]:
        """Returns transpiled gate operations, depth, and parameter counts."""
        return {
            "branch": self.branch_name,
            "num_qubits": self.num_qubits,
            "trainable_parameters": self.num_trainable_params,
            "circuit_depth": self.circuit.depth(),
            "operations_count": dict(self.circuit.count_ops())
        }

    def compute_expectation(self, x_sample: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Computes <Z_0> for a single input vector."""
        w = self.weights if weights is None else weights
        theta_dict = {p: w[i] for i, p in enumerate(self.theta_params)}
        x_dict = {self.x_params[j]: x_sample[j] for j in range(self.num_qubits)}
        
        bound_circ = self.circuit.assign_parameters({**theta_dict, **x_dict})
        sv = Statevector.from_instruction(bound_circ)
        exp_val = sv.expectation_value(self.observable).real
        return float(exp_val)

    def predict_proba(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes binary classification probability p(y=1) in [0, 1].
        p = sigmoid(scale * <Z_0> + bias)
        """
        w = self.weights if weights is None else weights
        exp_vals = np.array([self.compute_expectation(x, w) for x in X])
        logits = self.scale * exp_vals + self.bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Predicts binary class labels {0, 1}."""
        probs = self.predict_proba(X, weights)
        return (probs >= threshold).astype(int)
