"""
Phase 6: IBM Quantum QPU Submission & Hardware Inference Verification.
Transpiles the trained 12-qubit QCNN circuit onto IBM Quantum Hardware (156-qubit Heron QPU),
submits an inference-only execution job via Qiskit Runtime Service, and verifies
hardware expectation values against ideal simulated baselines.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_aer import AerSimulator

from .qcnn_ansatz import QCNNModel
from swarmguard_pipeline.config import PipelineConfig

load_dotenv()

class QPUInferenceExecutor:
    """
    Submits 12-qubit QCNN inference jobs to IBM Quantum hardware.
    """
    def __init__(self, backend_name: Optional[str] = None):
        token = os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN")
        channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")
        
        self.service = QiskitRuntimeService(channel=channel, token=token)
        if backend_name:
            self.backend = self.service.backend(backend_name)
        else:
            self.backend = self.service.least_busy(operational=True, simulator=False)

    def transpile_qcnn_for_hardware(self, qcnn_model: QCNNModel, sample_x: np.ndarray) -> QuantumCircuit:
        """Binds parameters and transpiles 12-qubit QCNN circuit to native QPU basis gates."""
        theta_dict = {p: qcnn_model.weights[i] for i, p in enumerate(qcnn_model.theta_params)}
        x_dict = {qcnn_model.x_params[j]: sample_x[j] for j in range(12)}
        
        bound_qc = qcnn_model.circuit.assign_parameters({**theta_dict, **x_dict})
        
        # Add measurement on surviving wire q0
        meas_qc = QuantumCircuit(12, 1)
        meas_qc.compose(bound_qc, inplace=True)
        meas_qc.measure(0, 0)
        
        t_qc = transpile(meas_qc, backend=self.backend, optimization_level=2)
        return t_qc

    def evaluate_qpu_job(self, sample_x: np.ndarray, shots: int = 1024) -> Dict[str, Any]:
        """
        Runs hardware evaluation and compares against ideal simulator.
        """
        print(f"\n[*] Transpiling 12-Qubit QCNN for IBM Quantum QPU: {self.backend.name} ({self.backend.num_qubits} Qubits)...")
        qcnn = QCNNModel("Network", 12)
        
        t_qc = self.transpile_qcnn_for_hardware(qcnn, sample_x)
        depth = t_qc.depth()
        ops = dict(t_qc.count_ops())
        print(f"    Transpiled Hardware Circuit Depth: {depth}")
        print(f"    Native Gate Basis Operations: {ops}")

        # Compute ideal expectation value
        sim = AerSimulator()
        sim_result = sim.run(t_qc, shots=shots).result()
        sim_counts = sim_result.get_counts()
        sim_p0 = sim_counts.get("0", 0) / shots
        sim_p1 = sim_counts.get("1", 0) / shots
        sim_z = sim_p0 - sim_p1 # <Z> = P(0) - P(1)

        print(f"\n[*] Submitting Inference Job to {self.backend.name}...")
        t0 = time.time()

        # Real hardware only: any failure propagates to the caller, which logs it as NOT_RUN.
        # Never substitute simulator counts or a synthetic job ID for a hardware result.
        sampler = SamplerV2(mode=self.backend)
        job = sampler.run([t_qc], shots=shots)
        job_id = job.job_id()
        print(f"    Submitted Job ID: {job_id}")
        pub_res = job.result()[0]
        # SamplerV2 keys results by classical register name ("c" for QuantumCircuit(12, 1))
        counts = getattr(pub_res.data, t_qc.cregs[0].name).get_counts()
        hw_p0 = counts.get("0", 0) / shots
        hw_p1 = counts.get("1", 0) / shots
        hw_z = hw_p0 - hw_p1

        exec_time = time.time() - t0
        z_diff = abs(sim_z - hw_z)
        status = "PASSED (Within Shot Tolerance)" if z_diff <= 0.08 else "COMPLETED (Outside Shot Tolerance)"

        return {
            "backend": self.backend.name,
            "job_id": job_id,
            "num_qubits": 12,
            "depth": depth,
            "shots": shots,
            "sim_z": round(float(sim_z), 4),
            "hw_z": round(float(hw_z), 4),
            "time_sec": round(exec_time, 2),
            "status": status
        }

