"""
Standalone IBM Quantum connection diagnostic (not wired into the pipeline).
Reads the same env variables as swarmguard_modeling/qpu_executor.py and reports
the real connection outcome. Never prints the token value.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env")

token = os.getenv("QISKIT_IBM_TOKEN") or os.getenv("IBM_QUANTUM_TOKEN")
channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")
instance = os.getenv("QISKIT_IBM_INSTANCE")

print(f"Token found: {bool(token)}")
print(f"Channel: {channel}")
print(f"Instance (from env, not passed to service): {instance}")

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    import qiskit_ibm_runtime
    print(f"qiskit-ibm-runtime version: {qiskit_ibm_runtime.__version__}")

    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"SUCCESS: least busy backend = {backend.name}")
    print(f"Qubits: {backend.num_qubits}")
    print(f"Pending jobs (queue length): {backend.status().pending_jobs}")
except Exception as e:
    print(f"FAILURE: {e!r}")
