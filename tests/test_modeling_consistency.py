"""
CI Consistency and Integrity Test Suite for SwarmGuard Modeling Phase.
Verifies that SWARMGUARD_MODELING_LOG.xlsx contains all 8 sheets with real data,
and validates the 12-qubit decoupled QCNN architecture and parameter counts.
"""

import re
import sys
import pytest
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
import openpyxl

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig
from swarmguard_modeling.qcnn_ansatz import QCNNModel

@pytest.fixture
def config():
    return PipelineConfig()

def test_qcnn_ansatz_parameter_budget():
    """Verifies that the 12-qubit QCNN ansatz has exactly 36 trainable parameters."""
    for branch in ["Network", "Physical"]:
        model = QCNNModel(branch, 12)
        meta = model.get_circuit_metadata()
        assert meta["trainable_parameters"] == 36, f"{branch} QCNN parameter count mismatch: {meta['trainable_parameters']} != 36"
        assert meta["num_qubits"] == 12
        assert meta["circuit_depth"] > 0

def _encoding_wires(model: QCNNModel):
    """Maps input feature index j -> set of wires its Ry(x_j) gates act on, read from the built circuit."""
    wires = {}
    for inst in model.circuit.data:
        if inst.operation.name != "ry":
            continue
        names = {p.name for p in inst.operation.params[0].parameters}
        x_names = [n for n in names if n.startswith("x[")]
        if x_names:
            j = int(x_names[0][2:-1])
            wires.setdefault(j, set()).add(model.circuit.find_bit(inst.qubits[0]).index)
    return wires

def test_strongest_pca_components_on_pass_through_wires():
    """PCA features arrive in descending variance; features 0-3 must land on pass-through wires 8-11."""
    wires = _encoding_wires(QCNNModel("Network", 12))
    assert sorted(wires) == list(range(12))
    for j in range(4):
        assert wires[j] == {11 - j}, f"Strong PCA feature {j} encoded on {wires[j]}, expected pass-through wire {11 - j}"
    for j in range(4, 12):
        assert max(wires[j]) <= 7, f"Weak PCA feature {j} encoded on {wires[j]}, expected a pre-pooled wire (0-7)"

def test_crz_parameters_use_4pi_period_in_federated_aggregation():
    """CRz-gated parameters are 4pi-periodic; aggregation and wrapping must not treat them as 2pi-periodic."""
    import numpy as np
    from swarmguard_modeling.federated_trainer import FederatedQCNNTrainer

    model = QCNNModel("Network", 12)
    crz_idx = [i for i, p in enumerate(model.param_periods) if np.isclose(p, 4 * np.pi)]
    assert crz_idx == [12, 13, 14, 15, 24, 25, 26, 27, 32, 33, 35]

    x = np.full(12, 0.7)
    w = model.weights.copy()
    w[12] = 2.5
    shifted_2pi, shifted_4pi = w.copy(), w.copy()
    shifted_2pi[12] -= 2 * np.pi
    shifted_4pi[12] -= 4 * np.pi
    assert not np.isclose(model.compute_expectation(x, w), model.compute_expectation(x, shifted_2pi))
    assert np.isclose(model.compute_expectation(x, w), model.compute_expectation(x, shifted_4pi))

    # Two clients holding the same model (theta_12 = 5.0 and 5.0 - 4pi are the same CRz gate) must aggregate
    # to that model. The old 2pi circular mean returned 5.0 - 2pi here, which changes the output.
    trainer = FederatedQCNNTrainer.__new__(FederatedQCNNTrainer)  # aggregation needs only the model
    trainer.global_model = model
    a = model.weights.copy()
    a[12] = 5.0
    b = a.copy()
    b[12] = 5.0 - 4 * np.pi
    agg = trainer.aggregate_circular_mean([a, b], [3, 1])
    assert np.isclose(model.compute_expectation(x, agg), model.compute_expectation(x, a))

def test_federated_baseline_requires_real_centralized_run(tmp_path):
    """The federated benchmark must load the centralized QCNN run output, never fall back to a hardcoded number."""
    from swarmguard_modeling.federated_trainer import load_centralized_qcnn_weights
    with pytest.raises(FileNotFoundError):
        load_centralized_qcnn_weights(PipelineConfig(output_dir=tmp_path))

def test_modeling_log_workbook_structure(config):
    """Verifies that SWARMGUARD_MODELING_LOG.xlsx exists, has all 8 sheets, and is populated."""
    log_path = config.reports_dir / "SWARMGUARD_MODELING_LOG.xlsx"
    if not log_path.exists():
        alt = repo_root / "reports" / "SWARMGUARD_MODELING_LOG.xlsx"
        if alt.exists():
            log_path = alt

    assert log_path.exists(), f"SWARMGUARD_MODELING_LOG.xlsx not found at {log_path}"

    wb = openpyxl.load_workbook(log_path, data_only=True)
    expected_sheets = [
        "Log",
        "0_Angle_Domain_Fix",
        "1_QCNN_Ansatz",
        "2_Control_Matrix",
        "3_Optimizers",
        "4_Federated_Training",
        "5_Alert_Fusion",
        "6_QPU_Submission"
    ]

    assert len(wb.sheetnames) == 8, f"Expected 8 sheets in modeling log, found {len(wb.sheetnames)}: {wb.sheetnames}"
    for s_name in expected_sheets:
        assert s_name in wb.sheetnames, f"Missing sheet '{s_name}' in modeling log"
        ws = wb[s_name]
        assert ws.max_row >= 5, f"Sheet '{s_name}' appears empty (max_row = {ws.max_row})"

# IBM Quantum Platform job IDs are 20 lowercase alphanumeric chars (e.g. "daq9f7lb46qs73a0so3g")
IBM_JOB_ID_PATTERN = re.compile(r"^[a-z0-9]{20}$")

def _qpu_submission_rows(log_path: Path):
    """Yields the data rows below the 'Hardware Backend' header in sheet 6_QPU_Submission."""
    ws = openpyxl.load_workbook(log_path, data_only=True)["6_QPU_Submission"]
    rows = list(ws.iter_rows(values_only=True))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Hardware Backend")
    return [r for r in rows[header_idx + 1:] if any(v not in (None, "") for v in r)]

def test_qpu_log_has_no_fabricated_hardware_results(config):
    """
    Fails if 6_QPU_Submission claims a hardware result without a real IBM job ID,
    or if a NOT_RUN row carries placeholder job IDs / expectation values.
    """
    log_path = config.reports_dir / "SWARMGUARD_MODELING_LOG.xlsx"
    if not log_path.exists():
        log_path = repo_root / "reports" / "SWARMGUARD_MODELING_LOG.xlsx"
    assert log_path.exists(), f"SWARMGUARD_MODELING_LOG.xlsx not found at {log_path}"

    rows = _qpu_submission_rows(log_path)
    assert rows, "6_QPU_Submission has no result rows"
    for backend, job_id, _, depth, shots, sim_z, hw_z, status in (r[:8] for r in rows):
        status = str(status)
        if status.startswith("NOT_RUN - "):
            assert job_id in (None, ""), f"NOT_RUN row must not carry a job ID, found {job_id!r}"
            assert sim_z in (None, "") and hw_z in (None, ""), f"NOT_RUN row must not carry <Z> values, found {sim_z!r}/{hw_z!r}"
            continue
        assert "PASSED" in status.upper() or status.startswith("COMPLETED"), f"Unrecognized QPU status {status!r}"
        assert job_id is not None and IBM_JOB_ID_PATTERN.match(str(job_id)), (
            f"Row with status {status!r} has job ID {job_id!r}, which is not a real IBM Quantum job ID"
        )
        float(sim_z), float(hw_z)

def test_qpu_failure_is_logged_as_not_run():
    """Any failure (connect or job) must produce a NOT_RUN row with the real error and blank numeric fields."""
    from scripts.update_modeling_log import build_qpu_table

    def failing_connect():
        raise RuntimeError("401 Unauthorized: invalid token")

    class FailingJobExecutor:
        class backend:
            name = "ibm_marrakesh"
        def evaluate_qpu_job(self, sample_x, shots=1024):
            raise AttributeError("'DataBin' object has no attribute 'c0'")

    for factory, expected_backend, expected_status in [
        (failing_connect, None, "NOT_RUN - RuntimeError: 401 Unauthorized: invalid token"),
        (FailingJobExecutor, "ibm_marrakesh", "NOT_RUN - AttributeError: 'DataBin' object has no attribute 'c0'"),
    ]:
        (row,) = build_qpu_table(executor_factory=factory)
        backend, job_id, _, depth, shots, sim_z, hw_z, status = row
        assert backend == expected_backend
        assert status == expected_status
        assert job_id is None and depth is None and shots is None and sim_z is None and hw_z is None

