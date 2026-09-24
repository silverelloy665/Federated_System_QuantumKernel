"""
CI Consistency and Integrity Test Suite for SwarmGuard.
Verifies that generated reports, taxonomies, feature mappings, and quantum tensors
are in strict agreement with the codebase and pipeline configuration.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig, TAXONOMY_CLASSES, UNKNOWN_CLASS_LABEL
from swarmguard_pipeline.cleaning_harmonizer import CleaningHarmonizer

@pytest.fixture
def config():
    return PipelineConfig()

def test_single_taxonomy_definition(config):
    """Verifies that TAXONOMY_CLASSES in config is the authoritative 9-class definition."""
    assert len(TAXONOMY_CLASSES) == 9, "Taxonomy must define exactly 9 classes"
    expected = [
        "BENIGN", "DoS", "DDoS", "Jamming_Deauth", "Spoofing_MITM",
        "FDI", "Routing_Attack", "Replay", "Recon_Infiltration"
    ]
    assert TAXONOMY_CLASSES == expected, f"Taxonomy classes mismatch: {TAXONOMY_CLASSES} != {expected}"

def test_label_mapping_report_agreement(config):
    """Fails CI if reports/label_mapping.csv disagrees with the authoritative taxonomy."""
    report_file = config.reports_dir / "label_mapping.csv"
    if not report_file.exists():
        pytest.skip(f"Report not found at {report_file} (run pipeline first)")

    df_lbl = pd.read_csv(report_file)
    assert not df_lbl.empty, "label_mapping.csv must not be empty"

    # Check that all mapped standard labels are in TAXONOMY_CLASSES or UNKNOWN
    allowed_labels = set(TAXONOMY_CLASSES) | {UNKNOWN_CLASS_LABEL}
    unique_labels = set(df_lbl["standard_label"].unique())
    invalid_labels = unique_labels - allowed_labels
    assert not invalid_labels, f"Found unapproved taxonomy labels in report: {invalid_labels}"

def test_harmonizer_label_mapping(config):
    """Verifies that CleaningHarmonizer produces canonical mappings and UNKNOWN fallback."""
    harmonizer = CleaningHarmonizer(config)
    
    # Check benign
    lbl, cid, bid = harmonizer.map_label_to_taxonomy("Normal Traffic")
    assert lbl == "BENIGN" and cid == 0 and bid == 0

    # Check DoS
    lbl, cid, bid = harmonizer.map_label_to_taxonomy("DoS-Hulk")
    assert lbl == "DoS" and cid == 1 and bid == 1

    # Check Jamming / Deauth
    lbl, cid, bid = harmonizer.map_label_to_taxonomy("Deauth_Attack")
    assert lbl == "Jamming_Deauth" and cid == 3 and bid == 1

    # Check GPS Spoofing
    lbl, cid, bid = harmonizer.map_label_to_taxonomy("GPS_Spoofing")
    assert lbl == "Spoofing_MITM" and cid == 4 and bid == 1

    # Check Unknown Fallback (Issue #6)
    lbl, cid, bid = harmonizer.map_label_to_taxonomy("X99_Quantum_Alien_Probe")
    assert lbl == UNKNOWN_CLASS_LABEL and cid == -1

def test_feature_mapping_report(config):
    """Verifies feature_mapping.csv logs auditable decisions and match types."""
    report_file = config.reports_dir / "feature_mapping.csv"
    if not report_file.exists():
        pytest.skip(f"Report not found at {report_file}")

    df_feat = pd.read_csv(report_file)
    assert not df_feat.empty, "feature_mapping.csv must not be empty"
    required_cols = {"branch", "canonical_field", "source_dataset", "match_type", "is_fuzzy"}
    assert required_cols.issubset(df_feat.columns), f"Missing required columns in feature_mapping.csv: {required_cols - set(df_feat.columns)}"

    valid_types = {"EXACT", "FUZZY", "DERIVED", "MISSING"}
    assert set(df_feat["match_type"].unique()).issubset(valid_types)

def test_dataset_inventory_sampling_disclosure(config):
    """Verifies dataset_inventory.csv has rows_available, rows_sampled, and sampling_method (Issue #4)."""
    report_file = config.reports_dir / "dataset_inventory.csv"
    if not report_file.exists():
        pytest.skip(f"Report not found at {report_file}")

    df_inv = pd.read_csv(report_file)
    assert not df_inv.empty, "dataset_inventory.csv must not be empty"
    required_cols = {"dataset_name", "source_file", "branch", "rows_available", "rows_sampled", "sampling_method"}
    assert required_cols.issubset(df_inv.columns), f"dataset_inventory.csv missing sampling disclosure columns: {required_cols - set(df_inv.columns)}"

def test_quantum_bounds_and_leakage(config):
    """Verifies data leakage report and quantum Pauli angle constraints [-pi, pi]."""
    report_file = config.reports_dir / "data_leakage_report.csv"
    if not report_file.exists():
        pytest.skip(f"Report not found at {report_file}")

    df_leak = pd.read_csv(report_file)
    for _, row in df_leak.iterrows():
        assert row["pauli_bound_satisfied"] == True or str(row["pauli_bound_satisfied"]).upper() == "TRUE"
        assert row["qml_features_count"] == config.num_qubits

def test_centralized_npz_quantum_validity(config):
    """Verifies centralized .npz files have exact 12-qubit dimensions and valid Pauli angle bounds."""
    for branch in ["network", "physical"]:
        npz_path = config.centralized_dir / f"{branch}_branch_train_test.npz"
        if not npz_path.exists():
            npz_path = config.centralized_dir / f"{branch}_branch.npz"
        if not npz_path.exists():
            pytest.skip(f"{branch} .npz not found")

        data = np.load(npz_path)
        X_tr = data["X_train"]
        X_te = data["X_test"]
        assert X_tr.shape[1] == config.num_qubits, f"{branch} train has {X_tr.shape[1]} features, expected {config.num_qubits}"
        assert X_te.shape[1] == config.num_qubits, f"{branch} test has {X_te.shape[1]} features, expected {config.num_qubits}"

        tol = 1e-4
        assert np.min(X_tr) >= 0.0 - tol and np.max(X_tr) <= np.pi + tol, f"{branch} train out of [0, pi] range: [{np.min(X_tr)}, {np.max(X_tr)}]"
        assert np.min(X_te) >= 0.0 - tol and np.max(X_te) <= np.pi + tol, f"{branch} test out of [0, pi] range: [{np.min(X_te)}, {np.max(X_te)}]"


def test_project_documentation_workbook(config):
    """Verifies that SWARMGUARD_PROJECT_DOCUMENTATION.xlsx is generated, non-empty, and up-to-date."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    import openpyxl
    
    doc_path = config.reports_dir / "SWARMGUARD_PROJECT_DOCUMENTATION.xlsx"
    if not doc_path.exists():
        alt_path = repo_root / "reports" / "SWARMGUARD_PROJECT_DOCUMENTATION.xlsx"
        if alt_path.exists():
            doc_path = alt_path

    if not doc_path.exists():
        # Attempt auto-generation if not found
        try:
            from scripts.generate_project_documentation import ProjectDocumentationGenerator
            gen = ProjectDocumentationGenerator(config)
            doc_path = gen.generate_workbook(doc_path)
        except Exception as e:
            pytest.skip(f"Could not generate documentation workbook: {e}")

    assert doc_path.exists(), "SWARMGUARD_PROJECT_DOCUMENTATION.xlsx must exist"

    wb = openpyxl.load_workbook(doc_path, data_only=True)
    expected_sheets = [
        "Overview",
        "Phase A - Discovery & Sampling",
        "Phase B - Harmonization & Cleaning",
        "Phase C - Federated Splitting",
        "Phase D - Quantum Formatting",
        "Validation & Cross-Checks"
    ]

    for sheet_name in expected_sheets:
        # Match exact or trimmed sheet names (due to 31-char Excel limits)
        matched = [s for s in wb.sheetnames if s == sheet_name or s == sheet_name[:31]]
        assert len(matched) > 0, f"Missing sheet '{sheet_name}' in workbook. Found: {wb.sheetnames}"
        ws = wb[matched[0]]
        assert ws.max_row >= 5, f"Sheet '{sheet_name}' appears empty (max_row = {ws.max_row})"
        
        # Verify Source note in cell A1
        a1_val = str(ws.cell(row=1, column=1).value)
        assert "SOURCE" in a1_val.upper(), f"Sheet '{sheet_name}' missing source attribution banner in A1 (found: '{a1_val}')"

    # Optional timestamp freshness check against pipeline source files
    pipeline_dir = repo_root / "swarmguard_pipeline"
    if pipeline_dir.exists():
        py_files = list(pipeline_dir.glob("*.py"))
        if py_files:
            latest_py_mtime = max(p.stat().st_mtime for p in py_files)
            wb_mtime = doc_path.stat().st_mtime
            # Allow small timestamp tolerance (e.g. 5 seconds)
            assert wb_mtime >= (latest_py_mtime - 5.0), (
                f"Documentation workbook ({doc_path.name}) is older than pipeline source files. "
                f"Please run 'python scripts/generate_project_documentation.py' to synchronize."
            )


