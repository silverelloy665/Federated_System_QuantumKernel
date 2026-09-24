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
        assert np.min(X_tr) >= -np.pi - tol and np.max(X_tr) <= np.pi + tol
        assert np.min(X_te) >= -np.pi - tol and np.max(X_te) <= np.pi + tol

