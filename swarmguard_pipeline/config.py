"""
Configuration and constants for SwarmGuard Preprocessing Pipeline.
Unified Configuration and Constants for SwarmGuard Preprocessing Pipeline.
Supports environment variables (.env via python-dotenv), CLI overrides, and sensible cross-platform defaults.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# The Single Canonical 9-Class Taxonomy Definition
TAXONOMY_CLASSES: List[str] = [
    "BENIGN",               # 0
    "DoS",                  # 1
    "DDoS",                 # 2
    "Jamming_Deauth",       # 3
    "Spoofing_MITM",        # 4
    "FDI",                  # 5
    "Routing_Attack",       # 6
    "Replay",               # 7
    "Recon_Infiltration"    # 8
]

UNKNOWN_CLASS_LABEL: str = "UNKNOWN"
UNKNOWN_CLASS_ID: int = -1

# Canonical dataset target files
DEFAULT_TARGET_FILES: Dict[str, str] = {
    "uavids": "UAVIDS-2025.csv",
    "uav_ndd": "UAV-NDD CSV.zip",
    "csv_iot": "CSV.zip",
    "flows_iscx": "GeneratedLabelledFlows.zip",
    "cve_ml": "MachineLearningCSV.zip",
    "tits": "Dataset_T-ITS.csv",
}

def get_default_input_dirs() -> List[Path]:
    """Resolves default search paths from env or cross-platform locations."""
    env_dirs = os.getenv("INPUT_SEARCH_DIRS")
    if env_dirs:
        separator = ";" if ";" in env_dirs else ","
        return [Path(p.strip()) for p in env_dirs.split(separator) if p.strip()]

    # Standard fallback paths
    candidates = [
        Path("./data/raw"),
        Path("./data"),
        Path("../data/raw"),
        Path("../data"),
        Path.cwd(),
        Path.home() / "OneDrive" / "Desktop",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]
    # Return unique paths maintaining order
    seen = set()
    result = []
    for c in candidates:
        norm = str(c.resolve()) if c.exists() else str(c)
        if norm not in seen:
            seen.add(norm)
            result.append(c)
    return result

def get_default_output_dir() -> Path:
    """Resolves default output directory from env or relative path."""
    env_out = os.getenv("OUTPUT_DIR")
    if env_out:
        return Path(env_out.strip())
    
    # Default to ./data/processed if running locally, or fallback to MERGED_CSV
    if Path("./data/processed").parent.exists():
        return Path("./data/processed")
    return Path("./MERGED_CSV")

@dataclass
class PipelineConfig:
    # Potential source file locations
    input_search_dirs: List[Path] = field(default_factory=lambda: [
        Path(r"C:\Users\Aarush\OneDrive\Desktop"),
        Path(r"C:\Users\Aarush\Downloads"),
        Path(r"C:\Users\Aarush\Desktop"),
    ])
    # Input search directories
    input_search_dirs: List[Path] = field(default_factory=get_default_input_dirs)

    # Target output base directory
    output_dir: Path = Path(r"C:\Users\Aarush\OneDrive\Desktop\SwarmGuard_Processed_Data")
    # Output base directory
    output_dir: Path = field(default_factory=get_default_output_dir)

    # Specific dataset filenames to search for
    target_files: Dict[str, str] = field(default_factory=lambda: {
        "uavids": "UAVIDS-2025.csv",
        "uav_ndd": "UAV-NDD CSV.zip",
        "csv_iot": "CSV.zip",
        "flows_iscx": "GeneratedLabelledFlows.zip",
        "cve_ml": "MachineLearningCSV.zip",
        "tits": "Dataset_T-ITS.csv",
    })
    target_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TARGET_FILES))

    # Cleaning & Imputation Thresholds
    impute_null_threshold: float = 0.05  # Impute if missing < 5%
    drop_col_null_threshold: float = 0.30 # Drop column if missing > 30%
    impute_null_threshold: float = 0.05   # Impute if missing < 5%
    drop_col_null_threshold: float = 0.30  # Drop column if missing > 30%

    # Train / Test Splitting
    test_size: float = 0.20
    random_state: int = 42

    # Federated Non-IID Partitioning
    num_federated_clients: int = 5
    dirichlet_alpha: float = 0.5  # Non-IID heterogeneity parameter

    # Quantum 12-Qubit Parameters
    num_qubits: int = 12
    quantum_angle_min: float = -np.pi
    quantum_angle_max: float = np.pi

    # Sample capacity limits for balanced processing (memory & speed efficiency)
    max_network_samples: int = 200000
    max_physical_samples: int = 100000
    # Named Sampling Caps (Disclosed & Configurable)
    max_cic_iot_folders: int = 20
    max_cic_iot_rows_per_file: int = 1000
    max_cicids2017_files: int = 5
    max_cicids2017_rows_per_file: int = 2000
    max_uav_ndd_case1_rows: int = 60000
    max_uav_ndd_excel_rows: int = 15000

    # Flags
    full_ingest: bool = False     # If True, bypass all sampling caps
    write_reports: bool = False   # If True, generate and overwrite report artifacts

    # 9-Class Taxonomy Map
    taxonomy_classes: List[str] = field(default_factory=lambda: [
        "BENIGN",
        "DoS",
        "DDoS",
        "Jamming_Deauth",
        "Spoofing_MITM",
        "FDI",
        "Routing_Attack",
        "Replay",
        "Recon_Infiltration"
    ])
    taxonomy_classes: List[str] = field(default_factory=lambda: list(TAXONOMY_CLASSES))

    # Aliases for backward compatibility
    @property
    def INPUT_SEARCH_DIRS(self) -> List[Path]:
        return self.input_search_dirs

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.output_dir

    @property
    def TARGET_FILES(self) -> Dict[str, str]:
        return self.target_files

    @property
    def TAXONOMY_CLASSES(self) -> List[str]:
        return self.taxonomy_classes

    @property
    def NUM_QUBITS(self) -> int:
        return self.num_qubits

    @property
    def QUANTUM_ANGLE_MIN(self) -> float:
        return self.quantum_angle_min

    @property
    def QUANTUM_ANGLE_MAX(self) -> float:
        return self.quantum_angle_max

    @property
    def TEST_SPLIT_SIZE(self) -> float:
        return self.test_size

    @property
    def NUM_FEDERATED_CLIENTS(self) -> int:
        return self.num_federated_clients

    @property
    def DIRICHLET_ALPHA(self) -> float:
        return self.dirichlet_alpha

    @property
    def RANDOM_SEED(self) -> int:
        return self.random_state

    @property
    def MISSING_DROP_THRESHOLD(self) -> float:
        return self.drop_col_null_threshold

    @property
    def MISSING_IMPUTE_THRESHOLD(self) -> float:
        return self.impute_null_threshold

    # Directory Paths
    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def centralized_dir(self) -> Path:
        return self.output_dir / "centralized"

    @property
    def centralized_ml_dir(self) -> Path:
        return self.output_dir / "centralized_ml"

    @property
    def federated_dir(self) -> Path:
        return self.output_dir / "federated_clients"

    @property
    def federated_network_dir(self) -> Path:
        return self.federated_dir / "network"

    @property
    def federated_physical_dir(self) -> Path:
        return self.federated_dir / "physical"

    @property
    def preprocessing_dir(self) -> Path:
        return self.output_dir / "preprocessing_objects"

    @property
    def scalers_dir(self) -> Path:
        return self.preprocessing_dir / "scalers"

    def create_directories(self):
        """Ensure all required output directories exist."""
        for d in [
            self.output_dir,
            self.reports_dir,
            self.centralized_dir,
            self.centralized_ml_dir,
            self.federated_dir,
            self.federated_network_dir,
            self.federated_physical_dir,
            self.preprocessing_dir,
            self.scalers_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

# Aliases
SwarmGuardConfig = PipelineConfig
