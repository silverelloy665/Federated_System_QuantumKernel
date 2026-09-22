"""
Configuration and constants for SwarmGuard Preprocessing Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import numpy as np

@dataclass
class PipelineConfig:
    # Potential source file locations
    input_search_dirs: List[Path] = field(default_factory=lambda: [
        Path(r"C:\Users\Aarush\OneDrive\Desktop"),
        Path(r"C:\Users\Aarush\Downloads"),
        Path(r"C:\Users\Aarush\Desktop"),
    ])

    # Target output base directory
    output_dir: Path = Path(r"C:\Users\Aarush\OneDrive\Desktop\SwarmGuard_Processed_Data")

    # Specific dataset filenames to search for
    target_files: Dict[str, str] = field(default_factory=lambda: {
        "uavids": "UAVIDS-2025.csv",
        "uav_ndd": "UAV-NDD CSV.zip",
        "csv_iot": "CSV.zip",
        "flows_iscx": "GeneratedLabelledFlows.zip",
        "cve_ml": "MachineLearningCSV.zip",
        "tits": "Dataset_T-ITS.csv",
    })

    # Cleaning & Imputation Thresholds
    impute_null_threshold: float = 0.05  # Impute if missing < 5%
    drop_col_null_threshold: float = 0.30 # Drop column if missing > 30%

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

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def centralized_dir(self) -> Path:
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
            self.federated_dir,
            self.federated_network_dir,
            self.federated_physical_dir,
            self.preprocessing_dir,
            self.scalers_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)
