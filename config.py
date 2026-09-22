"""
SwarmGuard: Pipeline Configuration and Parameters.
"""

from pathlib import Path
import numpy as np

# Base paths
INPUT_SEARCH_DIRS = [
    Path(r"C:\Users\Aarush\OneDrive\Desktop"),
    Path(r"C:\Users\Aarush\Downloads"),
    Path(r"C:\Users\Aarush\Desktop"),
]

OUTPUT_DIR = Path(r"C:\Users\Aarush\OneDrive\Desktop\MERGED_CSV")

TARGET_FILES = {
    "uavids": "UAVIDS-2025.csv",
    "uav_ndd": "UAV-NDD CSV.zip",
    "csv_iot": "CSV.zip",
    "flows_iscx": "GeneratedLabelledFlows.zip",
    "cve_ml": "MachineLearningCSV.zip",
    "tits": "Dataset_T-ITS.csv",
}

# 9-Class Taxonomy
TAXONOMY_CLASSES = [
    "BENIGN",                 # 0
    "GPS_SPOOFING",           # 1
    "GPS_JAMMING",            # 2
    "FALSE_DATA_INJECTION",   # 3
    "REPLAY_ATTACK",          # 4
    "EVIL_TWIN",              # 5
    "DENIAL_OF_SERVICE",      # 6
    "RECONNAISSANCE",         # 7
    "MALWARE_INJECTION"       # 8
]

# Quantum Circuit Parameters
NUM_QUBITS = 12
QUANTUM_ANGLE_MIN = -np.pi
QUANTUM_ANGLE_MAX = np.pi

# Federated Learning & Splitting Parameters
TEST_SPLIT_SIZE = 0.20
NUM_FEDERATED_CLIENTS = 5
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42

# Cleaning Thresholds
MISSING_DROP_THRESHOLD = 0.30
MISSING_IMPUTE_THRESHOLD = 0.05

