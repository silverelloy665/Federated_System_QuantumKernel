"""
SwarmGuard: Two-Branch Federated Quantum-Kernel UAV Intrusion Detection Preprocessing Pipeline.
Script: preprocess_uav_ids.py (Canonical Pipeline CLI Entrypoint)

Delegates to the modular swarmguard_pipeline implementation.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig, SwarmGuardConfig, TAXONOMY_CLASSES
from swarmguard_pipeline.run_pipeline import execute_pipeline, main

if __name__ == "__main__":
    main()
