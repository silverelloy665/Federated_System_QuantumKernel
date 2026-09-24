"""
Root execution entrypoint for SwarmGuard Preprocessing Pipeline.
Supports CLI arguments (--input-dirs, --output-dir, --full-ingest, --write-reports).
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig
from swarmguard_pipeline.run_pipeline import execute_pipeline
from swarmguard_pipeline.run_pipeline import main

if __name__ == "__main__":
    config = PipelineConfig()
    execute_pipeline(config)
    main()
