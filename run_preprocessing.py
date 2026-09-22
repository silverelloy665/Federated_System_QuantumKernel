"""
Root execution entrypoint for SwarmGuard Preprocessing Pipeline.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swarmguard_pipeline.config import PipelineConfig
from swarmguard_pipeline.run_pipeline import execute_pipeline

if __name__ == "__main__":
    config = PipelineConfig()
    execute_pipeline(config)
