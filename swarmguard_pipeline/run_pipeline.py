"""
Master execution script for SwarmGuard Preprocessing Pipeline.
Executes Phase A -> Phase B -> Phase C -> Phase D -> Reporting.
"""

from .config import PipelineConfig
from .discovery_router import DiscoveryRouter
from .cleaning_harmonizer import CleaningHarmonizer
from .federated_splitter import FederatedSplitter
from .quantum_formatter import QuantumFormatter
from .reporter import SwarmGuardReporter

def execute_pipeline(config: PipelineConfig = None):
    if config is None:
        config = PipelineConfig()

    print("\n" + "="*70)
    print("      LAUNCHING SWARMGUARD FEDERATED QML PREPROCESSING PIPELINE")
    print("="*70)
    print(f"[*] Output Directory: {config.output_dir}")

    # Phase A: Discovery & Routing
    router = DiscoveryRouter(config)
    inventory_records, routed_data = router.run_discovery()

    # Phase B: Cleaning, Harmonization & Taxonomy
    harmonizer = CleaningHarmonizer(config)
    harmonized_branches = harmonizer.run_harmonization(routed_data)

    # Phase C: Centralized & Federated Splitting
    splitter = FederatedSplitter(config)
    split_results = splitter.run_splitting(harmonized_branches)

    # Phase D: 12-Qubit QML Formatting & Quantum Scaling
    formatter = QuantumFormatter(config)
    processed_outputs = formatter.run_quantum_formatting(split_results)

    # Reporting
    reporter = SwarmGuardReporter(config)
    html_path = reporter.generate_html_report(
        inventory_records,
        harmonizer.dedup_records,
        split_results,
        processed_outputs
    )
    reporter.print_terminal_summary(processed_outputs)

    print(f"[SUCCESS] All pipeline artifacts generated at: {config.output_dir}")
    print(f"[SUCCESS] View interactive report at: {html_path}\n")

    return processed_outputs

if __name__ == "__main__":
    execute_pipeline()
