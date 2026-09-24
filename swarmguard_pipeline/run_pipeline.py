"""
Master execution script for SwarmGuard Preprocessing Pipeline.
Executes Phase A -> Phase B -> Phase C -> Phase D -> Reporting.
Supports CLI arguments for input/output paths, full ingestion, and automated report regeneration.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, TAXONOMY_CLASSES
from .discovery_router import DiscoveryRouter
from .cleaning_harmonizer import CleaningHarmonizer
from .federated_splitter import FederatedSplitter
from .quantum_formatter import QuantumFormatter
from .reporter import SwarmGuardReporter

def sync_readme_documentation(config: PipelineConfig):
    """
    Synchronizes README.md documentation with the live 9-class taxonomy and architecture.
    """
    readme_path = Path("./README.md")
    if not readme_path.exists():
        candidate = config.output_dir.parent / "README.md"
        if candidate.exists():
            readme_path = candidate

    if not readme_path.exists():
        return

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Build dynamic taxonomy table
        descriptions = {
            "BENIGN": ("0 (BENIGN)", "Normal flight telemetry & benign network traffic"),
            "DoS": ("1 (ATTACK)", "Denial of service, SYN/UDP floods, slowloris"),
            "DDoS": ("1 (ATTACK)", "Distributed denial of service, botnet/mirai floods"),
            "Jamming_Deauth": ("1 (ATTACK)", "RF jamming, 802.11 de-authentication frames, Evil Twin"),
            "Spoofing_MITM": ("1 (ATTACK)", "FakeLanding, GPS spoofing, ARP cache poisoning, MITM"),
            "FDI": ("1 (ATTACK)", "False data injection into sensors and control telemetry"),
            "Routing_Attack": ("1 (ATTACK)", "Sybil, Blackhole, Wormhole, routing loops"),
            "Replay": ("1 (ATTACK)", "Replayed telemetry commands and network packets"),
            "Recon_Infiltration": ("1 (ATTACK)", "Port scans, ping sweep, OS fingerprinting, malware/backdoor")
        }

        tax_rows = []
        for idx, cls_name in enumerate(TAXONOMY_CLASSES):
            bin_lbl, desc = descriptions.get(cls_name, ("1 (ATTACK)", "UAV intrusion attack pattern"))
            tax_rows.append(f"| `{idx}` | **{cls_name}** | `{bin_lbl}` | {desc} |")

        tax_table_md = "| Class ID | Standard Label | Binary Label | Description & Sub-Types |\n|:---:|:---|:---:|:---|\n" + "\n".join(tax_rows)

        # Replace taxonomy table section
        import re
        content = re.sub(
            r"### 3\. Unified 9-Class UAV Intrusion Taxonomy\s*\n\s*\|.*?(?=\n\s*###|\n\s*---|\Z)",
            f"### 3. Unified 9-Class UAV Intrusion Taxonomy\n\n{tax_table_md}\n\n",
            content,
            flags=re.DOTALL
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Synchronized live taxonomy documentation into: {readme_path}")
    except Exception as e:
        print(f"[!] Warning: Could not auto-sync README.md ({e})")

def execute_pipeline(config: Optional[PipelineConfig] = None):
    if config is None:
        config = PipelineConfig()

    print("\n" + "="*70)
    print("      LAUNCHING SWARMGUARD FEDERATED QML PREPROCESSING PIPELINE")
    print("="*70)
    print(f"[*] Input Search Dirs : {[str(p) for p in config.input_search_dirs]}")
    print(f"[*] Output Directory  : {config.output_dir}")
    print(f"[*] Full Ingestion    : {config.full_ingest}")

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
        processed_outputs,
        formatter.skewness_records
    )
    reporter.print_terminal_summary(processed_outputs)

    # Auto-sync documentation and reports
    if config.write_reports:
        sync_readme_documentation(config)

    print(f"[SUCCESS] All pipeline artifacts generated at: {config.output_dir}")
    print(f"[SUCCESS] View interactive report at: {html_path}\n")

    return processed_outputs

def main():
    parser = argparse.ArgumentParser(description="SwarmGuard Dual-Branch QML Preprocessing Pipeline")
    parser.add_argument(
        "--input-dirs",
        type=str,
        default=None,
        help="Semicolon or comma-separated list of directories to search for raw datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target base directory for processed artifacts and reports"
    )
    parser.add_argument(
        "--full-ingest",
        action="store_true",
        help="Disable row and file sampling caps for complete from-scratch dataset ingestion"
    )
    parser.add_argument(
        "--write-reports",
        action="store_true",
        default=True,
        help="Dump/update live config, mapping state to reports/ and synchronize README.md"
    )

    args = parser.parse_args()

    config = PipelineConfig()
    if args.input_dirs:
        sep = ";" if ";" in args.input_dirs else ","
        config.input_search_dirs = [Path(p.strip()) for p in args.input_dirs.split(sep) if p.strip()]
    if args.output_dir:
        config.output_dir = Path(args.output_dir.strip())
    config.full_ingest = args.full_ingest
    config.write_reports = args.write_reports

    execute_pipeline(config)

if __name__ == "__main__":
    main()
