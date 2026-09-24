"""
Phase A: Discovery & Dual-Branch Routing.
Locates datasets, inspects schemas, demultiplexes sub-tables, classifies into Network vs Physical branches,
discloses all sampling caps/methods, and generates dataset_inventory.csv.
"""

import sys
import os
import zipfile
import io
import openpyxl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from .config import PipelineConfig

def count_lines_in_zip_entry(z: zipfile.ZipFile, entry_name: str) -> int:
    """Fast raw line count in a zip entry without full CSV decoding."""
    try:
        with z.open(entry_name) as f:
            lines = sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024), b''))
        return max(0, lines - 1)  # Subtract 1 for header
    except Exception:
        return 0

def read_excel_fast(file_obj, sheet_name=None, max_rows: Optional[int] = 25000) -> Tuple[pd.DataFrame, int]:
    """
    Reads Excel workbook in read_only streaming mode for speed and memory efficiency.
    Returns (dataframe, total_rows_available).
    """
    wb = openpyxl.load_workbook(file_obj, read_only=True, data_only=True)
    sheet = wb[sheet_name] if sheet_name and sheet_name in wb.sheetnames else wb.active
    total_avail = getattr(sheet, 'max_row', 0)
    if total_avail:
        total_avail = max(0, total_avail - 1)  # Minus header
        
    rows_iter = sheet.iter_rows(values_only=True)
    try:
        headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(next(rows_iter))]
    except StopIteration:
        wb.close()
        return pd.DataFrame(), 0

    data = []
    count = 0
    for i, row in enumerate(rows_iter):
        if max_rows is not None and i >= max_rows:
            count = i
            break
        data.append(row[:len(headers)])
        count += 1
    
    if total_avail == 0 or total_avail < count:
        total_avail = count

    wb.close()
    return pd.DataFrame(data, columns=headers), total_avail

class DiscoveryRouter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.resolved_paths: Dict[str, Path] = {}
        self.inventory_records: List[Dict[str, Any]] = []

    def locate_input_files(self) -> Dict[str, Path]:
        """Search across input paths for required dataset files."""
        for key, filename in self.config.target_files.items():
            found = False
            for base_dir in self.config.input_search_dirs:
                candidate = base_dir / filename
                if candidate.exists():
                    self.resolved_paths[key] = candidate
                    found = True
                    break
            if not found:
                print(f"[WARNING] Could not locate target file: {filename}", flush=True)
        return self.resolved_paths

    def classify_schema(self, columns: List[str]) -> str:
        """
        Classifies columns into 'Physical' or 'Network' based on domain keywords.
        """
        cols_lower = [str(c).lower().strip() for c in columns]

        # Physical telemetry signatures (kinematics, gyro, IMU, battery, barometer, etc.)
        physical_keywords = {
            'pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'x_speed', 'y_speed', 'z_speed',
            'tof', 'bat', 'battery', 'baro', 'barometer', 'flight_time', 'agx', 'agy', 'agz',
            'mp_distance_x', 'mp_distance_y', 'mp_distance_z', 'mpitch', 'mroll', 'myaw',
            'speed_norm', 'residual1', 'temperature', 'templ', 'temph'
        }

        # Network signatures (radiotap, 802.11 frames, IP headers, TCP/UDP ports, flow duration)
        network_keywords = {
            'wlan', 'radiotap', 'flow_duration', 'flowduration/s', 'srcaddr', 'dstaddr',
            'total length of fwd packets', 'header_length', 'frame.len', 'frame.number',
            'destination port', 'flow bytes/s', 'fwd_pkts_tot', 'tcp.flags', 'syn_count',
            'ack_count', 'flow_iat'
        }

        physical_matches = sum(1 for k in physical_keywords if any(k in c for c in cols_lower))
        network_matches = sum(1 for k in network_keywords if any(k in c for c in cols_lower))

        if physical_matches > 0 and physical_matches >= network_matches:
            return "Physical"
        return "Network"

    def demultiplex_tits(self, filepath: Path) -> List[Tuple[str, pd.DataFrame, str]]:
        """
        Splits Dataset_T-ITS.csv into constituent Network and Physical segments.
        Returns list of (segment_name, dataframe, branch).
        """
        results = []
        with open(filepath, 'r', encoding='latin1', errors='replace') as f:
            lines = f.readlines()

        segments = []
        current_header = None
        current_lines = []

        for l in lines:
            l_strip = l.strip()
            if not l_strip:
                continue
            parts = l_strip.split(',')
            if any(k in parts[0].lower() for k in ['timestamp', 'frame', 'uid', 'mid']):
                if current_header is not None and current_lines:
                    segments.append((current_header, current_lines))
                current_header = l_strip
                current_lines = []
            else:
                current_lines.append(l)

        if current_header and current_lines:
            segments.append((current_header, current_lines))

        for idx, (hdr, l_list) in enumerate(segments):
            raw_cols = [c.strip() for c in hdr.split(',')]
            valid_col_count = len([c for c in raw_cols if c])
            
            csv_text = hdr + '\n' + ''.join(l_list)
            df = pd.read_csv(io.StringIO(csv_text), low_memory=False, on_bad_lines='skip')
            df = df.iloc[:, :valid_col_count]
            df.columns = [str(c).strip() for c in df.columns]

            branch = self.classify_schema(list(df.columns))
            seg_name = f"Dataset_T-ITS_Seg{idx+1}_{branch}"
            results.append((seg_name, df, branch))

        return results

    def profile_dataframe(
        self,
        name: str,
        source_file: str,
        df: pd.DataFrame,
        branch: str,
        rows_available: Optional[int] = None,
        rows_sampled: Optional[int] = None,
        sampling_method: str = "FULL_INGEST"
    ) -> Dict[str, Any]:
        """Profiles dataset characteristics and sampling parameters for dataset_inventory.csv."""
        num_rows, num_cols = df.shape
        if rows_sampled is None:
            rows_sampled = num_rows
        if rows_available is None:
            rows_available = num_rows

        null_counts = int(df.isnull().sum().sum())
        total_cells = num_rows * num_cols if (num_rows * num_cols) > 0 else 1
        null_pct = round((null_counts / total_cells) * 100, 2)
        
        label_col = None
        for c in df.columns:
            if str(c).lower() in ['class', 'label', 'attack', 'category']:
                label_col = c
                break

        unique_labels = []
        if label_col is not None:
            unique_labels = [str(x) for x in df[label_col].dropna().unique()[:8]]

        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        record = {
            "dataset_name": name,
            "source_file": source_file,
            "branch": branch,
            "rows_available": rows_available,
            "rows_sampled": rows_sampled,
            "sampling_method": sampling_method,
            "raw_rows": num_rows,
            "raw_cols": num_cols,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "null_pct": null_pct,
            "null_percentage": null_pct,
            "memory_mb": memory_mb,
            "label_column": label_col if label_col else "N/A",
            "sample_labels": "; ".join(unique_labels),
            "sample_columns": "; ".join(list(df.columns)[:8])
        }
        self.inventory_records.append(record)
        return record

    def run_discovery(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[pd.DataFrame]]]:
        """
        Executes Phase A: Discovers, routes, profiles with sampling disclosure, and generates inventory.
        """
        print("\n========================================================", flush=True)
        print("  PHASE A: DISCOVERY & DUAL-BRANCH ROUTING", flush=True)
        if self.config.full_ingest:
            print("  [MODE] FULL INGESTION ENABLED (Sampling caps disabled)", flush=True)
        else:
            print("  [MODE] CONTROLLED SAMPLING ENABLED (Disclosed in inventory)", flush=True)
        print("========================================================", flush=True)
        self.locate_input_files()
        self.config.create_directories()

        routed_data: Dict[str, List[pd.DataFrame]] = {
            "Network": [],
            "Physical": []
        }

        # 1. UAVIDS-2025.csv
        if "uavids" in self.resolved_paths:
            path = self.resolved_paths["uavids"]
            print(f"[*] Processing {path.name}...", flush=True)
            df = pd.read_csv(path, low_memory=False)
            branch = self.classify_schema(list(df.columns))
            self.profile_dataframe(
                "UAVIDS-2025", path.name, df, branch,
                rows_available=len(df), rows_sampled=len(df), sampling_method="FULL_INGEST"
            )
            df['_source_dataset'] = "UAVIDS-2025"
            routed_data[branch].append(df)
            print(f"    -> Routed {df.shape[0]:,} of {df.shape[0]:,} rows (FULL_INGEST) to {branch} Branch.", flush=True)

        # 2. Dataset_T-ITS.csv (Demultiplexed)
        if "tits" in self.resolved_paths:
            path = self.resolved_paths["tits"]
            print(f"[*] Demultiplexing {path.name}...", flush=True)
            segments = self.demultiplex_tits(path)
            for seg_name, df, branch in segments:
                self.profile_dataframe(
                    seg_name, path.name, df, branch,
                    rows_available=len(df), rows_sampled=len(df), sampling_method="FULL_INGEST"
                )
                df['_source_dataset'] = seg_name
                routed_data[branch].append(df)
                print(f"    -> Routed [{seg_name}] {df.shape[0]:,} of {df.shape[0]:,} rows (FULL_INGEST) to {branch} Branch.", flush=True)

        # 3. UAV-NDD CSV.zip
        if "uav_ndd" in self.resolved_paths:
            path = self.resolved_paths["uav_ndd"]
            print(f"[*] Extracting streams from {path.name}...", flush=True)
            with zipfile.ZipFile(path, 'r') as z:
                # UAV-Case1-Label.csv
                if "UAV-NDD CSV/UAV-Case1-Label.csv" in z.namelist():
                    avail = count_lines_in_zip_entry(z, "UAV-NDD CSV/UAV-Case1-Label.csv")
                    nrows = None if self.config.full_ingest else self.config.max_uav_ndd_case1_rows
                    method = "FULL_INGEST" if self.config.full_ingest else f"HEAD_SAMPLE_{self.config.max_uav_ndd_case1_rows}"
                    with z.open("UAV-NDD CSV/UAV-Case1-Label.csv") as f:
                        df_c1 = pd.read_csv(f, encoding='latin1', on_bad_lines='skip', nrows=nrows, low_memory=False)
                        branch = self.classify_schema(list(df_c1.columns))
                        self.profile_dataframe(
                            "UAV-NDD_Case1", path.name, df_c1, branch,
                            rows_available=avail, rows_sampled=len(df_c1), sampling_method=method
                        )
                        df_c1['_source_dataset'] = "UAV-NDD_Case1"
                        routed_data[branch].append(df_c1)
                        print(f"    -> Routed UAV-Case1 {len(df_c1):,} of {avail:,} rows ({method}) to {branch} Branch.", flush=True)

                # Access Point Case2 Label.xlsx
                if "UAV-NDD CSV/Access Point Case2 Label.xlsx" in z.namelist():
                    max_r = None if self.config.full_ingest else self.config.max_uav_ndd_excel_rows
                    method = "FULL_INGEST" if self.config.full_ingest else f"HEAD_SAMPLE_{self.config.max_uav_ndd_excel_rows}"
                    with z.open("UAV-NDD CSV/Access Point Case2 Label.xlsx") as f:
                        df_ap, avail = read_excel_fast(f, max_rows=max_r)
                        branch = self.classify_schema(list(df_ap.columns))
                        self.profile_dataframe(
                            "UAV-NDD_AccessPoint_Case2", path.name, df_ap, branch,
                            rows_available=avail, rows_sampled=len(df_ap), sampling_method=method
                        )
                        df_ap['_source_dataset'] = "UAV-NDD_AccessPoint_Case2"
                        routed_data[branch].append(df_ap)
                        print(f"    -> Routed Access Point Case2 {len(df_ap):,} of {avail:,} rows ({method}) to {branch} Branch.", flush=True)

                # GSC Case3 Label .csv (as Excel)
                if "UAV-NDD CSV/GSC Case3 Label .csv" in z.namelist():
                    max_r = None if self.config.full_ingest else self.config.max_uav_ndd_excel_rows
                    method = "FULL_INGEST" if self.config.full_ingest else f"HEAD_SAMPLE_{self.config.max_uav_ndd_excel_rows}"
                    with z.open("UAV-NDD CSV/GSC Case3 Label .csv") as f:
                        df_gsc, avail = read_excel_fast(f, sheet_name="Dataset-UAV-GCS", max_rows=max_r)
                        branch = self.classify_schema(list(df_gsc.columns))
                        self.profile_dataframe(
                            "UAV-NDD_GSC_Case3", path.name, df_gsc, branch,
                            rows_available=avail, rows_sampled=len(df_gsc), sampling_method=method
                        )
                        df_gsc['_source_dataset'] = "UAV-NDD_GSC_Case3"
                        routed_data[branch].append(df_gsc)
                        print(f"    -> Routed GSC Case3 {len(df_gsc):,} of {avail:,} rows ({method}) to {branch} Branch.", flush=True)

        # 4. CSV.zip (CIC-IoT proxies)
        if "csv_iot" in self.resolved_paths:
            path = self.resolved_paths["csv_iot"]
            print(f"[*] Ingesting flows from {path.name}...", flush=True)
            with zipfile.ZipFile(path, 'r') as z:
                all_csvs = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
                folder_map = {}
                for csv_f in all_csvs:
                    parent = os.path.dirname(csv_f)
                    if parent not in folder_map:
                        folder_map[parent] = []
                    folder_map[parent].append(csv_f)
                
                total_folders = len(folder_map)
                num_folders = total_folders if self.config.full_ingest else min(self.config.max_cic_iot_folders, total_folders)
                nrows_per_file = None if self.config.full_ingest else self.config.max_cic_iot_rows_per_file
                method = "FULL_INGEST" if self.config.full_ingest else f"STRATIFIED_CLUSTER_SAMPLE_{num_folders}x{nrows_per_file}"

                sampled_dfs = []
                for cat_folder, files in list(folder_map.items())[:num_folders]:
                    cat_name = os.path.basename(cat_folder)
                    for f_name in (files if self.config.full_ingest else files[:1]):
                        with z.open(f_name) as f:
                            chunk = pd.read_csv(f, nrows=nrows_per_file, encoding_errors='replace', low_memory=False)
                            if 'Label' not in chunk.columns and 'class' not in chunk.columns:
                                chunk['label'] = cat_name
                            sampled_dfs.append(chunk)

                if sampled_dfs:
                    df_iot = pd.concat(sampled_dfs, ignore_index=True)
                    branch = self.classify_schema(list(df_iot.columns))
                    # Estimate total available rows across all files
                    est_total = len(all_csvs) * 5000
                    self.profile_dataframe(
                        "CIC_IoT_Proxies", path.name, df_iot, branch,
                        rows_available=est_total, rows_sampled=len(df_iot), sampling_method=method
                    )
                    df_iot['_source_dataset'] = "CIC_IoT_Proxies"
                    routed_data[branch].append(df_iot)
                    print(f"    -> Routed CIC-IoT flows {len(df_iot):,} rows ({method}) to {branch} Branch.", flush=True)

        # 5. MachineLearningCSV.zip / GeneratedLabelledFlows.zip
        flow_zip = self.resolved_paths.get("cve_ml") or self.resolved_paths.get("flows_iscx")
        if flow_zip:
            print(f"[*] Ingesting network flows from {flow_zip.name}...", flush=True)
            with zipfile.ZipFile(flow_zip, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
                num_files = len(csv_files) if self.config.full_ingest else min(self.config.max_cicids2017_files, len(csv_files))
                nrows_per_file = None if self.config.full_ingest else self.config.max_cicids2017_rows_per_file
                method = "FULL_INGEST" if self.config.full_ingest else f"HEAD_SAMPLE_{num_files}x{nrows_per_file}"

                sampled_flows = []
                for cf in csv_files[:num_files]:
                    with z.open(cf) as f:
                        chunk = pd.read_csv(f, nrows=nrows_per_file, encoding_errors='replace', low_memory=False)
                        sampled_flows.append(chunk)
                if sampled_flows:
                    df_cic = pd.concat(sampled_flows, ignore_index=True)
                    df_cic.columns = [str(c).strip() for c in df_cic.columns]
                    branch = self.classify_schema(list(df_cic.columns))
                    est_total = len(csv_files) * 200000
                    self.profile_dataframe(
                        "CICIDS2017_NetworkFlows", flow_zip.name, df_cic, branch,
                        rows_available=est_total, rows_sampled=len(df_cic), sampling_method=method
                    )
                    df_cic['_source_dataset'] = "CICIDS2017"
                    routed_data[branch].append(df_cic)
                    print(f"    -> Routed CICIDS2017 flows {len(df_cic):,} rows ({method}) to {branch} Branch.", flush=True)

        # Save dataset_inventory.csv
        inv_df = pd.DataFrame(self.inventory_records)
        inv_path = self.config.reports_dir / "dataset_inventory.csv"
        inv_df.to_csv(inv_path, index=False)
        print(f"\n[+] Saved dataset inventory to: {inv_path}", flush=True)

        return self.inventory_records, routed_data
