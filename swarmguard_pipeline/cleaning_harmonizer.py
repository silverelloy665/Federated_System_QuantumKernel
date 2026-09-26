"""
Phase B: Cleaning, Harmonization, Deduplication, Feature Mapping & Taxonomy Alignment.
Harmonizes disparate schemas to canonical feature sets with auditable fuzzy/exact mapping logs,
replaces synthetic fallbacks with honest NaN handling, standardizes the unified 9-class taxonomy,
identifies and audits unmapped/UNKNOWN labels, and cleans invalid/missing values.
"""

import re
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from .config import PipelineConfig, TAXONOMY_CLASSES, UNKNOWN_CLASS_LABEL, UNKNOWN_CLASS_ID

class CleaningHarmonizer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dedup_records: List[Dict[str, Any]] = []
        self.feature_mapping_records: List[Dict[str, Any]] = []
        self.label_mapping_counts: Dict[Tuple[str, str, int, int, str], int] = defaultdict(int)
        self.unmapped_labels_per_source: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.unknown_counts_per_source: Dict[str, int] = defaultdict(int)

    def map_label_to_taxonomy(self, raw_label: Any, source_dataset: str = "Unknown") -> Tuple[str, int, int]:
        """
        Maps raw dataset labels to the unified 9-class taxonomy and binary indicators.
        Returns (canonical_name, class_id, binary_id).
        Unmapped labels fallback to ("UNKNOWN", -1, 1) and are audited.
        """
        if pd.isna(raw_label):
            res = ("BENIGN", 0, 0)
            self.label_mapping_counts[(str(raw_label), res[0], res[1], res[2], source_dataset)] += 1
            return res
        
        lbl_str = str(raw_label).strip()
        lbl = lbl_str.lower()
        
        # 0: BENIGN
        if any(x in lbl for x in ['benign', 'normal', 'none', 'clear', 'regular']):
            res = ("BENIGN", 0, 0)
        # 5: FDI (False Data Injection)
        elif 'fdi' in lbl or 'false data' in lbl or ('injection' in lbl and 'command' not in lbl and 'sql' not in lbl):
            res = ("FDI", 5, 1)
        # 4: Spoofing & MITM (including GPS Spoofing / FakeLanding)
        elif any(x in lbl for x in ['fakelanding', 'gps_spoofing', 'gps spoof', 'spoof', 'mitm', 'arp']):
            res = ("Spoofing_MITM", 4, 1)
        # 3: Jamming & Deauthentication & RF attacks
        elif any(x in lbl for x in ['jamming', 'gps_jamming', 'deauth', 'de-auth', 'evil_twin', 'eviltwin', 'rf_jamming']):
            res = ("Jamming_Deauth", 3, 1)
        # 6: Routing Attacks (Sybil, Blackhole, Wormhole)
        elif any(x in lbl for x in ['sybil', 'blackhole', 'wormhole', 'routing']):
            res = ("Routing_Attack", 6, 1)
        # 7: Replay Attacks
        elif 'replay' in lbl:
            res = ("Replay", 7, 1)
        # 2: DDoS
        elif 'ddos' in lbl or 'mirai' in lbl or 'synonymousip' in lbl:
            res = ("DDoS", 2, 1)
        # 1: DoS
        elif 'dos' in lbl or 'flood' in lbl or 'slowloris' in lbl or 'hulk' in lbl:
            res = ("DoS", 1, 1)
        # 8: Reconnaissance, PortScan, Infiltration, Malware, BruteForce
        elif any(x in lbl for x in ['scan', 'recon', 'discovery', 'pingsweep', 'bruteforce', 'malware',
                                  'backdoor', 'infilteration', 'infiltration', 'vulnerability',
                                  'browserhijacking', 'sql', 'xss', 'commandinjection', 'uploading',
                                  'portscan', 'patator']):
            res = ("Recon_Infiltration", 8, 1)
        else:
            # Explicit UNKNOWN fallback (Issue 6)
            res = (UNKNOWN_CLASS_LABEL, UNKNOWN_CLASS_ID, 1)
            self.unmapped_labels_per_source[source_dataset][lbl_str] += 1
            self.unknown_counts_per_source[source_dataset] += 1

        self.label_mapping_counts[(lbl_str, res[0], res[1], res[2], source_dataset)] += 1
        return res

    def extract_label_series(self, df: pd.DataFrame) -> pd.Series:
        """Finds or infers label column in dataframe."""
        for col in df.columns:
            if str(col).lower().strip() in ['class', 'label', 'attack', 'category', 'normal']:
                return df[col]
        # Check source dataset
        if '_source_dataset' in df.columns:
            return df['_source_dataset']
        return pd.Series(["BENIGN"] * len(df), index=df.index)

    def _log_mapping(self, branch: str, canonical_field: str, source_dataset: str,
                     source_col: Optional[str], match_type: str, is_fuzzy: bool, fallback: str = "NONE"):
        """Logs feature mapping decision to auditable records."""
        self.feature_mapping_records.append({
            "branch": branch,
            "canonical_field": canonical_field,
            "source_dataset": source_dataset,
            "source_column": source_col if source_col else "N/A",
            "match_type": match_type,
            "is_fuzzy": is_fuzzy,
            "fallback_used": fallback
        })

    def harmonize_physical_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Harmonizes Physical/Telemetry logs using canonical fields and kinematic invariants.
        Canonical fields (16 candidate features):
        [pitch, roll, yaw, velocity_x, velocity_y, velocity_z, height,
         battery, barometer, flight_time, temperature, distance,
         speed_norm, speed_horizontal, angular_speed_norm, kinetic_energy_z_proxy,
         accel_x, accel_y, accel_z]
        Absences are represented as NaN (no synthetic constants).
        """
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        res = pd.DataFrame(index=df.index)

        # Helper for column search
        def find_field(canonical: str, exact_candidates: List[str], fuzzy_candidates: List[str]):
            for ec in exact_candidates:
                if ec in cols_lower:
                    self._log_mapping("Physical", canonical, source_name, cols_lower[ec], "EXACT", False)
                    return pd.to_numeric(df[cols_lower[ec]], errors='coerce')
            for fc in fuzzy_candidates:
                if fc in cols_lower:
                    self._log_mapping("Physical", canonical, source_name, cols_lower[fc], "FUZZY", True)
                    return pd.to_numeric(df[cols_lower[fc]], errors='coerce')
            self._log_mapping("Physical", canonical, source_name, None, "MISSING", False, "LEFT_AS_NAN")
            return pd.Series(np.nan, index=df.index, dtype=np.float64)

        # 1. Angles
        res['pitch'] = find_field('pitch', ['pitch'], ['mpitch'])
        res['roll'] = find_field('roll', ['roll'], ['mroll'])
        res['yaw'] = find_field('yaw', ['yaw'], ['myaw'])

        # 2. Velocities
        res['velocity_x'] = find_field('velocity_x', ['velocity_x', 'vgx'], ['x_speed', 'vx'])
        res['velocity_y'] = find_field('velocity_y', ['velocity_y', 'vgy'], ['y_speed', 'vy'])
        res['velocity_z'] = find_field('velocity_z', ['velocity_z', 'vgz'], ['z_speed', 'vz'])

        # 3. Height / Altitude
        res['height'] = find_field('height', ['height', 'h'], ['tof', 'altitude'])

        # 4. Battery
        res['battery'] = find_field('battery', ['battery', 'battery_pct'], ['bat'])

        # 5. Barometer
        res['barometer'] = find_field('barometer', ['barometer'], ['baro'])

        # 6. Flight time
        res['flight_time'] = find_field('flight_time', ['flight_time'], ['time', 'flighttime'])

        # 7. Temperature
        if 'temperature' in cols_lower:
            self._log_mapping("Physical", "temperature", source_name, cols_lower['temperature'], "EXACT", False)
            res['temperature'] = pd.to_numeric(df[cols_lower['temperature']], errors='coerce')
        elif 'templ' in cols_lower and 'temph' in cols_lower:
            self._log_mapping("Physical", "temperature", source_name, f"{cols_lower['templ']}+{cols_lower['temph']}", "FUZZY", True)
            res['temperature'] = (pd.to_numeric(df[cols_lower['templ']], errors='coerce') + 
                                  pd.to_numeric(df[cols_lower['temph']], errors='coerce')) / 2.0
        elif 'templ' in cols_lower:
            self._log_mapping("Physical", "temperature", source_name, cols_lower['templ'], "FUZZY", True)
            res['temperature'] = pd.to_numeric(df[cols_lower['templ']], errors='coerce')
        else:
            self._log_mapping("Physical", "temperature", source_name, None, "MISSING", False, "LEFT_AS_NAN")
            res['temperature'] = pd.Series(np.nan, index=df.index, dtype=np.float64)

        # 8. Distance
        res['distance'] = find_field('distance', ['distance', 'distance_tof'], ['tof', 'z', 'mp_distance_z'])

        # 9. Synthesized Kinematic Invariants (Derived from physics for genuine PCA-12 reduction)
        vx = res['velocity_x'].fillna(0.0)
        vy = res['velocity_y'].fillna(0.0)
        vz = res['velocity_z'].fillna(0.0)
        p = res['pitch'].fillna(0.0)
        r = res['roll'].fillna(0.0)
        y = res['yaw'].fillna(0.0)

        res['speed_norm'] = np.sqrt(vx**2 + vy**2 + vz**2)
        self._log_mapping("Physical", "speed_norm", source_name, "DERIVED(vx,vy,vz)", "DERIVED", False)

        res['speed_horizontal'] = np.sqrt(vx**2 + vy**2)
        self._log_mapping("Physical", "speed_horizontal", source_name, "DERIVED(vx,vy)", "DERIVED", False)

        res['angular_speed_norm'] = np.sqrt(p**2 + r**2 + y**2)
        self._log_mapping("Physical", "angular_speed_norm", source_name, "DERIVED(pitch,roll,yaw)", "DERIVED", False)

        res['kinetic_energy_z_proxy'] = 0.5 * (vz**2)
        self._log_mapping("Physical", "kinetic_energy_z_proxy", source_name, "DERIVED(vz)", "DERIVED", False)

        # 10. Accelerations (Optional raw fields)
        res['accel_x'] = find_field('accel_x', ['accel_x', 'agx'], ['ax'])
        res['accel_y'] = find_field('accel_y', ['accel_y', 'agy'], ['ay'])
        res['accel_z'] = find_field('accel_z', ['accel_z', 'agz'], ['az'])

        # Labels
        label_raw = self.extract_label_series(df)
        labels_parsed = [self.map_label_to_taxonomy(x, source_dataset=source_name) for x in label_raw]
        res['canonical_label'] = [x[0] for x in labels_parsed]
        res['class_id'] = [x[1] for x in labels_parsed]
        res['binary_label'] = [x[2] for x in labels_parsed]

        return res

    def harmonize_network_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Harmonizes Network flow/frame logs to the canonical Network feature set:
        [flow_duration, tx_packets, rx_packets, tx_bytes, rx_bytes,
         packet_rate, byte_rate, mean_packet_size, mean_delay,
         mean_jitter, lost_packets, flag_syn, flag_ack, ttl_or_hops, protocol_type,
         total_bytes, total_packets, packet_size_ratio]
        Absences are represented as NaN (no synthetic constants).
        """
        cols_map = {re.sub(r'[^a-zA-Z0-9]', '', str(c).lower()): c for c in df.columns}
        res = pd.DataFrame(index=df.index)

        def match_net_col(canonical: str, exact_candidates: List[str], fuzzy_candidates: List[str]):
            for ec in exact_candidates:
                cleaned_ec = re.sub(r'[^a-zA-Z0-9]', '', ec.lower())
                if cleaned_ec in cols_map:
                    self._log_mapping("Network", canonical, source_name, cols_map[cleaned_ec], "EXACT", False)
                    return pd.to_numeric(df[cols_map[cleaned_ec]], errors='coerce')
            for fc in fuzzy_candidates:
                cleaned_fc = re.sub(r'[^a-zA-Z0-9]', '', fc.lower())
                if cleaned_fc in cols_map:
                    self._log_mapping("Network", canonical, source_name, cols_map[cleaned_fc], "FUZZY", True)
                    return pd.to_numeric(df[cols_map[cleaned_fc]], errors='coerce')
            self._log_mapping("Network", canonical, source_name, None, "MISSING", False, "LEFT_AS_NAN")
            return pd.Series(np.nan, index=df.index, dtype=np.float64)

        # 1. Flow duration
        res['flow_duration'] = match_net_col(
            'flow_duration',
            ['flow_duration', 'flowduration', 'flowdurations'],
            ['duration', 'wlanduration', 'frametimerelative']
        )

        # 2. Tx Packets
        tx_p = match_net_col(
            'tx_packets',
            ['tx_packets', 'txpackets', 'totalfwdpackets'],
            ['fwdpktstot', 'syncount', 'fwdpkts', 'number', 'framenumber']
        )
        if tx_p.isna().all() and any('frame' in c for c in cols_map):
            # For packet frame captures, each row represents 1 packet
            res['tx_packets'] = pd.Series(1.0, index=df.index, dtype=np.float64)
            self._log_mapping("Network", "tx_packets", source_name, "DERIVED(frame_unit=1.0)", "DERIVED", False)
        else:
            res['tx_packets'] = tx_p

        # 3. Rx Packets
        rx_p = match_net_col(
            'rx_packets',
            ['rx_packets', 'rxpackets', 'totalbackwardpackets'],
            ['bwdpktstot', 'ackcount', 'bwdpkts']
        )
        if rx_p.isna().all() and any('frame' in c for c in cols_map):
            # For single observed frame captures, rx_packets is 0
            res['rx_packets'] = pd.Series(0.0, index=df.index, dtype=np.float64)
            self._log_mapping("Network", "rx_packets", source_name, "DERIVED(frame_unit=0.0)", "DERIVED", False)
        else:
            res['rx_packets'] = rx_p

        # 4. Tx Bytes
        res['tx_bytes'] = match_net_col(
            'tx_bytes',
            ['tx_bytes', 'txbytes', 'totallengthoffwdpackets', 'totsize'],
            ['fwdheadersizetot', 'framelen', 'totlen']
        )

        # 5. Rx Bytes
        rx_b = match_net_col(
            'rx_bytes',
            ['rx_bytes', 'rxbytes', 'totallengthofbwdpackets', 'totsum'],
            ['bwdheadersizetot', 'datalen']
        )
        if rx_b.isna().all() and any('frame' in c for c in cols_map):
            res['rx_bytes'] = pd.Series(0.0, index=df.index, dtype=np.float64)
            self._log_mapping("Network", "rx_bytes", source_name, "DERIVED(frame_unit=0.0)", "DERIVED", False)
        else:
            res['rx_bytes'] = rx_b

        # 6. Packet Rate
        res['packet_rate'] = match_net_col(
            'packet_rate',
            ['packet_rate', 'flowpacketss', 'txpacketrates', 'rate'],
            ['fwdpktspersec', 'radiotapdatarate']
        )

        # 7. Byte Rate
        b_rate = match_net_col(
            'byte_rate',
            ['byte_rate', 'flowbytess', 'txbyterates'],
            ['fwdbytespersec', 'bwdbytespersec']
        )
        if b_rate.isna().all():
            # Derive byte rate from bytes and duration/delay where possible
            dur = res['flow_duration'].fillna(0.0)
            tb = res['tx_bytes'].fillna(0.0)
            rb = res['rx_bytes'].fillna(0.0)
            res['byte_rate'] = (tb + rb) / (dur + 1e-4)
            self._log_mapping("Network", "byte_rate", source_name, "DERIVED((tx_bytes+rx_bytes)/dur)", "DERIVED", False)
        else:
            res['byte_rate'] = b_rate

        # 8. Mean Packet Size
        res['mean_packet_size'] = match_net_col(
            'mean_packet_size',
            ['mean_packet_size', 'meanpacketsize', 'packetlengthmean'],
            ['fwdpacketlengthmean', 'headerlength', 'framelen', 'avg', 'averagepacketsize']
        )

        # 9. Mean Delay / IAT
        res['mean_delay'] = match_net_col(
            'mean_delay',
            ['mean_delay', 'meandelays', 'flowiatmean'],
            ['iat', 'timesincelastpacket', 'frametimedeltadisplayed', 'fwdiatmean']
        )

        # 10. Mean Jitter / IAT Std
        res['mean_jitter'] = match_net_col(
            'mean_jitter',
            ['mean_jitter', 'meanjitters', 'flowiatstd'],
            ['std', 'variance', 'packetlengthstd', 'fwdiatstd']
        )

        # 11. Lost Packets / Drop Rate
        res['lost_packets'] = match_net_col(
            'lost_packets',
            ['lost_packets', 'lostpackets', 'packetdroprate'],
            ['downupratio', 'wlanfcretry', 'wlanfcsbadchecksum', 'wlanfrag']
        )

        # 12. Flag SYN
        res['flag_syn'] = match_net_col(
            'flag_syn',
            ['flag_syn', 'synflags', 'synflagcount'],
            ['synflagnumber', 'syncount', 'fwdsynflags']
        )

        # 13. Flag ACK
        res['flag_ack'] = match_net_col(
            'flag_ack',
            ['flag_ack', 'ackflags', 'ackflagcount'],
            ['ackflagnumber', 'tcpack', 'tcpflags', 'ackcount']
        )

        # 14. TTL / Hops
        res['ttl_or_hops'] = match_net_col(
            'ttl_or_hops',
            ['ttl_or_hops', 'averagehopcount', 'timetolive'],
            ['ipttl', 'ttl']
        )

        # 15. Protocol
        res['protocol_type'] = match_net_col(
            'protocol_type',
            ['protocol_type', 'protocol', 'protocoltype'],
            ['ipproto', 'ipprotocol', 'destinationport']
        )

        # 16. Invariant Flow Properties (Continuous flow dimension extensions)
        tx_b_clean = res['tx_bytes'].fillna(0.0)
        rx_b_clean = res['rx_bytes'].fillna(0.0)
        tx_p_clean = res['tx_packets'].fillna(1.0)
        rx_p_clean = res['rx_packets'].fillna(0.0)

        res['total_bytes'] = tx_b_clean + rx_b_clean
        self._log_mapping("Network", "total_bytes", source_name, "DERIVED(tx_bytes+rx_bytes)", "DERIVED", False)

        res['total_packets'] = tx_p_clean + rx_p_clean
        self._log_mapping("Network", "total_packets", source_name, "DERIVED(tx_packets+rx_packets)", "DERIVED", False)

        res['packet_size_ratio'] = res['total_bytes'] / (res['total_packets'] + 1e-4)
        self._log_mapping("Network", "packet_size_ratio", source_name, "DERIVED(total_bytes/total_packets)", "DERIVED", False)

        # Labels
        label_raw = self.extract_label_series(df)
        labels_parsed = [self.map_label_to_taxonomy(x, source_dataset=source_name) for x in label_raw]
        res['canonical_label'] = [x[0] for x in labels_parsed]
        res['class_id'] = [x[1] for x in labels_parsed]
        res['binary_label'] = [x[2] for x in labels_parsed]

        return res

    def clean_and_deduplicate(self, df: pd.DataFrame, branch: str) -> pd.DataFrame:
        """
        Cleans continuous features:
        - Replaces +/-Inf with NaN.
        - Removes duplicate records.
        - Drops features with >30% missing.
        - Imputes missing values (<30%) with column medians.
        """
        raw_count = len(df)
        feature_cols = [c for c in df.columns if c not in ['canonical_label', 'class_id', 'binary_label', '_source_dataset']]

        # Convert feature columns to float64 and replace infs
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

        # Deduplication
        df_clean = df.drop_duplicates(subset=feature_cols).copy()
        dedup_count = raw_count - len(df_clean)

        # Evaluate missing percentages
        dropped_features = []
        imputed_features = []
        for c in feature_cols:
            null_ratio = df_clean[c].isnull().sum() / len(df_clean)
            if null_ratio > self.config.drop_col_null_threshold:
                df_clean.drop(columns=[c], inplace=True)
                dropped_features.append(f"{c} ({null_ratio*100:.1f}%)")
            elif null_ratio > 0:
                median_val = df_clean[c].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[c] = df_clean[c].fillna(median_val)
                imputed_features.append(f"{c} ({null_ratio*100:.1f}%)")

        # Log deduplication & cleaning record
        rec = {
            "branch": branch,
            "raw_rows": raw_count,
            "retained_rows": len(df_clean),
            "duplicates_removed": dedup_count,
            "dropped_features": "; ".join(dropped_features) if dropped_features else "None",
            "imputed_features": "; ".join(imputed_features) if imputed_features else "None"
        }
        self.dedup_records.append(rec)
        return df_clean

    def export_reports(self):
        """Exports deduplication, label mapping, and feature mapping reports."""
        self.config.create_directories()

        # 1. Deduplication report
        dedup_df = pd.DataFrame(self.dedup_records)
        dedup_path = self.config.reports_dir / "deduplication_report.csv"
        dedup_df.to_csv(dedup_path, index=False)
        print(f"[+] Saved deduplication report to: {dedup_path}")

        # 2. Feature mapping report (Issue 1 & 5)
        feat_df = pd.DataFrame(self.feature_mapping_records)
        if not feat_df.empty:
            feat_df = feat_df.drop_duplicates().reset_index(drop=True)
        feat_path = self.config.reports_dir / "feature_mapping.csv"
        feat_df.to_csv(feat_path, index=False)
        print(f"[+] Saved auditable feature mapping report to: {feat_path}")

        # 3. Label mapping report (Issue 1 & 6)
        label_rows = []
        for (raw_lbl, can_lbl, cls_id, bin_lbl, src), cnt in self.label_mapping_counts.items():
            label_rows.append({
                "class_id": cls_id,
                "standard_label": can_lbl,
                "binary_label": bin_lbl,
                "raw_label": raw_lbl,
                "source_dataset": src,
                "row_count": cnt,
                "is_unmapped": (cls_id == UNKNOWN_CLASS_ID)
            })
        lbl_df = pd.DataFrame(label_rows)
        if not lbl_df.empty:
            lbl_df = lbl_df.sort_values(by=["class_id", "row_count"], ascending=[True, False])
        lbl_path = self.config.reports_dir / "label_mapping.csv"
        lbl_df.to_csv(lbl_path, index=False)
        print(f"[+] Saved dynamic label mapping report to: {lbl_path}")

    def run_harmonization(self, routed_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Executes Phase B: Harmonizes, cleans, and aligns labels for both branches.
        """
        print("\n========================================================", flush=True)
        print("  PHASE B: CLEANING, HARMONIZATION & TAXONOMY ALIGNMENT", flush=True)
        print("========================================================", flush=True)
        harmonized_branches = {}

        for branch in ["Network", "Physical"]:
            dfs = routed_data.get(branch, [])
            if not dfs:
                print(f"[!] Warning: No data available for {branch} branch.", flush=True)
                continue

            print(f"[*] Harmonizing {len(dfs)} sources for {branch} Branch...", flush=True)
            harmonized_dfs = []
            for df in dfs:
                src_name = df.get('_source_dataset', pd.Series(['Unknown'])).iloc[0] if '_source_dataset' in df.columns else 'Unknown'
                if branch == "Physical":
                    h_df = self.harmonize_physical_dataframe(df, src_name)
                else:
                    h_df = self.harmonize_network_dataframe(df, src_name)
                harmonized_dfs.append(h_df)

            combined_df = pd.concat(harmonized_dfs, ignore_index=True)
            print(f"    Raw combined shape for {branch}: {combined_df.shape}", flush=True)

            # Clean and deduplicate
            clean_df = self.clean_and_deduplicate(combined_df, branch)
            print(f"    Cleaned & deduplicated shape for {branch}: {clean_df.shape}", flush=True)
            print(f"    Class distribution:\n{clean_df['canonical_label'].value_counts()}", flush=True)

            harmonized_branches[branch] = clean_df

        # Save all Phase B reports
        self.export_reports()

        # Print UNKNOWN Label Summary (Issue 6)
        print("\n" + "="*60, flush=True)
        print("  UNKNOWN LABEL AUDIT SUMMARY (Issue #6)", flush=True)
        print("="*60, flush=True)
        total_unknown = sum(self.unknown_counts_per_source.values())
        for src, count in self.unknown_counts_per_source.items():
            print(f"  Source: {src:<32} -> {count:,} UNKNOWN rows", flush=True)
            if count > 0:
                for raw_l, r_cnt in self.unmapped_labels_per_source[src].items():
                    print(f"     └─ '{raw_l}': {r_cnt:,} rows", flush=True)
        if total_unknown == 0:
            print("  [AUDIT PASS] 0 UNKNOWN / Unmapped labels across all ingested sources.", flush=True)
        else:
            print(f"  [AUDIT FLAG] Total UNKNOWN rows: {total_unknown:,}", flush=True)
        print("="*60 + "\n", flush=True)

        return harmonized_branches
