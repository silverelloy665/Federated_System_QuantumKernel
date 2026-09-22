"""
Phase B: Cleaning, Harmonization, Deduplication & Label Alignment.
Harmonizes disparate schemas to canonical feature sets, standardizes 9-class attack taxonomy,
and cleans invalid/missing values with destructive logging.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path
from .config import PipelineConfig

class CleaningHarmonizer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dedup_records: List[Dict[str, Any]] = []

    def map_label_to_taxonomy(self, raw_label: Any) -> Tuple[str, int, int]:
        """
        Maps raw dataset labels to the 9-class taxonomy and binary indicators.
        Returns (canonical_name, class_id, binary_id).
        """
        if pd.isna(raw_label):
            return ("BENIGN", 0, 0)
        
        lbl = str(raw_label).strip().lower()
        
        # 0: BENIGN
        if any(x in lbl for x in ['benign', 'normal', 'none', 'clear']):
            return ("BENIGN", 0, 0)

        # 5: FDI (False Data Injection)
        if 'fdi' in lbl or 'false data' in lbl or 'injection' in lbl and 'command' not in lbl and 'sql' not in lbl:
            return ("FDI", 5, 1)

        # 4: Spoofing & MITM (including GPS Spoofing / FakeLanding)
        if any(x in lbl for x in ['fakelanding', 'gps', 'spoof', 'mitm', 'arp']):
            return ("Spoofing_MITM", 4, 1)

        # 3: Jamming & Deauthentication & RF attacks
        if any(x in lbl for x in ['jamming', 'deauth', 'de-auth', 'evil_twin', 'eviltwin']):
            return ("Jamming_Deauth", 3, 1)

        # 6: Routing Attacks (Sybil, Blackhole, Wormhole)
        if any(x in lbl for x in ['sybil', 'blackhole', 'wormhole', 'routing']):
            return ("Routing_Attack", 6, 1)

        # 7: Replay Attacks
        if 'replay' in lbl:
            return ("Replay", 7, 1)

        # 2: DDoS
        if 'ddos' in lbl or 'mirai' in lbl or 'synonymousip' in lbl:
            return ("DDoS", 2, 1)

        # 1: DoS
        if 'dos' in lbl or 'flood' in lbl or 'slowloris' in lbl:
            return ("DoS", 1, 1)

        # 8: Reconnaissance, PortScan, Infiltration, Malware, BruteForce
        if any(x in lbl for x in ['scan', 'recon', 'discovery', 'pingsweep', 'bruteforce', 'malware',
                                  'backdoor', 'infilteration', 'infiltration', 'vulnerability',
                                  'browserhijacking', 'sql', 'xss', 'commandinjection', 'uploading']):
            return ("Recon_Infiltration", 8, 1)

        # Default fallback
        return ("Recon_Infiltration", 8, 1)

    def extract_label_series(self, df: pd.DataFrame) -> pd.Series:
        """Finds or infers label column in dataframe."""
        for col in df.columns:
            if str(col).lower().strip() in ['class', 'label', 'attack', 'category', 'normal']:
                return df[col]
        # Check source dataset
        if '_source_dataset' in df.columns:
            return df['_source_dataset']
        return pd.Series(["BENIGN"] * len(df), index=df.index)

    def harmonize_physical_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonizes Physical/Telemetry logs using the canonical intersection of
        Generic and DJI Tello SDK schemas.
        Canonical fields:
        [pitch, roll, yaw, velocity_x, velocity_y, velocity_z, height,
         battery, barometer, flight_time, temperature, distance,
         accel_x, accel_y, accel_z]
        """
        cols_lower = {str(c).lower().strip(): c for c in df.columns}
        res = pd.DataFrame(index=df.index)

        # 1. Pitch, Roll, Yaw
        res['pitch'] = df[cols_lower['pitch']] if 'pitch' in cols_lower else 0.0
        res['roll'] = df[cols_lower['roll']] if 'roll' in cols_lower else 0.0
        res['yaw'] = df[cols_lower['yaw']] if 'yaw' in cols_lower else 0.0

        # 2. Velocities (vgx/x_speed -> velocity_x, etc.)
        res['velocity_x'] = df[cols_lower['vgx']] if 'vgx' in cols_lower else (df[cols_lower['x_speed']] if 'x_speed' in cols_lower else 0.0)
        res['velocity_y'] = df[cols_lower['vgy']] if 'vgy' in cols_lower else (df[cols_lower['y_speed']] if 'y_speed' in cols_lower else 0.0)
        res['velocity_z'] = df[cols_lower['vgz']] if 'vgz' in cols_lower else (df[cols_lower['z_speed']] if 'z_speed' in cols_lower else 0.0)

        # 3. Height / Altitude
        if 'height' in cols_lower:
            res['height'] = df[cols_lower['height']]
        elif 'h' in cols_lower:
            res['height'] = df[cols_lower['h']]
        elif 'tof' in cols_lower:
            res['height'] = df[cols_lower['tof']]
        else:
            res['height'] = 0.0

        # 4. Battery
        if 'battery' in cols_lower:
            res['battery'] = df[cols_lower['battery']]
        elif 'bat' in cols_lower:
            res['battery'] = df[cols_lower['bat']]
        else:
            res['battery'] = 100.0

        # 5. Barometer
        if 'barometer' in cols_lower:
            res['barometer'] = df[cols_lower['barometer']]
        elif 'baro' in cols_lower:
            res['barometer'] = df[cols_lower['baro']]
        else:
            res['barometer'] = 0.0

        # 6. Flight time
        if 'flight_time' in cols_lower:
            res['flight_time'] = df[cols_lower['flight_time']]
        elif 'time' in cols_lower:
            res['flight_time'] = df[cols_lower['time']]
        else:
            res['flight_time'] = 0.0

        # 7. Temperature
        if 'temperature' in cols_lower:
            res['temperature'] = df[cols_lower['temperature']]
        elif 'templ' in cols_lower and 'temph' in cols_lower:
            res['temperature'] = (pd.to_numeric(df[cols_lower['templ']], errors='coerce') + 
                                  pd.to_numeric(df[cols_lower['temph']], errors='coerce')) / 2.0
        elif 'templ' in cols_lower:
            res['temperature'] = df[cols_lower['templ']]
        else:
            res['temperature'] = 25.0

        # 8. Distance
        if 'distance' in cols_lower:
            res['distance'] = df[cols_lower['distance']]
        elif 'tof' in cols_lower:
            res['distance'] = df[cols_lower['tof']]
        elif 'z' in cols_lower:
            res['distance'] = df[cols_lower['z']]
        else:
            res['distance'] = 0.0

        # 9. Accelerations
        res['accel_x'] = df[cols_lower['agx']] if 'agx' in cols_lower else 0.0
        res['accel_y'] = df[cols_lower['agy']] if 'agy' in cols_lower else 0.0
        res['accel_z'] = df[cols_lower['agz']] if 'agz' in cols_lower else 0.0

        # Labels
        label_raw = self.extract_label_series(df)
        labels_parsed = [self.map_label_to_taxonomy(x) for x in label_raw]
        res['canonical_label'] = [x[0] for x in labels_parsed]
        res['class_id'] = [x[1] for x in labels_parsed]
        res['binary_label'] = [x[2] for x in labels_parsed]

        return res

    def harmonize_network_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonizes Network flow/frame logs to the canonical Network feature set:
        [flow_duration, tx_packets, rx_packets, tx_bytes, rx_bytes,
         packet_rate, byte_rate, mean_packet_size, mean_delay,
         mean_jitter, lost_packets, flag_syn, flag_ack, ttl_or_hops, protocol_type]
        """
        cols_map = {re.sub(r'[^a-zA-Z0-9]', '', str(c).lower()): c for c in df.columns}
        res = pd.DataFrame(index=df.index)

        # 1. Flow duration
        for k in ['flowdurations', 'flowduration', 'duration', 'frametimerelative']:
            if k in cols_map:
                res['flow_duration'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'flow_duration' not in res:
            res['flow_duration'] = 0.0

        # 2. Tx Packets
        for k in ['txpackets', 'totalfwdpackets', 'fwdpktstot', 'syncount']:
            if k in cols_map:
                res['tx_packets'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'tx_packets' not in res:
            res['tx_packets'] = 1.0

        # 3. Rx Packets
        for k in ['rxpackets', 'totalbackwardpackets', 'bwdpktstot', 'ackcount']:
            if k in cols_map:
                res['rx_packets'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'rx_packets' not in res:
            res['rx_packets'] = 0.0

        # 4. Tx Bytes
        for k in ['txbytes', 'totallengthoffwdpackets', 'fwdheadersizetot', 'totsize', 'framelen']:
            if k in cols_map:
                res['tx_bytes'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'tx_bytes' not in res:
            res['tx_bytes'] = 0.0

        # 5. Rx Bytes
        for k in ['rxbytes', 'totallengthofbwdpackets', 'bwdheadersizetot', 'datalen']:
            if k in cols_map:
                res['rx_bytes'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'rx_bytes' not in res:
            res['rx_bytes'] = 0.0

        # 6. Packet Rate
        for k in ['txpacketrates', 'flowpacketss', 'rate', 'fwdpktspersec']:
            if k in cols_map:
                res['packet_rate'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'packet_rate' not in res:
            res['packet_rate'] = res['tx_packets'] / (res['flow_duration'] + 1e-5)

        # 7. Byte Rate
        for k in ['txbyterates', 'flowbytess']:
            if k in cols_map:
                res['byte_rate'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'byte_rate' not in res:
            res['byte_rate'] = (res['tx_bytes'] + res['rx_bytes']) / (res['flow_duration'] + 1e-5)

        # 8. Mean Packet Size
        for k in ['meanpacketsize', 'fwdpacketlengthmean', 'headerlength', 'framelen']:
            if k in cols_map:
                res['mean_packet_size'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'mean_packet_size' not in res:
            res['mean_packet_size'] = 64.0

        # 9. Mean Delay / IAT
        for k in ['meandelays', 'flowiatmean', 'iat', 'timesincelastpacket', 'frametimedeltadisplayed']:
            if k in cols_map:
                res['mean_delay'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'mean_delay' not in res:
            res['mean_delay'] = 0.0

        # 10. Mean Jitter / IAT Std
        for k in ['meanjitters', 'flowiatstd', 'std', 'variance']:
            if k in cols_map:
                res['mean_jitter'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'mean_jitter' not in res:
            res['mean_jitter'] = 0.0

        # 11. Lost Packets / Drop Rate
        for k in ['lostpackets', 'packetdroprate', 'downupratio']:
            if k in cols_map:
                res['lost_packets'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'lost_packets' not in res:
            res['lost_packets'] = 0.0

        # 12. Flag SYN
        for k in ['synflagnumber', 'wlanfcretry', 'synflagcount']:
            if k in cols_map:
                res['flag_syn'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'flag_syn' not in res:
            res['flag_syn'] = 0.0

        # 13. Flag ACK
        for k in ['ackflagnumber', 'tcpflags', 'ackflagcount']:
            if k in cols_map:
                res['flag_ack'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'flag_ack' not in res:
            res['flag_ack'] = 0.0

        # 14. TTL / Hops
        for k in ['averagehopcount', 'timetolive', 'ipttl']:
            if k in cols_map:
                res['ttl_or_hops'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'ttl_or_hops' not in res:
            res['ttl_or_hops'] = 64.0

        # 15. Protocol
        for k in ['protocol', 'protocoltype', 'ipproto']:
            if k in cols_map:
                res['protocol_type'] = pd.to_numeric(df[cols_map[k]], errors='coerce')
                break
        if 'protocol_type' not in res:
            res['protocol_type'] = 6.0

        # Labels
        label_raw = self.extract_label_series(df)
        labels_parsed = [self.map_label_to_taxonomy(x) for x in label_raw]
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
        - Imputes missing values (<5%) with column medians.
        """
        raw_count = len(df)
        feature_cols = [c for c in df.columns if c not in ['canonical_label', 'class_id', 'binary_label']]

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
                dropped_features.append(c)
            elif null_ratio > 0:
                median_val = df_clean[c].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[c] = df_clean[c].fillna(median_val)
                imputed_features.append(c)

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

    def run_harmonization(self, routed_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Executes Phase B: Harmonizes, cleans, and aligns labels for both branches.
        """
        print("\n========================================================")
        print("  PHASE B: CLEANING, HARMONIZATION & TAXONOMY ALIGNMENT")
        print("========================================================")
        harmonized_branches = {}

        for branch in ["Network", "Physical"]:
            dfs = routed_data.get(branch, [])
            if not dfs:
                print(f"[!] Warning: No data available for {branch} branch.")
                continue

            print(f"[*] Harmonizing {len(dfs)} sources for {branch} Branch...")
            harmonized_dfs = []
            for df in dfs:
                if branch == "Physical":
                    h_df = self.harmonize_physical_dataframe(df)
                else:
                    h_df = self.harmonize_network_dataframe(df)
                harmonized_dfs.append(h_df)

            combined_df = pd.concat(harmonized_dfs, ignore_index=True)
            print(f"    Raw combined shape for {branch}: {combined_df.shape}")

            # Clean and deduplicate
            clean_df = self.clean_and_deduplicate(combined_df, branch)
            print(f"    Cleaned & deduplicated shape for {branch}: {clean_df.shape}")
            print(f"    Class distribution:\n{clean_df['canonical_label'].value_counts()}")

            harmonized_branches[branch] = clean_df

        # Save deduplication report
        dedup_df = pd.DataFrame(self.dedup_records)
        dedup_path = self.config.reports_dir / "deduplication_report.csv"
        dedup_df.to_csv(dedup_path, index=False)
        print(f"\n[+] Saved deduplication report to: {dedup_path}")

        return harmonized_branches
