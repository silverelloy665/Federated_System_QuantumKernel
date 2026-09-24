# SwarmGuard: Federated Quantum-Kernel UAV Intrusion Detection System

SwarmGuard is a federated quantum machine learning intrusion detection preprocessing pipeline engineered for autonomous UAV swarms. It utilizes a decoupled **Two-Branch Architecture** that independently processes cyber-physical telemetry and network traffic flows, extracting continuous features, performing outlier transformation, reducing them to 12 orthogonal dimensions for a 12-qubit quantum kernel, and partitioning for Federated Learning (FedAvg with non-IID Dirichlet heterogeneity $\alpha=0.5$).

---

## 🚀 Key Architectural Features

### 1. Two-Branch Structural Isolation
* **Physical / Telemetry Branch:** Processes cyber-physical kinematics and sensor streams (`Dataset_T-ITS.csv`, `UAV-NDD`). Extracts 12 continuous canonical features (`pitch`, `roll`, `yaw`, `velocity_x`, `velocity_y`, `velocity_z`, `height`, `battery`, `barometer`, `flight_time`, `temperature`, `distance`) and synthesizes 4 deterministic kinematic invariants (`speed_norm`, `speed_horizontal`, `angular_speed_norm`, `kinetic_energy_z_proxy`) into 16 continuous dimensions before PCA-12 compression.
* **Network Traffic Branch:** Processes flow captures and packet frames (`UAVIDS-2025`, `CICIDS2017`, `CIC-IoT`, `UAV-NDD`). Harmonizes durations, packet/byte counts, packet rates, inter-arrival time (IAT) statistics, drop rates, flags, and hop counts into continuous flow vectors with fit-on-train $\text{log1p}$ outlier stabilization.
* **Orthogonal Separation:** These two branches operate on orthogonal feature spaces and are processed strictly in parallel without table concatenation.

### 2. Zero-Leakage Pipeline
* All transformers (`log1p`, `SimpleImputer`, `StandardScaler`, `PCA(n_components=12)`, `QuantumPhaseAngleScaler`) are fitted strictly on training partitions and applied to validation, test, and federated client sets.
* Target-leaking identifiers (raw IP addresses, MACs, ports, timestamps, flow IDs) are stripped prior to model transformations.

### 3. Unified 9-Class UAV Intrusion Taxonomy

| Class ID | Standard Label | Binary Label | Description & Sub-Types |
|:---:|:---|:---:|:---|
| `0` | **BENIGN** | `0 (BENIGN)` | Normal flight telemetry & benign network traffic |
| `1` | **DoS** | `1 (ATTACK)` | Denial of service, SYN/UDP floods, slowloris |
| `2` | **DDoS** | `1 (ATTACK)` | Distributed denial of service, botnet/mirai floods |
| `3` | **Jamming_Deauth** | `1 (ATTACK)` | RF jamming, 802.11 de-authentication frames, Evil Twin |
| `4` | **Spoofing_MITM** | `1 (ATTACK)` | FakeLanding, GPS spoofing, ARP cache poisoning, MITM |
| `5` | **FDI** | `1 (ATTACK)` | False data injection into sensors and control telemetry |
| `6` | **Routing_Attack** | `1 (ATTACK)` | Sybil, Blackhole, Wormhole, routing loops |
| `7` | **Replay** | `1 (ATTACK)` | Replayed telemetry commands and network packets |
| `8` | **Recon_Infiltration** | `1 (ATTACK)` | Port scans, ping sweep, OS fingerprinting, malware/backdoor |

### 4. 12-Qubit Quantum Kernel Encoding
* Features are standardized and compressed to 12 orthogonal dimensions matching the 12-qubit quantum register.
* PCA features are scaled into the Pauli rotation angle space $[-\pi, \pi]$:
  $$\theta_j = 2\pi \cdot \frac{x_j - x_{j,\min}}{x_{j,\max} - x_{j,\min}} - \pi$$

### 5. Federated Non-IID Dirichlet Swarm Partitioning
* Training splits are partitioned across 5 simulated client UAV nodes using a Dirichlet distribution ($\alpha = 0.5$) to replicate realistic swarm heterogeneity.

---

## 📁 Repository & Artifact Directory Structure

```text
MERGED_CSV/
├── reports/
│   ├── dataset_inventory.csv          # Profiling of rows, columns, nulls, and sampling disclosure
│   ├── deduplication_report.csv       # Exact record-level deduplication logs per branch
│   ├── feature_mapping.csv            # Auditable exact/fuzzy/derived column mapping decisions
│   ├── label_mapping.csv              # Authoritative 9-class taxonomy and raw label audit
│   ├── skewness_report.csv            # Pre/post log1p skewness reduction metrics
│   └── data_leakage_report.csv        # Zero-leakage, PCA reduction, and Pauli angle bounds
├── centralized/
│   ├── physical_branch_train_test.npz # (X_train, X_test, y_train, y_test, y_train_binary, y_test_binary)
│   └── network_branch_train_test.npz  # (X_train, X_test, y_train, y_test, y_train_binary, y_test_binary)
├── federated_clients/
│   ├── physical/ (client_1.npz ... client_5.npz)
│   └── network/ (client_1.npz ... client_5.npz)
└── preprocessing_objects/
    ├── physical_scaler.pkl
    ├── physical_pca_12.pkl
    ├── network_scaler.pkl
    ├── network_pca_12.pkl
    ├── physical_angle_scaler.pkl
    ├── network_angle_scaler.pkl
    └── label_mappings.json
```

---

## 📦 Dataset Setup & Reproducibility

Place raw source datasets into `./data/raw/` or configure search directories in `.env` / CLI flags:

| Target Identifier | Dataset Name | Source File | Description |
|:---|:---|:---|:---|
| `uavids` | UAVIDS-2025 | `UAVIDS-2025.csv` | Drone network flow captures |
| `tits` | Dataset_T-ITS | `Dataset_T-ITS.csv` | Multiplexed Wi-Fi and DJI kinematics |
| `uav_ndd` | UAV-NDD | `UAV-NDD CSV.zip` | 802.11 raw packet frames and GCS telemetry |
| `csv_iot` | CIC-IoT | `CSV.zip` | IoT proxy network flows |
| `cve_ml` / `flows_iscx` | CICIDS2017 | `MachineLearningCSV.zip` | Flow captures with IAT statistics |

---

## ⚡ Quickstart: Running the Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing with default controlled sampling and report generation
python run_preprocessing.py --write-reports

# 3. (Optional) Run full ingestion on all raw rows
python run_preprocessing.py --full-ingest --write-reports

# 4. Run CI test suite
pytest tests/test_report_consistency.py -v

# 5. Verify preprocessed artifact integrity and Pauli bounds
python verify_artifacts.py
python test_quantum_tensors.py
```
