# SwarmGuard: Federated Quantum-Kernel UAV Intrusion Detection System

SwarmGuard is a federated quantum machine learning intrusion detection system engineered for autonomous UAV swarms. It utilizes a decoupled **Two-Branch Architecture** that independently processes cyber-physical telemetry and network traffic flows, feeding a 12-qubit quantum kernel and trained via Federated Learning (FedAvg with non-IID Dirichlet partitioning $\alpha=0.5$).

---

## 🚀 Key Architectural Features

### 1. Two-Branch Structural Isolation (Non-Negotiable)
* **Physical/Telemetry Branch:** Processes cyber-physical kinematics and sensor streams (`Dataset_T-ITS.csv`, `UAV-NDD`). Extracts the 8-feature schema intersection between generic kinematic logs and DJI Tello SDK telemetry (`height`, `velocity_x`, `velocity_y`, `velocity_z`, `battery_pct`, `barometer`, `flight_time`, `distance_tof`) and synthesizes 4 deterministic kinematic invariants (`speed_norm`, `speed_horizontal`, `kinetic_energy_z_proxy`, `accel_z_approx`) to form a 12-dimensional physical baseline.
* **Network Traffic Branch:** Processes flow captures and packet frames (`UAVIDS-2025`, `CICIDS2017`, `CIC-IoT`, `UAV-NDD`). Harmonizes durations, packet/byte counts, inter-arrival time (IAT) statistics, and flags into continuous flow vectors.
* **Orthogonal Separation:** These two branches operate on orthogonal feature spaces and are processed strictly in parallel without table concatenation.

### 2. Zero-Leakage Pipeline
* All transformers (`SimpleImputer`, `StandardScaler`, `PCA(n_components=12)`, `QuantumPhaseAngleScaler`) are fitted strictly on training partitions and applied to validation/test sets.
* Target-leaking identifiers (raw IP addresses, MACs, ports, timestamps, flow IDs) are stripped prior to model transformations.

### 3. Unified 9-Class UAV Intrusion Taxonomy

| Class ID | Standard Label | Binary Label | Description & Sub-Types |
|:---:|:---|:---:|:---|
| `0` | **BENIGN** | `0` (BENIGN) | Normal flight telemetry & benign network traffic |
| `1` | **GPS_SPOOFING** | `1` (ATTACK) | FakeLanding, GPS coordinate manipulation |
| `2` | **GPS_JAMMING** | `1` (ATTACK) | RF jamming, control link disruption |
| `3` | **FALSE_DATA_INJECTION** | `1` (ATTACK) | FDI sensor attacks (*Flagged: T-ITS has network FDI, zero physical FDI*) |
| `4` | **REPLAY_ATTACK** | `1` (ATTACK) | Replayed telemetry commands and network packets |
| `5` | **EVIL_TWIN** | `1` (ATTACK) | Rogue access points, de-authentication frames |
| `6` | **DENIAL_OF_SERVICE** | `1` (ATTACK) | DoS, DDoS, SYN Flood, UDP/ICMP Flood, Routing Blackhole/Sybil |
| `7` | **RECONNAISSANCE** | `1` (ATTACK) | Port scanning, OS fingerprinting, ping sweep |
| `8` | **MALWARE_INJECTION** | `1` (ATTACK) | Backdoors, command injection, brute force, MITM |

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
│   ├── dataset_inventory.csv          # Profiling of rows, columns, nulls across all datasets
│   ├── duplicate_file_report.csv      # SHA-256 hash audit of raw input containers
│   ├── deduplication_report.csv       # Exact record-level deduplication logs per branch
│   ├── missing_value_report.csv       # Null percentage audit (<5% impute, >30% drop)
│   ├── feature_mapping.csv            # Mapping from canonical features to 12 qubits
│   ├── label_mapping.csv              # 9-class taxonomy and binary mappings
│   └── data_leakage_report.csv        # Zero-leakage and [-pi, pi] angle boundary verification
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
    └── label_mappings.json
```

---

## ⚡ Quickstart: Running Preprocessing

```bash
# Execute end-to-end preprocessing pipeline
python preprocess_uav_ids.py

# Open interactive data quality report
open DATASET_QUALITY_REPORT.html
```
