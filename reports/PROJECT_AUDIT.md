# SwarmGuard Project Audit

**Date:** 2026-09-25
**Scope:** Independent check of every fix claimed so far, across preprocessing, modeling, the QPU fabrication fix and the test suite.
**Method:** Every verdict below comes from reading the current code, loading the committed data files, running the tests, or querying IBM Quantum from this machine. Commit messages, the README and Excel log contents were not taken as evidence. Where one was checked, the check is stated.

**Environment used:** a repo-local `.venv` (Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0). The global Python has scikit-learn 1.5.0 and no `qiskit-ibm-runtime`.

**Repo state at audit time:** `main` @ `0608764`, with the QPU fix still **uncommitted**:
`M scripts/update_modeling_log.py`, `M swarmguard_modeling/modeling_logger.py`, `M swarmguard_modeling/qpu_executor.py`, `M tests/test_modeling_consistency.py`, `M reports/SWARMGUARD_MODELING_LOG.xlsx`, `M MERGED_CSV/reports/SWARMGUARD_MODELING_LOG.xlsx`, `?? scripts/test_ibm_connection.py`.

**Verdict key:**
- **VERIFIED:** the claim holds.
- **PARTIALLY VERIFIED:** the claim holds, but defects the claim doesn't mention were found alongside it.
- **NOT VERIFIED:** the claim does not hold, or the known issue is still open.
- **REGRESSED:** was working, now broken. No item received this verdict.

## Summary

| # | Item | Verdict |
|---|---|---|
| A1 | Single config / taxonomy | PARTIALLY VERIFIED |
| A2 | PCA genuinely reduces | VERIFIED |
| A3 | Angle bounds [0, π] | VERIFIED |
| A4 | `verify_artifacts.py` staleness | NOT VERIFIED (still stale) |
| A5 | Sampling disclosure | VERIFIED |
| A6 | Feature-mapping fallback | PARTIALLY VERIFIED |
| A7 | Label fallback / `is_unmapped` | PARTIALLY VERIFIED |
| A8 | Duplicate train/test rows | VERIFIED (0.42% / 2.43%) |
| A9 | scikit-learn version mismatch | NOT VERIFIED (unresolved) |
| B10 | QCNN architecture | PARTIALLY VERIFIED (feature→wire order inverted) |
| B11 | Optimizers / QNSPSA metric | PARTIALLY VERIFIED (QNSPSA is still a proxy) |
| B12 | Evaluation slice | NOT VERIFIED (still `X_test[:150]`) |
| B13 | Binary-only scope documented | NOT VERIFIED (undocumented) |
| B14 | Circular-mean aggregation | PARTIALLY VERIFIED (wrong period for CRz; hardcoded baseline) |
| B15 | Alert fusion 4-case + Δt | PARTIALLY VERIFIED (Δt is never used) |
| B+ | *Extra:* control-matrix results | NOT VERIFIED (QCNN untrained; all arms are constant predictors) |
| C16 | QPU root-cause fix in code | VERIFIED (uncommitted) |
| C17 | `build_qpu_table()` NOT_RUN behaviour | VERIFIED |
| C18 | Job-ID test against current log | VERIFIED (executed, passes; job confirmed on IBM) |
| C19 | `cross_check_ibm_quantum.py` PASSED | NOT VERIFIED (still unconditional, as expected) |
| C20 | Placeholder QPU input | NOT VERIFIED (still `np.ones(12)*0.5`, untrained weights) |
| C21 | `requirements.txt` completeness | NOT VERIFIED (no Qiskit packages listed) |
| D22 | Test-suite health / 4 skips | VERIFIED (cause found: path mismatch) |
| (extra) | `preprocess_uav_ids.py` | NOT VERIFIED (file does not compile) |

---

## A. Preprocessing pipeline

### A1. Single source of truth for taxonomy/config
**Evidence:**
- **One `config.py`:** `git ls-files` shows exactly one: `swarmguard_pipeline/config.py`.
- **Canonical taxonomy:** `TAXONOMY_CLASSES` is defined at `swarmguard_pipeline/config.py:20` (9 classes).
- **Harmonizer import:** `cleaning_harmonizer.py:14` imports it: `from .config import PipelineConfig, TAXONOMY_CLASSES, UNKNOWN_CLASS_LABEL, UNKNOWN_CLASS_ID`. `run_pipeline.py:12` and `generate_project_documentation.py:33` import it from config as well.
- **Test:** `test_single_taxonomy_definition` passes.

**Defects found:**
1. **Second, conflicting taxonomy:** `preprocess_uav_ids.py:63-73` defines `TAXONOMY_CLASSES` with different names (`GPS_SPOOFING`, `EVIL_TWIN`, `MALWARE_INJECTION`, …) and `QUANTUM_ANGLE_MIN = -np.pi` (`:77`). The file is a broken merge: line 46 puts an `import` at column 0 inside the class body. `python -m py_compile preprocess_uav_ids.py` fails with `IndentationError: unexpected indent (line 50)`. It can't run, so it can't feed data, but it's a contradictory second definition in the repo.
2. **Duplicate field declarations:** `config.py` declares several dataclass fields twice, which is merge debris. They are `input_search_dirs` (87, 93), `output_dir` (96, 98), `target_files` (101, 109), `impute_null_threshold`/`drop_col_null_threshold` (112-115) and `taxonomy_classes` (146, 157). The later line wins, so behaviour is correct: the hardcoded `C:\Users\Aarush\...` defaults are overridden. Still, the file has two versions of the truth.

**Verdict: PARTIALLY VERIFIED.** The live pipeline uses one source. The repo still contains a second, contradictory taxonomy in a file that doesn't compile.

### A2. PCA is genuinely reducing dimensionality
**Evidence:** Loaded `centralized/*_branch_train_test.npz` and `preprocessing_objects/*_pca_12.pkl`:

| Branch | Raw features into PCA | Components | Explained variance | X_train shape | X_test shape |
|---|---|---|---|---|---|
| Network | 14 | 12 | 99.28% | (200163, 12) | (50041, 12) |
| Physical | 16 | 12 | 92.82% | (9556, 12) | (2390, 12) |

`reports/data_leakage_report.csv` shows the same values (`REDUCED (14 -> 12)`, `REDUCED (16 -> 12)`). Both branches have more than 12 raw features, so PCA is not a no-op rotation.

**Note:** Network is only a 14 → 12 reduction, and 99.28% of variance is kept, so it is close to a rotation in practice. See B10 for how the components are then assigned to qubits.

**Verdict: VERIFIED.**

### A3. Angle bounds are [0, π]
**Evidence:** Loaded both `.npz` files directly (π = 3.1415926536):

| Branch | Split | min | max |
|---|---|---|---|
| Network | X_train | 0.0000000000 | 3.1415926536 |
| Network | X_test | 0.0000000000 | 3.1415926536 |
| Physical | X_train | 0.0000000000 | 3.1415926536 |
| Physical | X_test | 0.0000000000 | 3.1415926536 |

- **No leftover [-π, π] data:** zero negative values in either branch.
- **Scaler fitted on train only:** each qubit's train min is exactly 0 and max exactly π.
- **Federated clients:** all 10 client files are within [0, π].
- **Identical copies:** `MERGED_CSV/centralized` holds the same files (md5 matches).

**Verdict: VERIFIED.**

### A4. `verify_artifacts.py` staleness
**Evidence:**
- **Old range still checked:** `verify_artifacts.py:3` (docstring "bounds [-pi, pi]"), `:92-100` (`tr_min >= -np.pi - tol …`, prints "Pauli angle bounds [-pi, pi]") and `:115` (client check `c_min >= -np.pi - 1e-4`). A regression to [-π, π] data would still pass.
- **Run result:** executing `python verify_artifacts.py` ends in `>> VERIFICATION FAILED`. The failures are 11 missing-file errors, because the report CSVs are looked for under `MERGED_CSV/reports/` (see D22).

**Verdict: NOT VERIFIED (still stale, and the script currently fails).**

### A5. Sampling disclosure
**Evidence:** `reports/dataset_inventory.csv` has `rows_available`, `rows_sampled` and `sampling_method` for all 16 sources. `test_dataset_inventory_sampling_disclosure` passes when pointed at `reports/`.

| Source | Available | Sampled | Method |
|---|---|---|---|
| UAVIDS-2025, T-ITS Seg1–10 | (all) | (all) | FULL_INGEST |
| UAV-NDD_Case1 | 867,842 | 60,000 | HEAD_SAMPLE_60000 |
| UAV-NDD_AccessPoint_Case2 | 631,356 | 15,000 | HEAD_SAMPLE_15000 |
| UAV-NDD_GSC_Case3 | 823,632 | 15,000 | HEAD_SAMPLE_15000 |
| CIC_IoT_Proxies | 1,545,000 | 20,000 | STRATIFIED_CLUSTER_SAMPLE_20x1000 |
| CICIDS2017_NetworkFlows | 1,600,000 | 10,000 | HEAD_SAMPLE_5x2000 |

**Observation (not a disclosure failure):** head sampling produced label-degenerate subsets. Per `reports/label_mapping.csv`:
- **CICIDS2017:** all 10,000 sampled rows are `BENIGN`.
- **UAV-NDD_AccessPoint_Case2:** all 15,000 are `Normal`.

These sources contribute zero attack examples despite being attack datasets.

**Verdict: VERIFIED.** The caps are disclosed, not silent.

### A6. Feature-mapping fallback
**Evidence:**
- **The report:** `reports/feature_mapping.csv` has a `fallback_used` column. Counts are `NONE`: 245 and `LEFT_AS_NAN`: 65. All 65 `MISSING` rows are `LEFT_AS_NAN`, and there is no constant such as `battery=100.0`.
- **The code:** `cleaning_harmonizer.py:124, 161, 224` write `LEFT_AS_NAN` and set the column to `np.nan`.

**Defects the report does not surface:**
1. **Undisclosed constants in derived features:** missing inputs are filled with constants before deriving features, and none of this appears in `fallback_used`. Examples: `velocity_*`, `pitch/roll/yaw` → `fillna(0.0)` at `:168-173`; `tx_bytes`/`rx_bytes`/`rx_packets` → `0.0`, and `tx_packets` → `fillna(1.0)` at `:359-362`. So for a source missing `tx_packets`, `total_packets` and `packet_size_ratio` are computed from a fabricated packet count of 1.
2. **Imputation before the split:** `clean_and_deduplicate` (`:411-414`) fills NaNs with the median of the whole branch before the train/test split. Test-set statistics leak into train values. The effect is small, but it contradicts the "fitted strictly on X_train" claim in `data_leakage_report.csv`, which is only true of the Phase D imputer.

**Verdict: PARTIALLY VERIFIED.** The canonical fields are honest. Derived features still use undisclosed constant fills.

### A7. Label fallback / `is_unmapped`
**Evidence:**
- **The report:** `reports/label_mapping.csv` has an `is_unmapped` column, currently `False` for all 42 rows.
- **It can be True:** I ran `CleaningHarmonizer.map_label_to_taxonomy` plus `export_reports()` into a temporary directory. `'X99_Quantum_Alien_Probe'` → `('UNKNOWN', -1, 1)` and was exported with `is_unmapped = True`. The code is `cleaning_harmonizer.py:457`, `"is_unmapped": (cls_id == UNKNOWN_CLASS_ID)`.

**Defects (same run):**
1. **Missing labels become BENIGN:** `map(nan)` → `('BENIGN', 0, 0)` (`:31-34`), so a row with no label is labeled benign and never flagged as unmapped.
2. **Substring matching:** `map('none_of_the_above_attack')` → `('BENIGN', 0, 0)`, because `'none'` is a benign keyword (`:40`). Any label containing `normal`, `none`, `clear` or `regular` becomes BENIGN.
3. **Label-column fallback:** if no label column is found, `extract_label_series` (`:78-86`) uses the `_source_dataset` string, or all `"BENIGN"`, as the label.
4. **Questionable mapping:** `evil_twin` (a rogue access point, which is MITM-like) maps to `Jamming_Deauth` (`:49`). That covers 11,156 rows across T-ITS Seg7 and Seg8.

**Verdict: PARTIALLY VERIFIED.** The mechanism exists and works for unrecognized strings. Missing labels and substring collisions silently become BENIGN instead of UNKNOWN.

### A8. Duplicate train/test rows
**Evidence:** Byte-exact comparison of each X_test row against the X_train rows:

| Branch | Test rows identical to a train row | Within-test duplicates | Class of leaked rows |
|---|---|---|---|
| Network | 209 / 50,041 = **0.42%** | 93 | 100% BENIGN (vs 27.1% overall) |
| Physical | 58 / 2,390 = **2.43%** | 9 | 100% ATTACK (vs 64.2% overall) |

- **Earlier 0.44% figure:** that check rounded values to 1e-10 and found 220 rows. The exact byte-level count is 209 (0.42%). Physical is unchanged at 58 (2.43%).
- **Label conflicts:** 3 leaked network rows have a train twin with the opposite binary label.
- **Likely mechanism (not verified):** deduplication (`cleaning_harmonizer.py:399`) runs on all harmonized columns, including columns later dropped for >30% missing. Median imputation happens after deduplication. So rows that differ only in a dropped column, or in which cells are NaN, become identical after cleaning.
- **Impact:** this can inflate test accuracy by at most the leak rate, if a model memorizes (≤0.42 pp network, ≤2.43 pp physical). The physical leak is all one class. The current models are effectively untrained (see B+), so today's numbers aren't affected, but trained models would be.

**Verdict: VERIFIED** (0.42% / 2.43%). Still unfixed; flag it before reporting trained-model accuracy.

### A9. scikit-learn version mismatch
**Evidence:**
- **Warning on load:** unpickling `preprocessing_objects/{network,physical}_{pca_12,scaler}.pkl` in the venv (scikit-learn 1.9.1) emits `InconsistentVersionWarning: Trying to unpickle estimator PCA from version 1.8.0 when using version 1.9.1`, and the same for StandardScaler. The global Python (1.5.0) gives the equivalent warning for 1.8.0.
- **No pin:** `requirements.txt` has `scikit-learn>=1.2.0`, so any version installs.

**Verdict: NOT VERIFIED (unresolved).** Pin `scikit-learn==1.8.0`, or regenerate the pickles under a pinned version.

---

## B. Modeling pipeline

### B10. QCNN architecture
**Evidence:** Read `swarmguard_modeling/qcnn_ansatz.py:40-147` gate by gate:
- **Encoding:** `Ry(x_i)` on wires 0–11 (`:58-59`).
- **Conv 1:** 6 pairs (0,1)…(10,11), `Ry·Ry·CZ`, 12 parameters (`:66-69`).
- **Re-upload:** `Ry(x_i)` on all wires (`:75-76`).
- **Pre-pool:** `CRz(src→tgt)` + `CX(tgt→src)` on (1→0), (3→2), (5→4), (7→6), 4 parameters (`:84-87`). Wires 8–11 are untouched, leaving active wires {0, 2, 4, 6, 8, 9, 10, 11}, which is 8.
- **Conv 2:** (0,2), (4,6), (8,9), (10,11), 8 parameters.
- **Pool 8 → 4:** (2→0), (6→4), (9→8), (11→10) leaves {0, 4, 8, 10}, 4 parameters.
- **Conv 3:** (0,4), (8,10), 4 parameters.
- **Pool 4 → 2:** (4→0), (10→8) leaves {0, 8}, 2 parameters.
- **Conv 4:** `Ry(θ34)` on q0 + `CZ(0,8)`, 1 parameter.
- **Pool 2 → 1:** (8→0), 1 parameter.
- **Totals:** 12+4+8+4+4+2+1+1 = **36 parameters**. The observable `"I"*11+"Z"` is Z on q0 (Qiskit little-endian). `test_qcnn_ansatz_parameter_budget` passes.

**Defect: feature-to-wire order is the reverse of the design.**
- **The claim:** the docstring (`:4-7`), the comments at `:55-56` and the Step 1 log row all say the weakest 8 PCA components go on wires 0–7 (pooled) and the strongest 4 on wires 8–11 (pass-through).
- **What happens:** `quantum_formatter.py` saves PCA columns in scikit-learn order, which is descending variance. Nothing in `swarmguard_modeling/` reorders them (searched for `::-1`, `argsort` and `explained_variance`). So wire *i* = component *i*:
  - Network: wires 0–3 carry 32.8 + 13.66 + 11.54 + 10.89 = **68.9%** of the variance and are pooled first. Pass-through wires 8–11 carry 2.27 + 1.30 + 1.24 + 0.62 = **5.4%**.
  - Physical: the pass-through wires carry 5.20 + 5.05 + 4.13 + 3.59 = 18.0%.

**Verdict: PARTIALLY VERIFIED.** The gate sequence matches the 12 → 8 → 4 → 2 → 1 design. The data feeding it contradicts the stated reason for the asymmetric pre-pool.

### B11. Optimizers
**Evidence:** All four are in `swarmguard_modeling/optimizers.py`: `run_rotosolve` (`:34`), `run_spsa` (`:75`), `run_qnspsa` (`:114`) and `run_adam` (`:155`).
- **QNSPSA is still the proxy.** At `:135-137`: `ghat = …; F_diag = np.abs(ghat) + 0.05; nat_grad = ghat / F_diag`. There is no second perturbation, no state-fidelity evaluation and no Fubini–Study estimate. The update is close to a sign-SGD step. The formula column in the Excel log (`(F_hat_k + lambda*I)^-1 * g_k`) describes something the code doesn't do.
- **QNSPSA evaluation count is doubled:** `:133` adds `4 * len(xs)` evaluations per step, but only 2 loss evaluations run (`:131-132`). The logged "3,600 total circuit evals" is twice the real number.
- **Rotosolve and ADAM "exact" claims are overstated:** both apply the ±π/2 rule to the binary cross-entropy *loss*, which is not a sinusoid in θ. They also apply it to the 11 CRz parameters, which need the four-term shift rule. So the log's "exact analytic global minimum" and "exact parameter-shift analytical gradient" notes are not accurate.
- **What the benchmark numbers mean:** the QNSPSA row is a heuristic scaled-SPSA, not quantum natural gradient.

**Verdict: PARTIALLY VERIFIED.** All four are implemented. QNSPSA is not a genuine metric estimate.

### B12. Evaluation slice
**Evidence:**
- **Still unshuffled:** `optimizers.py:61, 100, 141, 185` all use `X_test[:150]` / `y_test[:150]`. `federated_trainer.py:103, 108` use `X_test[:200]` and `alert_fusion.py:118-119` uses `[:250]`. The control matrix (`control_models.py:245`) uses a random but unstratified sample of 150.
- **Class coverage (checked on the data):**
  - Network `y_test[:150]` contains classes {0, 1, 2, 3, 5, 6, 7, 8}. **Class 4 (Spoofing_MITM) is absent**, and it stays absent at 200 and 250.
  - Physical `[:150]` contains all 5 of its classes.
  - Binary split in network `[:150]`: 37 benign / 113 attack. Always predicting "attack" scores **75.3%**, and the four optimizers log 76.67–79.33%.

**Verdict: NOT VERIFIED (still unstratified).** With n = 150, one sample is 0.67 pp, so the optimizer ranking is within noise of the majority-class baseline.

### B13. Binary vs multiclass scope
**Evidence:**
- **Binary everywhere:**
  - `control_models.py:236-237`, `optimizers.py:207-208`, `federated_trainer.py:37, 88`: `y_*_binary`.
  - `alert_fusion.py:108-109`: binary.
  - `qcnn_ansatz.py:196-210`: binary `predict_proba`/`predict`.
- **No multiclass path:** the 9-class `y_train`/`y_test` arrays are never used by any model.
- **Documentation search** for `binary|multiclass|multi-class|limitation|scope`:
  - `README.md`: only the label table header and the `.npz` key list, with no scope statement.
  - `SWARMGUARD_MODELING_LOG.xlsx`: **0 hits**.
  - `SWARMGUARD_PROJECT_DOCUMENTATION.xlsx`: mentions of binary columns and multiclass-preserving splits, but no statement that the models are binary-only.

**Verdict: NOT VERIFIED.** The models are binary-only, and this is not documented anywhere.

### B14. Federated aggregation
**Evidence:** `federated_trainer.py:75-83`:
```python
total_samples = sum(client_sizes)
weights_array = np.array(client_weights)
fractions = np.array(client_sizes, dtype=np.float64) / float(total_samples)
sin_sum = np.sum(fractions[:, None] * np.sin(weights_array), axis=0)
cos_sum = np.sum(fractions[:, None] * np.cos(weights_array), axis=0)
theta_bar = np.arctan2(sin_sum, cos_sum)
```
This is a genuine sample-weighted, atan2-based circular mean with no linear averaging.

**Defects:**
1. **Wrong period for 11 of 36 parameters.**
   - The circular mean assumes period 2π, and `local_client_train` also wraps weights to [-π, π) (`:66`).
   - That holds for `Ry` (Ry(θ − 2π) = −Ry(θ), a global phase). It does not hold for the 11 `CRz` parameters (θ12–15, θ24–27, θ32, θ33, θ35): CRz has period 4π, and a 2π shift applies a relative phase on the control qubit.
   - Checked with `Operator.equiv`: `CRz(2.5) ≡ CRz(2.5 − 2π)` is **False**, while `Ry(2.5) ≡ Ry(2.5 − 2π)` is **True**.
   - On the real model, setting θ12 = 2.5 vs 2.5 − 2π changes ⟨Z0⟩ on `X_test[0]` from 0.2624 to 0.1997.
   - Aggregation and wrapping therefore change the model for those parameters. Use period 4π for CRz (for example, a circular mean of θ/2, doubled back).
2. **Hardcoded baseline in the log.** `run_federated_benchmark` (`:125`) sets `cent_dict = centralized_acc_dict or {"Network": 85.0, "Physical": 87.5}`, and `update_modeling_log.py` never passes real centralized accuracies. Sheet `4_Federated_Training` therefore reports "Centralized Accuracy 85 / 87.5" and "gap 7.5 / 26", which are **hardcoded numbers presented as measurements**. This is the same class of problem as the QPU bug.

**Verdict: PARTIALLY VERIFIED.** The atan2 aggregation is real. It uses the wrong period for CRz, and the reported comparison baseline is fabricated.

### B15. Alert fusion gate
**Evidence:** `alert_fusion.py:35-59` implements all 4 cases (Critical Compound / Cyber Infiltration / Kinematic Drift / Nominal) on `tau_c = tau_p = 0.60`.

**Defects:**
1. **No Δt window:** `delta_t_sec` is stored (`:27`) and printed in the config table (`:65, 68`), but **never used** in `evaluate_gate`. Each sample `i` is judged on `p_cyb[i]` and `p_phy[i]` alone. There are no timestamps and no sliding window.
2. **Synthetic ground truth:** `:125-136` pairs network test row *i* with physical test row *i*. They come from different datasets with no shared time axis.
3. **Untrained models:** both `QCNNModel`s run with random initial weights (`:115-116`). Sheet 5's confusion matrix shows 243 of 250 samples predicted "Critical Compound".
4. **Misleading empty-class score:** `class_acc` is set to 100.0 when a class has no samples (`:152`).

**Verdict: PARTIALLY VERIFIED.** The 4-case piecewise logic is real. The Δt sliding window is not implemented.

### B+. Extra finding: control-matrix results are not meaningful
**Evidence:**
- **QCNN arm never trained:** `control_models.py:253-257` builds `QCNNModel(branch, 12)` and calls `predict_proba` straight away, using random initial weights (`qcnn_ansatz.py:164-165`).
- **All arms predict a constant.** Sheet `2_Control_Matrix` in the current log:
  - Network: QCNN, MLP and MPS all show test accuracy 78.00 / macro-F1 0.4382. For 78% positives, predicting all-positive gives macro-F1 = ½ · 2(0.78)/1.78 = 0.438.
  - Physical: all four arms show 69.33 / 0.4094.
- **False claim in the log:** the "QCNN achieved superior generalization vs controls" text (`modeling_logger.py:291`) is hardcoded and not supported.

**Verdict: NOT VERIFIED.** The control matrix does not currently compare trained models.

---

## C. QPU submission fix

### C16. Root-cause fix in `qpu_executor.py`
**Evidence (current working tree):**
- **(a) Register name:** `:86`, `counts = getattr(pub_res.data, t_qc.cregs[0].name).get_counts()`. The name comes from the circuit (`"c"`), and `c0` appears nowhere in the file.
- **(b) No simulator substitution:** the `try`/`except` is gone (`:78-89`). The file contains no `except`. `AerSimulator` is used only at `:68-73` for `sim_z`, and its result is never assigned to `hw_z`.
- **(c) No hardcoded job ID:** `job_id` is assigned only at `:82` (`job.job_id()`). A grep for `ibm_\w+` finds only the import and the channel default string.

**Verdict: VERIFIED.** Caveat: the fix is **uncommitted** (see the repo state at the top of this report).

### C17. `build_qpu_table()` NOT_RUN behaviour
**Evidence:**
- **The code:** `scripts/update_modeling_log.py:32-72`. The import of `QPUInferenceExecutor` happens inside the factory, so a missing package is also caught. Every `Exception` returns `[backend_or_None, None, "12 Qubits", None, None, None, None, "NOT_RUN - <Type>: <msg>"]`. The only path to PASSED/COMPLETED is a successful `evaluate_qpu_job()` return, which needs a real `job.job_id()` and `job.result()`.
- **Timeline sheet:** `modeling_logger.py` `log_step_6` derives the Log status from the row.
- **Test:** `test_qpu_failure_is_logged_as_not_run` forces a connection failure and a job failure and passes. It printed:
  - `NOT_RUN` for `RuntimeError('401 Unauthorized: invalid token')`
  - `NOT_RUN` for `AttributeError("'DataBin' object has no attribute 'c0'")`

**Verdict: VERIFIED.**

### C18. Job-ID-format test against the current log
**Evidence:**
- **Test run:** `.venv\Scripts\python.exe -m pytest tests -rA` gives `PASSED tests/test_modeling_consistency.py::test_qpu_log_has_no_fabricated_hardware_results`.
- **Current row:** `ibm_marrakesh | daqo4ueekp0c73ar2b7g | Depth 94 | 1,024 | sim 0.7207 | hw 0.5820 | COMPLETED (Outside Shot Tolerance)`.
- **Independent network check:** `service.job("daqo4ueekp0c73ar2b7g")` gives `status: DONE, backend: ibm_marrakesh`. The stored counts `{'0': 810, '1': 214}` give ⟨Z⟩ = 0.5820, exactly the logged value. Account usage went from 178 s to 180 s.
- **Test sensitivity:** against the previous (fabricated) log, the same test failed on `ibm_hw_transpiled_1790220201`.

**Verdict: VERIFIED.**

### C19. `cross_check_ibm_quantum.py`
**Evidence:**
- **Unconditional PASSED:** `:202` hardcodes `"verification_status": "PASSED"`, and `:211` always prints "ALL 12-QUBIT TESTS PASSED".
- **No hardware execution:** the script connects to IBM and transpiles for the chosen backend (`:99-121`), but **never submits a job**. Every kernel and Gram-matrix value comes from `AerSimulator` (`:54-62, :129-171`).
- **Knock-on effect:** `generate_project_documentation.py:711-716` copies this status into the documentation workbook as "Hardware Verification Status: PASSED".

**Verdict: NOT VERIFIED.** Still broken, as expected. No new or undocumented progress.

### C20. Placeholder QPU input
**Evidence:**
- **Placeholder input:** `update_modeling_log.py:46` uses `dummy_sample = np.ones(12) * 0.5`.
- **Untrained weights:** `qpu_executor.py:59` builds a fresh `QCNNModel("Network", 12)`, so the weights are random seed-42 initial values, not trained ones.

The hardware check therefore validates transpilation and execution of an arbitrary circuit instance (sim 0.7207 vs hw 0.5820, depth 94). It says nothing about whether the trained detector transfers to hardware.

**Verdict: NOT VERIFIED.** It's still a placeholder.

### C21. `requirements.txt` completeness
**Evidence:** `requirements.txt` lists only numpy, pandas, scikit-learn, scipy, python-dotenv, openpyxl and pytest. It has **no** `qiskit`, `qiskit-aer` or `qiskit-ibm-runtime`, and nothing is pinned to an exact version. The audit venv only works because those packages were installed by hand. A fresh `pip install -r requirements.txt` would reproduce the original `ModuleNotFoundError: No module named 'qiskit_ibm_runtime'`. The versions that worked here are qiskit 2.5.2, qiskit-aer 0.17.2 and qiskit-ibm-runtime 0.50.0.

**Verdict: NOT VERIFIED.**

---

## D. Test suite health

### D22. Full suite and the 4 skips
**Evidence:**
- **Default run:** `.venv\Scripts\python.exe -m pytest tests -rA -q` gives **8 passed, 4 skipped, 0 failed**.
- **Skip reasons:** all four are explicit `pytest.skip()` calls when a file is missing:
  - `test_label_mapping_report_agreement`: `Report not found at MERGED_CSV\reports\label_mapping.csv`
  - `test_feature_mapping_report`: `…\feature_mapping.csv`
  - `test_dataset_inventory_sampling_disclosure`: `…\dataset_inventory.csv`
  - `test_quantum_bounds_and_leakage`: `…\data_leakage_report.csv`
- **Root cause:** `PipelineConfig.output_dir` resolves to `./MERGED_CSV`, both from `OUTPUT_DIR` in `.env` and from the code default in `get_default_output_dir()`. So `reports_dir` is `MERGED_CSV/reports`. The four CSVs exist only in the repo-root `reports/`, and `MERGED_CSV/reports/` holds just the two workbooks and the JSON. This is a path mismatch, not an import error or a broken fixture.
- **Confirmation:** with `OUTPUT_DIR=.` set, `tests/test_report_consistency.py` gives **8 passed, 0 skipped**, so all four previously-skipped tests pass on the real data.
- **Same cause elsewhere:** `verify_artifacts.py` fails for the same reason (A4).

**Suggested fix:** make one canonical output location, or have the tests fall back to `repo_root/reports`, as `test_modeling_log_workbook_structure` already does. Skipping on a missing file also hides regressions; consider failing instead.

**Verdict: VERIFIED.** The cause is found and the tests pass once the path is corrected. The mismatch itself is still unfixed.

---

## Final verdict

**Not yet safe to build new features on top of.** The preprocessing outputs are sound: shapes, [0, π] bounds, PCA reduction, split and client partitions all check out. The QPU fabrication fix is verified. But the modeling layer, which multiclass expansion and further QPU runs would build on, currently produces numbers that don't mean what the log says.

### Blocking items (fix before new work)
1. **Untrained models and a fabricated baseline in the log (B+, B14.2, B15.3, C20).** The QCNN is never trained anywhere its results are reported. All control-matrix arms are constant predictors. The federated "centralized accuracy" of 85 / 87.5 is hardcoded. Train the centralized QCNN, feed its real accuracy to the federated benchmark, and use trained weights for fusion and QPU inference.
2. **Feature-to-wire order inverted (B10).** The asymmetric pre-pool pools away the dominant PCA components and preserves the weakest. Either reverse the columns before encoding, or change the documented rationale.
3. **CRz periodicity in federated aggregation (B14.1).** 11 of 36 parameters are averaged and wrapped with the wrong period, which changes the model.
4. **Label fallback holes (A7).** NaN labels and substring collisions silently become BENIGN. This is especially blocking for multiclass expansion.
5. **Unstratified, too-small evaluation slices (B12).** Network class 4 never appears, and optimizer differences are within noise of the majority baseline. Required before any multiclass or accuracy claim.
6. **Reproducibility (C21, A9).** Add pinned `qiskit`, `qiskit-aer` and `qiskit-ibm-runtime` to `requirements.txt`, and pin scikit-learn to the version the pickles were saved with (1.8.0).
7. **Commit the verified QPU fix (C16–C18).** It currently exists only in the working tree.
8. **Before more QPU runs specifically (C19, C20):** use a real preprocessed test sample and trained weights, and stop `cross_check_ibm_quantum.py` from reporting PASSED unconditionally. The account has 420 s of QPU time left in the current period.

### Nice-to-have cleanup
- **Stale validation:** update `verify_artifacts.py` to check [0, π] (A4).
- **Report location:** unify it, and make the report tests fail rather than skip when files are missing (D22).
- **Broken legacy file:** delete or repair `preprocess_uav_ids.py` (it doesn't compile and holds the second taxonomy), and remove the duplicate field declarations in `config.py` (A1).
- **Undisclosed fills:** record the derived-feature constant fills (`fillna(0.0)` / `fillna(1.0)`) in `fallback_used`, and move median imputation after the split (A6).
- **Duplicates:** deduplicate on the final feature set after cleaning, to remove the 0.42% / 2.43% train/test overlap (A8).
- **QNSPSA:** replace the proxy with a real two-perturbation Fubini–Study estimate or rename the row, fix its evaluation counter, and soften the "exact" wording for Rotosolve and ADAM (B11).
- **Alert fusion:** implement the Δt window, or drop it from the gate description, and don't score empty classes as 100% (B15).
- **Scope:** document the binary-only scope in the README and the modeling log (B13).
- **Head sampling:** replace the head sampling that yielded all-benign CICIDS2017 and UAV-NDD Case2 subsets (A5), and revisit `evil_twin → Jamming_Deauth` (A7).
