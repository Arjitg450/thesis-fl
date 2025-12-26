# FL-Aware Multi-Class EDA Report

Generated: `2025-12-24T10:28:22`

## 1️⃣ Dataset Integrity & Sanity Checks
- Total rows: **176098**
- Total columns: **112** (includes `__label`, `__source_file`) 
- Files: **7** (see `INDEX.csv`)
- Duplicate rows: **0** (exact)
- Duplicate rows excluding [id, __source_file]: **2**
- Potential leakage/artifact columns detected: **['label']**
- Missingness outputs: `missingness_global.csv`, `missingness_by_class.csv`
- Constant/near-constant outputs: `constant_features.csv`, `near_constant_features.csv`

Mostly-missing features (missing_pct ≥ 95%):
| feature               |   missing_pct |   nunique_non_null | dtype   |
|:----------------------|--------------:|-------------------:|:--------|
| server_fingerprint    |       98.8972 |                 14 | object  |
| client_fingerprint    |       98.8813 |                  1 | object  |
| content_type          |       96.8404 |                  1 | object  |
| user_agent            |       96.7581 |                  2 | object  |
| requested_server_name |       95.6394 |                178 | object  |

Globally constant features (safe to drop from modeling unless required for schema):
| feature                                 |   top_fraction |   nunique |
|:----------------------------------------|---------------:|----------:|
| bidirectional_cwr_packets               |              1 |         1 |
| bidirectional_ece_packets               |              1 |         1 |
| bidirectional_urg_packets               |              1 |         1 |
| dst2src_cwr_packets                     |              1 |         1 |
| dst2src_ece_packets                     |              1 |         1 |
| dst2src_urg_packets                     |              1 |         1 |
| src2dst_cwr_packets                     |              1 |         1 |
| src2dst_ece_packets                     |              1 |         1 |
| src2dst_urg_packets                     |              1 |         1 |
| udps.association_setup_request_counter  |              1 |         1 |
| udps.association_setup_response_counter |              1 |         1 |
| udps.heartbeat_request_counter          |              1 |         1 |
| udps.heartbeat_response_counter         |              1 |         1 |
| udps.pfd_management_request_counter     |              1 |         1 |
| udps.pfd_management_response_counter    |              1 |         1 |
| vlan_id                                 |              1 |         1 |

Leakage risk notes (do not blindly trust importance scores on these):
- bidirectional_first_seen_ms, bidirectional_last_seen_ms, dst2src_first_seen_ms, dst2src_last_seen_ms, dst_ip, dst_mac, dst_oui, expiration_id, id, src2dst_first_seen_ms, src2dst_last_seen_ms, src_ip, src_mac, src_oui

Leakage risk notes:
- IP/MAC fields (`src_ip`, `dst_ip`, `src_mac`, `dst_mac`, OUI) are strong identifiers; can overfit under random splits.
- Absolute timestamps (`*_first_seen_ms`, `*_last_seen_ms`) can encode collection time windows; treat carefully for FL/generalization.

## 2️⃣ Class Distribution & Imbalance Analysis
Class counts (7-class):
|                            |   count |
|:---------------------------|--------:|
| ddos_tcp_ddos              |   45033 |
| bruteForce                 |   30778 |
| ddos_slowloris             |   28563 |
| pfcp_session_modification  |   27780 |
| benign                     |   21060 |
| pfcp_session_deletion      |   15296 |
| pfcp_session_establishment |    7588 |

Plots:
- `class_counts.png`
- `class_counts_log.png`

FL impact:
- Non-IID clients may have minority classes completely absent; evaluation must include per-class recall and macro-averages.

## 3️⃣ Feature Sparsity & Zero-Dominated Features
Outputs:
- `zero_rates_global.csv`
- `zero_rates_by_class.csv`

Always/near-always zero features (zero_pct ≥ 99.999%):
| feature                                     |   zero_pct | dtype   |
|:--------------------------------------------|-----------:|:--------|
| vlan_id                                     |   100      | int64   |
| bidirectional_urg_packets                   |   100      | int64   |
| src2dst_urg_packets                         |   100      | int64   |
| udps.heartbeat_request_counter              |   100      | int64   |
| udps.heartbeat_response_counter             |   100      | int64   |
| udps.pfd_management_request_counter         |   100      | int64   |
| src2dst_ece_packets                         |   100      | int64   |
| src2dst_cwr_packets                         |   100      | int64   |
| udps.pfd_management_response_counter        |   100      | int64   |
| dst2src_urg_packets                         |   100      | int64   |
| udps.association_setup_response_counter     |   100      | int64   |
| dst2src_ece_packets                         |   100      | int64   |
| dst2src_cwr_packets                         |   100      | int64   |
| udps.association_setup_request_counter      |   100      | int64   |
| bidirectional_ece_packets                   |   100      | int64   |
| bidirectional_cwr_packets                   |   100      | int64   |
| udps.session_set_deletion_request_counter   |    99.9994 | float64 |
| udps.version_not_supported_response_counter |    99.9994 | float64 |
| udps.node_report_response_counter           |    99.9994 | float64 |
| udps.node_report_request_counter            |    99.9994 | float64 |

Decision rule (multiclass + FL-aware):
- Drop features that are **globally always-zero** or **globally constant**.
- Retain features that are sparse globally but **class-specific non-zero** (can be discriminative under partial label visibility).

## 4️⃣ Statistical Feature Analysis (Global & Class-wise)
Computed for numeric features:
- mean, median, variance, std
- skewness, kurtosis
- IQR, MAD (robust)

Outputs:
- `robust_stats_global.csv`
- `robust_stats_by_class_long.csv`

Interpretation guidance:
- High skew / kurtosis suggests heavy tails (consider log1p / RobustScaler).
- Compare benign vs each attack class to identify sensitive features.

## 5️⃣ Feature Distribution Visualization
Generated visuals (relationships-focused):
- `mi_top25.png` (top MI numeric features)
- `corr_pearson_heatmap.png`, `corr_spearman_heatmap.png`
- `geometry/pca_2d.png`, `geometry/tsne_2d.png`
- `importance/proxy_rf_confusion_matrix.png`
- `importance/shap_summary.png` (if SHAP succeeded)

For deeper per-feature violins/boxplots, we can generate targeted plots for the top-K discriminative features once you confirm K.

## 6️⃣ Correlation & Redundancy Analysis
Outputs:
- `corr_pearson.csv` + heatmap
- `corr_spearman.csv` + heatmap
- `mi_numeric_feature_label.csv`

Pruning strategy:
- Identify highly correlated clusters; retain 1 representative per cluster chosen by MI/ANOVA stability across FL clients.

## 7️⃣ Class Separability & Geometry
Outputs:
- `geometry/pca_2d.png`
- `geometry/tsne_2d.png`
- `geometry/pca_info.json` (variance explained)

Interpretation:
- Overlap between benign and certain attacks indicates hard separation; prioritize robust engineered ratios and UDPS/TCP-flag features.

## 8️⃣ Feature Importance WITHOUT Final Model
Computed:
- Mutual information (numeric) → `mi_numeric_feature_label.csv`
- ANOVA F-test (numeric) → `importance/anova_f_scores.csv`
- Chi-square on one-hot encoded features (proxy) → `importance/chi2_top500.csv`
- Proxy RandomForest importances + permutation importances → `importance/proxy_rf_*`
- SHAP (tree proxy, numeric-only) → `importance/shap_summary.png` (if available)

Top-15 by MI (numeric):
| feature                     |   mutual_info |
|:----------------------------|--------------:|
| bidirectional_first_seen_ms |      1.81335  |
| bidirectional_last_seen_ms  |      1.81329  |
| src2dst_first_seen_ms       |      1.81312  |
| src2dst_last_seen_ms        |      1.8131   |
| bidirectional_bytes         |      1.34107  |
| bidirectional_mean_ps       |      1.33462  |
| src2dst_bytes               |      1.28667  |
| bidirectional_min_ps        |      1.18563  |
| src2dst_mean_ps             |      1.14382  |
| bidirectional_max_ps        |      1.12595  |
| src2dst_max_ps              |      1.11509  |
| src2dst_min_ps              |      1.07549  |
| dst_port                    |      0.918351 |
| dst2src_last_seen_ms        |      0.887165 |
| dst2src_first_seen_ms       |      0.887096 |

Top-15 by ANOVA (numeric):
| feature                                    |   f_score |   p_value |
|:-------------------------------------------|----------:|----------:|
| bidirectional_first_seen_ms                |  175815   |         0 |
| src2dst_first_seen_ms                      |  175815   |         0 |
| src2dst_last_seen_ms                       |  175815   |         0 |
| bidirectional_last_seen_ms                 |  175815   |         0 |
| udps.session_establishment_request_counter |   24323.9 |         0 |
| id                                         |   22080.7 |         0 |
| dst2src_rst_packets                        |   21013.1 |         0 |
| dst2src_first_seen_ms                      |   20807.4 |         0 |
| dst2src_last_seen_ms                       |   20807.4 |         0 |
| expiration_id                              |   20400.1 |         0 |
| bidirectional_mean_ps                      |   20012.2 |         0 |
| src2dst_duration_ms                        |   17477.2 |         0 |
| bidirectional_duration_ms                  |   17384.9 |         0 |
| src2dst_mean_ps                            |   16613.7 |         0 |
| src2dst_max_ps                             |   16386.9 |         0 |

Important caution:
- Extremely high importance for IP/MAC/timestamps can indicate leakage or collection-time artifacts (not robust in FL).

## 9️⃣ FL-Aware Feature Robustness Analysis
Client splits (as specified) saved in: `fl_simulation/client_splits.json`
Drift metric:
- KS statistic per numeric feature vs global → `fl_simulation/fl_drift_ks.csv`
- Top-25 drift features per client → `fl_simulation/fl_drift_ks_top25_per_client.csv`

Top-5 drift features per client (highest KS):
| client   | feature                     |   ks_stat |
|:---------|:----------------------------|----------:|
| client_1 | bidirectional_first_seen_ms |  0.406682 |
| client_1 | bidirectional_last_seen_ms  |  0.406682 |
| client_1 | src2dst_first_seen_ms       |  0.406682 |
| client_1 | src2dst_last_seen_ms        |  0.406682 |
| client_1 | dst_port                    |  0.247765 |
| client_2 | bidirectional_first_seen_ms |  0.522283 |
| client_2 | bidirectional_last_seen_ms  |  0.522283 |
| client_2 | src2dst_first_seen_ms       |  0.522283 |
| client_2 | src2dst_last_seen_ms        |  0.522283 |
| client_2 | dst2src_first_seen_ms       |  0.293345 |
| client_3 | bidirectional_first_seen_ms |  0.363963 |
| client_3 | bidirectional_last_seen_ms  |  0.363963 |
| client_3 | src2dst_first_seen_ms       |  0.363963 |
| client_3 | src2dst_last_seen_ms        |  0.363963 |
| client_3 | bidirectional_ack_packets   |  0.247391 |
| client_4 | dst2src_first_seen_ms       |  0.399374 |
| client_4 | dst2src_last_seen_ms        |  0.399374 |
| client_4 | dst2src_packets             |  0.370749 |
| client_4 | dst2src_bytes               |  0.370749 |
| client_4 | dst2src_min_ps              |  0.370749 |

How to use:
- Prefer features with low drift across clients (more FL-robust).
- Flag features with high drift (client-specific shortcuts).

## 🔟 Feature Engineering (Domain-Aware)
Implemented engineered features (saved list): `feature_engineering/engineered_features.json`
Examples:
- bytes/packet, bytes/ms, packets/ms
- directional ratios (s2d vs d2s)
- TCP-flag rates (syn/rst/fin per packet)
- UDPS normalized rates (per packet)

Importance after engineering:
- `feature_engineering/mi_numeric_with_engineering.csv`

Top-15 MI after adding engineered features (numeric):
| feature                     |   mutual_info |
|:----------------------------|--------------:|
| src2dst_last_seen_ms        |       1.81332 |
| bidirectional_first_seen_ms |       1.81311 |
| bidirectional_last_seen_ms  |       1.81305 |
| src2dst_first_seen_ms       |       1.81299 |
| fe_byte_ratio_s2d_over_d2s  |       1.37842 |
| bidirectional_bytes         |       1.34043 |
| fe_bytes_per_packet_bi      |       1.33299 |
| bidirectional_mean_ps       |       1.33076 |
| src2dst_bytes               |       1.28817 |
| fe_bytes_per_ms_bi          |       1.26588 |
| bidirectional_min_ps        |       1.18648 |
| fe_bytes_per_packet_s2d     |       1.14419 |
| src2dst_mean_ps             |       1.14332 |
| bidirectional_max_ps        |       1.124   |
| src2dst_max_ps              |       1.11182 |

## 1️⃣1️⃣ Preprocessing Pipeline Design
Recommended (FL-aware) pipeline structure:
- Numeric: median impute + RobustScaler
- Categorical: most-frequent impute + OneHot (handle_unknown='ignore')
- Add explicit missingness indicators for sparse text fields

FL guidance:
- Avoid fitting scalers on global pooled data in real FL; instead fit per-client robust scalers or use federated aggregation of summary stats where possible.
- Keep feature set identical across clients (same columns), even if some classes are absent.

## 1️⃣2️⃣ Final Outputs
All outputs are under this folder:
- `/home/arjit/thesis/EDA_FL/output`

Key files:
- `REPORT.md` (this report)
- `INDEX.csv`
- `missingness_global.csv`, `missingness_by_class.csv`
- `zero_rates_global.csv`, `zero_rates_by_class.csv`
- `constant_features.csv`, `near_constant_features.csv`
- `importance/` (ANOVA, chi2, proxy RF, SHAP)
- `geometry/` (PCA, t-SNE)
- `fl_simulation/` (drift tables)
- `feature_engineering/`
