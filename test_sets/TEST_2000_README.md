# Stratified 2000-row test set

This file was created by sampling rows from the client CSVs under `withoutSlowloris/preprocessing/datasets`. It contains 2000 rows stratified across all labels found.

Usage:
```bash
python run_client_full_explain.py --test-csv test_sets/test_2000.csv --index <row_index>
```

Class distribution in `test_2000.csv`:

- benign: 334 rows (source files: unique count 3)
- ddos_tcp_ddos: 334 rows (source files: unique count 1)
- bruteForce: 333 rows (source files: unique count 1)
- pfcp_session_deletion: 333 rows (source files: unique count 2)
- pfcp_session_modification: 333 rows (source files: unique count 1)
- pfcp_session_establishment: 333 rows (source files: unique count 1)

Each row includes two extra columns at the end: `original_file` and `original_index` pointing to the source CSV and row index.
