# Single-file test set

This CSV contains one example row per class (order preserved below). Use it with `run_client_full_explain.py --test-csv test_sets/test_single_per_class.csv --index <index>`

Rows in `test_single_per_class.csv`:

- Row 0: class `benign` from `withoutSlowloris/preprocessing/datasets/client_1.csv` (original index 0)
- Row 1: class `bruteForce` from `withoutSlowloris/preprocessing/datasets/client_1.csv` (original index 21060)
- Row 2: class `ddos_tcp_ddos` from `withoutSlowloris/preprocessing/datasets/client_2.csv` (original index 21060)
- Row 3: class `pfcp_session_deletion` from `withoutSlowloris/preprocessing/datasets/client_2.csv` (original index 66093)
- Row 4: class `pfcp_session_establishment` from `withoutSlowloris/preprocessing/datasets/client_3.csv` (original index 36356)
- Row 5: class `pfcp_session_modification` from `withoutSlowloris/preprocessing/datasets/client_3.csv` (original index 43944)
