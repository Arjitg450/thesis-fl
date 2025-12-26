# Test sets and index mapping

This folder contains example indices you can use with `run_client_full_explain.py`.

Example usage:
```
python run_client_full_explain.py --test-csv test_sets/<client_csv> --index <index>
```

Indices found:
- benign: file `withoutSlowloris/preprocessing/datasets/client_1.csv`, index `0`
- bruteForce: file `withoutSlowloris/preprocessing/datasets/client_1.csv`, index `21060`
- ddos_tcp_ddos: file `withoutSlowloris/preprocessing/datasets/client_2.csv`, index `21060`
- pfcp_session_deletion: file `withoutSlowloris/preprocessing/datasets/client_2.csv`, index `66093`
- pfcp_session_establishment: file `withoutSlowloris/preprocessing/datasets/client_3.csv`, index `36356`
- pfcp_session_modification: file `withoutSlowloris/preprocessing/datasets/client_3.csv`, index `43944`
