# EDA_FL

FL-aware, multi-class EDA + preprocessing analysis for `stage2Data/`.

Outputs go to: `/home/arjit/thesis/EDA_FL/output`

## Run

```bash
/home/arjit/thesis/.venv/bin/python /home/arjit/thesis/EDA_FL/run_full_eda_fl.py \
  --data-dir /home/arjit/thesis/stage2Data \
  --out-dir /home/arjit/thesis/EDA_FL/output
```

## Notes

- Labels are constructed as 7 classes from folder structure:
  - benign
  - bruteForce
  - ddos_slowloris
  - ddos_tcp_ddos
  - pfcp_session_deletion
  - pfcp_session_establishment
  - pfcp_session_modification
