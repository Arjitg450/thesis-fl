# FL_shap — SHAP explanations for federated models

This folder contains a small utility to produce SHAP explanations for a trained Keras model and a chosen data point from your client CSVs.

Files:
- `run_shap.py`: main script. Finds a Keras model in `--model-dir`, reads client CSVs, and writes SHAP outputs (CSV + PNG + optional HTML force plot).
- `requirements.txt`: minimal list of packages used by the script.

Quick example:

```
./FL_shap/run_shap.py \
  --model-dir /home/arjit/thesis/withoutSlowloris/output/flwr_fedadam_50r_with_model \
  --datasets-dir withoutSlowloris/preprocessing/datasets \
  --client client_1 --index 0 \
  --out-dir FL_shap/output/run1
```

Notes:
- The script does not modify any existing files; it only reads models and CSVs and writes explanations to the `--out-dir` you choose.
- If `shap` or other packages are missing, install from `requirements.txt`.

 ./flwrEnv/bin/python run_client_full_explain.py --test-csv test_sets/test_2000.csv --index 20