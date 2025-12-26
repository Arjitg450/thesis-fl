#!/usr/bin/env python3
"""Wrapper to run inference + SHAP for a saved client Keras model.

Usage examples:
  ./run_client_inference.py --client client_1 --index 0
  ./run_client_inference.py --client client_2 --index 10 --model-dir withoutSlowloris/output/flwr_fedadam_50r_with_model

The script will:
 - load the client model (looks for `client_model_{client}.keras` inside `--model-dir`)
 - run a single-row inference for the chosen `--index` from the client's CSV
 - write `inference_<client>_<index>.csv` to `--out-dir`
 - invoke `FL_shap/run_shap.py` to compute and save SHAP outputs to `--out-dir`
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf


def find_client_model(model_dir: Path, client: str) -> Path:
    cand = model_dir / f"client_model_{client}.keras"
    if cand.exists():
        return cand
    # fallback: any .keras or .h5 in dir
    for ext in ("*.keras", "*.h5"):
        matches = list(model_dir.glob(ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No client model found for {client} in {model_dir}")


def write_inference_csv(out_dir: Path, client: str, index: int, pred: int, probs: np.ndarray, true_val):
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"index": int(index), "pred": int(pred), "true": (str(true_val) if true_val is not None else "")}
    for i, p in enumerate(map(float, probs)):
        row[f"p_{i}"] = p
    pd.DataFrame([row]).to_csv(out_dir / f"inference_{client}_{index}.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=False, help="client_1, client_2, ...")
    ap.add_argument("--index", required=True, type=int, help="0-based row index in client CSV")
    ap.add_argument("--model-dir", type=Path, default=Path("withoutSlowloris/output/flwr_fedadam_50r_with_model"))
    ap.add_argument("--datasets-dir", type=Path, default=Path("withoutSlowloris/preprocessing/datasets"))
    ap.add_argument("--test-csv", type=Path, help="Path to a CSV file to use as the test set; index will refer to this file")
    ap.add_argument("--out-dir", type=Path, default=Path("FL_shap/output"))
    args = ap.parse_args()

    model_dir: Path = args.model_dir.resolve()
    datasets_dir: Path = args.datasets_dir.resolve()

    # Allow using an arbitrary test CSV (useful for test-index based runs)
    if args.test_csv:
        csv_path = args.test_csv.resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        client_name_for_shap = csv_path.stem
        datasets_dir_for_shap = csv_path.parent
        # Drop helper columns if present (e.g., original_file/original_index added by test set generator)
        for extra_col in ("original_file", "original_index"):
            if extra_col in df.columns:
                df = df.drop(columns=[extra_col])
    else:
        if not args.client:
            raise SystemExit("Either --client or --test-csv must be provided")
        csv_path = datasets_dir / f"{args.client}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Client CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        client_name_for_shap = args.client
        datasets_dir_for_shap = datasets_dir

    # Determine which model to load: prefer client-specific model when --client provided,
    # otherwise load any model found in model_dir.
    if args.client:
        model_path = find_client_model(model_dir, args.client)
        print(f"Loading Keras model: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
    else:
        # find any model file in model_dir
        matches = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
        if matches:
            model_path = matches[0]
            print(f"Loading Keras model: {model_path}")
            model = tf.keras.models.load_model(str(model_path))
        else:
            raise FileNotFoundError(f"No Keras model files found in {model_dir}")

    out_dir: Path = args.out_dir.resolve() / f"{client_name_for_shap}_{args.index}"

    print(f"Model dir: {model_dir}")
    print(f"Datasets dir: {datasets_dir}")
    print(f"Out dir: {out_dir}")
    if "__label" in df.columns:
        X = df.drop(columns=["__label"]).copy()
        true_vals = df["__label"].tolist()
    else:
        X = df.copy()
        true_vals = [None] * len(df)

    if args.index < 0 or args.index >= len(X):
        raise IndexError(f"Index {args.index} out of range for {csv_path} (n={len(X)})")

    x = X.iloc[[args.index]].values.astype(np.float32)
    probs = model.predict(x, verbose=0)[0]
    pred = int(int(np.argmax(probs)))
    true_val = true_vals[args.index]

    print(f"Index {args.index} -> pred={pred}, true={true_val}")
    write_inference_csv(out_dir, client_name_for_shap, args.index, pred, probs, true_val)
    print(f"Wrote inference CSV to: {out_dir / f'inference_{client_name_for_shap}_{args.index}.csv'}")

    # Run SHAP script (external) to compute explanations and plots. If a test CSV
    # was provided, pass it directly via --input-csv so the SHAP script uses it.
    cmd = [sys.executable, "FL_shap/run_shap.py", "--model-dir", str(model_dir), "--index", str(args.index), "--out-dir", str(out_dir), "--fast"]
    if args.test_csv:
        cmd += ["--input-csv", str(csv_path)]
    else:
        cmd += ["--datasets-dir", str(datasets_dir_for_shap), "--client", client_name_for_shap]
    print("Running SHAP explainer via:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SHAP explainer failed: {e}")
        return 2

    print(f"SHAP outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
