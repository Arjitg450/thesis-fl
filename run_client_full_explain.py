#!/usr/bin/env python3
"""
Run inference + SHAP + natural-language explanation in one command.

Behavior:
- Prints prediction immediately after inference
- Runs LLM explanation asynchronously
- Streams LLM output as soon as it appears
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", help="client_1, client_2, ... (optional if --test-csv provided)")
    ap.add_argument("--index", required=True, type=int, help="0-based index in CSV to explain")
    ap.add_argument("--test-csv", type=Path, help="Path to a CSV to use as test set")
    ap.add_argument("--model-dir", type=Path, default=Path("withoutSlowloris/output/test_central50r"))
    ap.add_argument("--datasets-dir", type=Path, default=Path("withoutSlowloris/preprocessing/datasets"))
    ap.add_argument("--out-dir", type=Path, default=Path("FL_shap/output"))
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    ap.add_argument("--ollama-model", default="qwen2.5:32b-instruct")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.out_dir)

    # ---------------------------------------------------
    # 1. Run inference + SHAP (blocking)
    # ---------------------------------------------------
    inf_cmd = [
        sys.executable,
        "run_client_inference.py",
        "--index", str(args.index),
        "--model-dir", str(model_dir),
        "--out-dir", str(out_dir),
    ]

    if args.test_csv:
        inf_cmd += ["--test-csv", str(args.test_csv.resolve())]
        client_name = Path(args.test_csv).stem
    else:
        if not args.client:
            print("Either --client or --test-csv must be provided")
            return 2
        inf_cmd += ["--client", args.client, "--datasets-dir", str(datasets_dir)]
        client_name = args.client

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        subprocess.run(
            inf_cmd,
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print("Inference + SHAP failed")
        return 3

    # ---------------------------------------------------
    # 2. Read inference output and PRINT IMMEDIATELY
    # ---------------------------------------------------
    run_out = out_dir / f"{client_name}_{args.index}"
    inf_csv = run_out / f"inference_{client_name}_{args.index}.csv"

    pred = "unknown"
    true_val = "unknown"

    if inf_csv.exists():
        try:
            df = pd.read_csv(inf_csv)
            if "pred" in df.columns:
                pred = df["pred"].iloc[0]
            if "true" in df.columns:
                true_val = df["true"].iloc[0]
        except Exception:
            pass

    CLASS_MAP = {
        0: "benign",
        1: "bruteForce",
        2: "ddos_tcp_ddos",
        3: "pfcp_session_deletion",
        4: "pfcp_session_establishment",
        5: "pfcp_session_modification",
    }

    try:
        pred_name = CLASS_MAP.get(int(pred), str(pred))
    except Exception:
        pred_name = str(pred)

    print(f"\nPrediction: {pred} ({pred_name})")
    print(f"True: {true_val}\n")

    # ---------------------------------------------------
    # 3. Start LLM explanation asynchronously
    # ---------------------------------------------------
    llm_cmd = [
        sys.executable,
        "FL_shap/run_llm_explanation.py",
        "--out-dir", str(run_out),
        "--ollama-url", args.ollama_url,
        "--ollama-model", args.ollama_model,
    ]

    print("Generating LLM explanation...\n", flush=True)

    llm_proc = subprocess.Popen(
        llm_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # ---------------------------------------------------
    # 4. Stream LLM output live
    # ---------------------------------------------------
    for line in llm_proc.stdout:
        if line.strip():
            print(line, flush=True)

    llm_proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
