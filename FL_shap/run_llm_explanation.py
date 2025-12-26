#!/usr/bin/env python3
"""
CLI wrapper to run LLM-based SHAP explanation for a given inference+SHAP output folder.

Usage:
  ./run_llm_explanation.py --out-dir FL_shap/output/run_test_csv2/client_1_0 --llm-api-key <API_KEY> [--topk 10]

Finds the SHAP summary CSV in the output folder and runs the LLM explainer.
"""
import argparse
import os
import sys
from pathlib import Path
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Folder containing SHAP outputs")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama server URL")
    ap.add_argument("--ollama-model", default="qwen2.5:32b-instruct", help="Ollama model name")
    ap.add_argument("--topk", type=int, default=10, help="Number of top features to include")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    # Find SHAP summary CSV
    shap_csvs = list(out_dir.glob("*_shap_summary.csv"))
    if not shap_csvs:
        print(f"No SHAP summary CSV found in {out_dir}")
        sys.exit(1)
    shap_summary = shap_csvs[0]
    print(f"Using SHAP summary: {shap_summary}")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "shap_llm_explainer.py"),
        "--shap-summary",
        str(shap_summary),
        "--ollama-url",
        args.ollama_url,
        "--ollama-model",
        args.ollama_model,
        "--topk",
        str(args.topk),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
