"""Run SHAP explanations for a saved FL model and a chosen data point.

Usage (example):
  ./FL_shap/run_shap.py \
    --model-dir /path/to/model_output \
    --datasets-dir withoutSlowloris/preprocessing/datasets \
    --client client_1 --index 10 \
    --out-dir FL_shap/output/run1

Notes:
- The script will try to find a Keras model in `--model-dir` (look for .h5 or a SavedModel).
- It reads the CSV for the selected client, uses all columns except `__label` as features.
- Outputs a SHAP summary bar plot and a per-feature CSV with SHAP values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import shap
except Exception as e:  # pragma: no cover - user environment may not have shap
    print("Missing dependency: shap. Install with `pip install shap`.")
    raise

try:
    import matplotlib.pyplot as plt
except Exception:
    print("Missing dependency: matplotlib. Install with `pip install matplotlib`.")
    raise

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    print("Missing dependency: tensorflow/keras. Install with `pip install tensorflow`.")
    raise


LABEL_COL = "__label"


def find_keras_model(path: Path) -> Path | None:
    # Look for .h5 or .keras files first
    for ext in ("*.h5", "*.keras", "saved_model.pb"):
        matches = list(path.rglob(ext))
        if matches:
            # If saved_model.pb matched, return its directory
            m = matches[0]
            if m.name == "saved_model.pb":
                return m.parent
            return m
    return None


def load_model_guess(model_dir: Path):
    p = Path(model_dir).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {p}")

    m = find_keras_model(p)
    if m is None:
        # try common filenames
        for fname in ("model.h5", "model.keras", "best_model.h5"):
            cand = p / fname
            if cand.exists():
                m = cand
                break

    if m is None:
        raise FileNotFoundError(f"No Keras model found in {p}")

    print(f"Loading model from: {m}")
    if m.is_dir():
        return keras.models.load_model(str(m))
    else:
        return keras.models.load_model(str(m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="Directory containing saved model")
    ap.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("withoutSlowloris/preprocessing/datasets"),
        help="Directory with client_*.csv",
    )
    ap.add_argument("--client", type=str, default="client_1", help="Which client CSV to use (client_1)")
    ap.add_argument("--input-csv", type=Path, help="Path to a single CSV file to explain (overrides --datasets-dir and --client)")
    ap.add_argument("--index", type=int, default=0, help="Row index (0-based) in the client CSV to explain")
    ap.add_argument("--out-dir", type=Path, default=Path("FL_shap/output"), help="Output directory")
    ap.add_argument("--background-size", type=int, default=100, help="Background sample size for SHAP explainer")
    ap.add_argument("--fast", action="store_true", help="Use fast perturbation-based explainer (no shap/numba)")
    ap.add_argument("--force-cpu", action="store_true", help="Disable GPU for reproducible SHAP behaviour")
    args = ap.parse_args()

    if args.force_cpu:
        # limit TF to CPU
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model = load_model_guess(args.model_dir)

    # Support either an explicit input CSV or a client CSV from a datasets directory
    if args.input_csv:
        csv_path = args.input_csv.resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        client_name = csv_path.stem
        # Remove helper columns added by test-set generators
        for extra_col in ("original_file", "original_index"):
            if extra_col in df.columns:
                df = df.drop(columns=[extra_col])
    else:
        datasets_dir = args.datasets_dir.resolve()
        csv_path = datasets_dir / f"{args.client}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Client CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        client_name = args.client

    if LABEL_COL in df.columns:
        X = df.drop(columns=[LABEL_COL])
        y = df[LABEL_COL]
    else:
        X = df.copy()
        y = None

    # Convert to numeric matrix (model expects numpy float32)
    feature_names = list(X.columns)
    X_mat = X.values.astype(np.float32)

    # select instance
    if args.index < 0 or args.index >= X_mat.shape[0]:
        raise IndexError(f"Index {args.index} out of range for {args.client} (n={X_mat.shape[0]})")

    x_instance = X_mat[args.index: args.index + 1]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose background sample
    bg_n = min(args.background_size, max(1, X_mat.shape[0] // 10))
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(X_mat.shape[0], size=bg_n, replace=False)
    background = X_mat[bg_idx]

    # Fast perturbation-based explainer (safe, no shap/numba)
    if args.fast:
        print("Running fast perturbation explainer (no SHAP)...")
        # predict on instance
        probs = model.predict(x_instance, verbose=0)[0]
        pred_class = int(probs.argmax())
        base_prob = float(probs[pred_class])

        # background mean per feature
        bg_mean = background.mean(axis=0)

        contribs = []
        raw_vals = []
        for j in range(X_mat.shape[1]):
            x_mod = x_instance.copy()
            x_mod[0, j] = bg_mean[j]
            probs_mod = model.predict(x_mod, verbose=0)[0]
            contrib = float(probs[pred_class] - probs_mod[pred_class])
            contribs.append(abs(contrib))
            raw_vals.append(contrib)

        shap_df = pd.DataFrame({"feature": feature_names, "shap_abs_mean": contribs})
        shap_df = shap_df.sort_values(by="shap_abs_mean", ascending=False)

        out_prefix = out_dir / f"{client_name}_{args.index}"
        def out_with(sfx: str) -> Path:
            return out_prefix.parent / f"{out_prefix.name}{sfx}"

        shap_df.to_csv(out_with("_shap_summary.csv"), index=False)
        raw_df = pd.DataFrame([raw_vals], columns=feature_names)
        raw_df.to_csv(out_with("_shap_raw.csv"), index=False)

        # Plot
        plt.figure(figsize=(8, min(6, 0.25 * len(feature_names) + 2)))
        topk = min(30, len(feature_names))
        sns = __import__("seaborn")
        sns.barplot(data=shap_df.head(topk), x="shap_abs_mean", y="feature", palette="viridis")
        plt.title(f"Perturbation importance — {args.client}[{args.index}]")
        plt.xlabel("|Delta prob| for predicted class")
        plt.tight_layout()
        plt.savefig(out_with("_shap_bar.png"), dpi=200)
        plt.close()

        print(f"Wrote SHAP (approx) outputs to: {out_dir}")
        return 0

    # Choose SHAP explainer - use shap.Explainer which autodetects
    print("Creating SHAP explainer (this may take a moment)...")
    # For Keras models, pass a wrapper function
    def predict_fn(x: np.ndarray) -> np.ndarray:
        # model.predict returns probabilities for classification
        return model.predict(x, verbose=0)

    try:
        import shap
        try:
            explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
        except Exception as exc:
            warnings.warn(f"shap.Explainer failed, falling back to KernelExplainer: {exc}")
            explainer = shap.KernelExplainer(predict_fn, background)
    except Exception as exc:
        warnings.warn(f"Could not import shap/explainers: {exc}; falling back to fast explainer.")
        # fallback to fast perturbation explainer
        args.fast = True
        return main()

    print(f"Explaining instance index {args.index} from {csv_path}")
    shap_values = explainer(x_instance)

    # shap_values can be shap.Explanation object. For classifier with K classes, shap_values.values shape may be (K, 1, F)
    # We'll save per-feature absolute importance and raw values.
    out_prefix = out_dir / f"{client_name}_{args.index}"

    # Save raw shap array(s) and a CSV summarizing
    try:
        vals = shap_values.values
    except Exception:
        vals = np.array(shap_values)

    # If multi-output or permutation-style explainer, handle shapes robustly.
    # Goal: collapse all non-feature axes by mean(abs) to get per-feature importances
    # and produce a 2D raw_vals shaped (rows, features) for CSV output.
    if vals.ndim == 3:
        # Detect which axis corresponds to features (match by length)
        feat_axis = None
        for ax in range(3):
            if vals.shape[ax] == len(feature_names):
                feat_axis = ax
                break
        if feat_axis is None:
            # fallback assume last axis
            feat_axis = 2

        # Move feature axis to last for consistent processing
        if feat_axis != 2:
            vals_reordered = np.moveaxis(vals, feat_axis, -1)
        else:
            vals_reordered = vals

        # Compute mean absolute importance across all non-feature axes
        axes_to_mean = tuple(range(vals_reordered.ndim - 1))
        vals_mean = np.mean(np.abs(vals_reordered), axis=axes_to_mean)

        # Flatten the leading axes into rows x features for raw CSV
        raw_vals = vals_reordered.reshape(-1, vals_reordered.shape[-1])
    elif vals.ndim == 2:
        # (samples, features) or (1, features)
        vals_mean = np.abs(vals).mean(axis=0)
        raw_vals = vals
    else:
        # 1D - treat as single feature vector
        vals_mean = np.abs(vals).mean(axis=0)
        raw_vals = np.atleast_2d(vals)

    shap_df = pd.DataFrame({"feature": feature_names, "shap_abs_mean": vals_mean})
    shap_df = shap_df.sort_values(by="shap_abs_mean", ascending=False)
    # Helper to append suffix to out_prefix name
    def out_with(sfx: str) -> Path:
        return out_prefix.parent / f"{out_prefix.name}{sfx}"

    shap_df.to_csv(out_with("_shap_summary.csv"), index=False)

    # Save raw shap values per feature
    raw_df = pd.DataFrame(raw_vals.reshape(-1, len(feature_names)), columns=feature_names)
    raw_df.to_csv(out_with("_shap_raw.csv"), index=False)

    # Plot horizontal bar of top features
    plt.figure(figsize=(8, min(6, 0.25 * len(feature_names) + 2)))
    topk = min(30, len(feature_names))
    sns = __import__("seaborn")
    sns.barplot(data=shap_df.head(topk), x="shap_abs_mean", y="feature", palette="viridis")
    plt.title(f"SHAP absolute-mean importance — {args.client}[{args.index}]")
    plt.xlabel("Mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(out_with("_shap_bar.png"), dpi=200)
    plt.close()

    # Also produce a force plot saved as HTML if possible
    try:
        fp_html = out_with("_shap_force.html")
        shap.plots.force(shap_values[0], matplotlib=False, show=False)
        # shap.plots.force returns a JS/HTML object when matplotlib=False; save via shap.save_html
        shap.save_html(str(fp_html), shap.plots.force(shap_values[0]))
    except Exception as exc:  # not critical
        warnings.warn(f"Could not produce force plot HTML: {exc}")

    print(f"Wrote SHAP outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
