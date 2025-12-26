"""Plot overall data distribution (sample counts) per client.

This is *not* label-wise; it only shows how many rows each client has.

Reads:
  withoutSlowloris/preprocessing/datasets/client_*.csv

Writes:
  withoutSlowloris/output/data_distribution/

Outputs:
- client_sample_counts.csv
- client_sample_counts.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABEL_COL = "__label"  # not used beyond quick validation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("withoutSlowloris/preprocessing/datasets"),
        help="Directory containing client_*.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("withoutSlowloris/output/data_distribution"),
        help="Output directory",
    )
    args = ap.parse_args()

    datasets_dir = args.datasets_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(datasets_dir.glob("client_*.csv")):
        # Fast row count: count lines minus header
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            n_lines = sum(1 for _ in f)
        n = max(0, n_lines - 1)

        # Optional sanity: ensure label column exists
        header = pd.read_csv(csv_path, nrows=0).columns
        if LABEL_COL not in header:
            raise ValueError(f"Expected column '{LABEL_COL}' in {csv_path}")

        rows.append({"client": csv_path.stem, "n": int(n)})

    if not rows:
        raise FileNotFoundError(f"No client CSVs found in {datasets_dir}")

    df = pd.DataFrame(rows).sort_values("client").reset_index(drop=True)
    total = int(df["n"].sum())
    df["percent"] = (df["n"] / total * 100.0) if total > 0 else 0.0

    df.to_csv(out_dir / "client_sample_counts.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x="client", y="n", color=sns.color_palette()[0], errorbar=None)
    plt.title("Data distribution by client (sample count)")
    plt.xlabel("Client")
    plt.ylabel("Number of samples")

    # Annotate bars with n and %
    for i, row in df.iterrows():
        ax.text(i, row["n"], f"{int(row['n'])}\n({row['percent']:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "client_sample_counts.png", dpi=200)
    plt.close()

    print(f"Wrote: {out_dir / 'client_sample_counts.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
