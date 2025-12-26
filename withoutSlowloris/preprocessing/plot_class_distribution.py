"""Plot overall class distribution (aggregated across all clients).

Reads:
  withoutSlowloris/preprocessing/datasets/client_*.csv

Writes:
  withoutSlowloris/output/class_distribution/

Outputs:
- class_counts.csv
- class_percent.csv
- class_counts.png
- class_percent.png

Labels are expected in column '__label'.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABEL_COL = "__label"


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
        default=Path("withoutSlowloris/output/class_distribution"),
        help="Output directory",
    )
    args = ap.parse_args()

    datasets_dir = args.datasets_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_series = []
    csvs = sorted(datasets_dir.glob("client_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No client CSVs found in {datasets_dir}")

    for csv_path in csvs:
        s = pd.read_csv(csv_path, usecols=[LABEL_COL])[LABEL_COL].astype(str)
        label_series.append(s)

    y = pd.concat(label_series, ignore_index=True)
    counts = y.value_counts().sort_index()
    total = int(counts.sum())

    counts_df = counts.rename_axis("class").reset_index(name="count")
    counts_df.to_csv(out_dir / "class_counts.csv", index=False)

    percent_df = counts_df.copy()
    percent_df["percent"] = (percent_df["count"] / total * 100.0) if total > 0 else 0.0
    percent_df.to_csv(out_dir / "class_percent.csv", index=False)

    sns.set_theme(style="whitegrid")

    # Counts plot
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=counts_df, x="class", y="count", color=sns.color_palette()[0], errorbar=None)
    plt.title(f"Overall class distribution (counts) — total n={total}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")

    for i, row in counts_df.iterrows():
        ax.text(i, row["count"], f"{int(row['count'])}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "class_counts.png", dpi=200)
    plt.close()

    # Percent plot
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=percent_df, x="class", y="percent", color=sns.color_palette()[1], errorbar=None)
    plt.title(f"Overall class distribution (%) — total n={total}")
    plt.xlabel("Class")
    plt.ylabel("Percent")
    plt.xticks(rotation=30, ha="right")

    for i, row in percent_df.iterrows():
        ax.text(i, row["percent"], f"{row['percent']:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "class_percent.png", dpi=200)
    plt.close()

    print(f"Wrote plots + CSVs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
