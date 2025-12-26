"""Plot client-wise label distributions for the withoutSlowloris 3-client dataset.

Reads CSVs from:
  withoutSlowloris/preprocessing/datasets/client_*.csv

Writes to:
  withoutSlowloris/output/label_distribution/

Outputs:
- label_counts_by_client.csv
- label_percent_by_client.csv
- bar_counts_by_client.png
- bar_percent_by_client.png
- stacked_percent_by_client.png

Notes:
- Labels are expected in column '__label' and are strings (e.g., 'benign').
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABEL_COL = "__label"


def _read_counts(datasets_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(datasets_dir.glob("client_*.csv")):
        client_name = csv_path.stem
        s = pd.read_csv(csv_path, usecols=[LABEL_COL])[LABEL_COL].astype(str)
        vc = s.value_counts()
        for label, count in vc.items():
            rows.append({"client": client_name, "class": label, "count": int(count)})

    if not rows:
        raise FileNotFoundError(f"No client CSVs found in {datasets_dir}")

    df = pd.DataFrame(rows)
    # Ensure deterministic ordering
    df["client"] = df["client"].astype(str)
    df["class"] = df["class"].astype(str)
    return df


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
        default=Path("withoutSlowloris/output/label_distribution"),
        help="Output directory for plots and CSVs",
    )
    args = ap.parse_args()

    datasets_dir = args.datasets_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_counts(datasets_dir)

    # Build full class/client grid to keep missing classes visible as zeros
    clients = sorted(df["client"].unique().tolist())
    classes = sorted(df["class"].unique().tolist())

    grid = (
        pd.MultiIndex.from_product([clients, classes], names=["client", "class"])\
        .to_frame(index=False)
        .merge(df, on=["client", "class"], how="left")
        .fillna({"count": 0})
    )
    grid["count"] = grid["count"].astype(int)

    counts_pivot = grid.pivot(index="class", columns="client", values="count").loc[classes, clients]
    counts_pivot.to_csv(out_dir / "label_counts_by_client.csv")

    # Totals summary
    client_totals = counts_pivot.sum(axis=0).astype(int)
    class_totals = counts_pivot.sum(axis=1).astype(int)
    totals_df = (
        pd.DataFrame({"client": client_totals.index, "n": client_totals.values})
        .sort_values("client")
        .reset_index(drop=True)
    )
    totals_df.to_csv(out_dir / "client_totals.csv", index=False)
    class_totals.to_frame(name="n").to_csv(out_dir / "class_totals.csv")

    percent_pivot = (counts_pivot / counts_pivot.sum(axis=0).replace({0: pd.NA})) * 100.0
    percent_pivot = percent_pivot.fillna(0.0)
    percent_pivot.to_csv(out_dir / "label_percent_by_client.csv")

    sns.set_theme(style="whitegrid")

    # 1) Grouped bar chart (counts)
    plt.figure(figsize=(12, 5))
    plot_df = grid.copy()
    sns.barplot(data=plot_df, x="class", y="count", hue="client", errorbar=None)
    plt.title("Class distribution by client (counts)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_counts_by_client.png", dpi=200)
    plt.close()

    # 2) Grouped bar chart (percent)
    plt.figure(figsize=(12, 5))
    percent_long = (
        percent_pivot.reset_index()
        .melt(id_vars=["class"], var_name="client", value_name="percent")
        .sort_values(["class", "client"])
    )
    sns.barplot(data=percent_long, x="class", y="percent", hue="client", errorbar=None)
    plt.title("Class distribution by client (%)")
    plt.xlabel("Class")
    plt.ylabel("Percent")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_percent_by_client.png", dpi=200)
    plt.close()

    # 3) Stacked bar chart (percent per client)
    plt.figure(figsize=(10, 5))
    x_labels = [f"{c}\n(n={int(client_totals[c])})" for c in clients]
    bottom = pd.Series([0.0] * len(clients), index=clients)
    for cls in classes:
        vals = percent_pivot.loc[cls, clients]
        plt.bar(x_labels, vals, bottom=bottom, label=cls)
        bottom = bottom + vals
    plt.title("Class distribution by client (stacked %)")
    plt.xlabel("Client")
    plt.ylabel("Percent")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "stacked_percent_by_client.png", dpi=200)
    plt.close()

    print(f"Wrote plots + CSVs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
