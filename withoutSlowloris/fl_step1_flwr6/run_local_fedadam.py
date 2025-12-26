"""Local runner for the 6-class / 3-client (without ddos_slowloris) variant.

Starts:
- one Flower server
- three Flower clients

Writes outputs under out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from .common import label_names, per_class_from_confusion
except ImportError:  # pragma: no cover
    from withoutSlowloris.fl_step1_flwr6.common import label_names, per_class_from_confusion


def compute_global_class_weights(datasets_dir: Path, label_col: str = "__label") -> dict:
    # Uses only labels, so safe to compute centrally.
    counts = {}
    for p in sorted(datasets_dir.glob("client_*.csv")):
        df = pd.read_csv(p, usecols=[label_col])
        vc = df[label_col].value_counts(dropna=False)
        for k, v in vc.items():
            counts[str(k)] = counts.get(str(k), 0) + int(v)

    labels = sorted([k for k in counts.keys() if k != "nan"])
    total = sum(counts[l] for l in labels)
    # simple inverse-frequency weight: w_c = total / (K * n_c)
    K = len(labels)
    weights = {l: (total / (K * counts[l])) for l in labels}
    return {"label_weights": weights, "label_counts": counts}


def run_proc(args, env=None):
    return subprocess.Popen(args, env=env)


def aggregate_client_diagnostics(out_dir: Path) -> bool:
    names = label_names()
    cm_total = np.zeros((len(names), len(names)), dtype=np.int64)
    found_any = False

    for client_name in ["client_1", "client_2", "client_3"]:
        cm_path = out_dir / f"confusion_matrix_{client_name}.csv"
        if not cm_path.exists():
            continue
        df = pd.read_csv(cm_path, index_col=0)
        df = df.reindex(index=names, columns=names).fillna(0)
        cm_total += df.to_numpy(dtype=np.int64)
        found_any = True

    if not found_any:
        return False

    pd.DataFrame(cm_total, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv")

    per_df = per_class_from_confusion(cm_total, class_names=names)
    per_df = per_df.rename(columns={"label": "class"})
    per_df = per_df[["class", "support", "precision", "recall", "f1"]]
    per_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    # Also aggregate central-per-client diagnostics if present
    cm_central_total = np.zeros((len(names), len(names)), dtype=np.int64)
    found_central = False
    for client_name in ["client_1", "client_2", "client_3"]:
        cm_path = out_dir / f"confusion_matrix_central_{client_name}.csv"
        if not cm_path.exists():
            continue
        df = pd.read_csv(cm_path, index_col=0)
        df = df.reindex(index=names, columns=names).fillna(0)
        cm_central_total += df.to_numpy(dtype=np.int64)
        found_central = True

    if found_central:
        pd.DataFrame(cm_central_total, index=names, columns=names).to_csv(out_dir / "confusion_matrix_central.csv")
        per_df_c = per_class_from_confusion(cm_central_total, class_names=names)
        per_df_c = per_df_c.rename(columns={"label": "class"})
        per_df_c = per_df_c[["class", "support", "precision", "recall", "f1"]]
        per_df_c.to_csv(out_dir / "per_class_metrics_central.csv", index=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc-dir", type=Path, default=Path("withoutSlowloris/preprocessing"))
    ap.add_argument("--out-dir", type=Path, default=Path("withoutSlowloris/output/flwr_fedadam"))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--central-test-csv", type=Path, default=None,
                    help="Optional central held-out test CSV (evaluated each round by clients)")

    ap.add_argument("--server-lr", type=float, default=0.01)
    ap.add_argument("--client-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--server", type=str, default="127.0.0.1:8085")
    ap.add_argument("--no-class-weights", action="store_true", default=False)

    # Optional W&B logging (API key must be provided via WANDB_API_KEY or `wandb login`)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-run-name", type=str, default=None)

    args = ap.parse_args()

    preproc_dir = args.preproc_dir.resolve()
    datasets_dir = preproc_dir / "datasets"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # write global class weights
    if not args.no_class_weights:
        cw = compute_global_class_weights(datasets_dir)
        with open(out_dir / "global_class_weights.json", "w", encoding="utf-8") as f:
            json.dump(cw, f, indent=2)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    server_cmd = [
        sys.executable,
        "-m",
        "withoutSlowloris.fl_step1_flwr6.server_app",
        "--server",
        args.server,
        "--rounds",
        str(args.rounds),
        "--min-clients",
        "3",
        "--out-dir",
        str(out_dir),
        "--preproc-dir",
        str(preproc_dir),
        "--server-lr",
        str(args.server_lr),
        "--eta-l",
        str(args.client_lr),
        "--seed",
        str(args.seed),
    ]

    # If provided, pass central test csv to server so it can include path in evaluate config
    if args.central_test_csv:
        server_cmd += ["--central-test-csv", str(args.central_test_csv)]

    client_cmd_base = [
        sys.executable,
        "-m",
        "withoutSlowloris.fl_step1_flwr6.client_app",
        "--server",
        args.server,
        "--preproc-dir",
        str(preproc_dir),
        "--lr",
        str(args.client_lr),
        "--seed",
        str(args.seed),
    ]

    if not args.no_class_weights:
        client_cmd_base += ["--class-weights-json", str(out_dir / "global_class_weights.json")]

    # Pass central test csv to clients too so they can write per-client central diagnostics
    if args.central_test_csv:
        client_cmd_base += ["--central-test-csv", str(args.central_test_csv)]

    print("Starting server:")
    print(" ".join(server_cmd))
    server = run_proc(server_cmd, env=env)

    # Give server time to bind
    time.sleep(2.0)

    clients = []
    for cid in ["1", "2", "3"]:
        client_name = f"client_{cid}"
        cmd = client_cmd_base + ["--client", client_name, "--out-dir", str(out_dir)]
        print(f"Starting {client_name}:")
        print(" ".join(cmd))
        clients.append(run_proc(cmd, env=env))
        time.sleep(0.4)

    # wait for clients
    rc_clients = [p.wait() for p in clients]
    rc_server = server.wait()

    aggregate_client_diagnostics(out_dir)

    if args.wandb_project:
        _maybe_log_to_wandb(
            out_dir=out_dir,
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity) if args.wandb_entity else None),
            run_name=(str(args.wandb_run_name) if args.wandb_run_name else None),
            config={
                "variant": "withoutSlowloris",
                "rounds": int(args.rounds),
                "server": str(args.server),
                "server_lr": float(args.server_lr),
                "client_lr": float(args.client_lr),
                "seed": int(args.seed),
                "use_class_weights": (not bool(args.no_class_weights)),
            },
        )

    print(f"Clients exit codes: {rc_clients}; server exit code: {rc_server}")
    print(f"Outputs: {out_dir}")

    return 0 if (rc_server == 0 and all(rc == 0 for rc in rc_clients)) else 1


def _maybe_log_to_wandb(
    out_dir: Path,
    project: str,
    entity: str | None,
    run_name: str | None,
    config: dict[str, Any],
) -> None:
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"W&B requested but wandb import failed: {e}")
        return

    try:
        run = wandb.init(project=project, entity=entity, name=run_name, config=config, reinit=True)
    except Exception as e:
        print(f"W&B init failed (check WANDB_API_KEY / login): {e}")
        return

    try:
        metrics_path = out_dir / "metrics_by_round.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            step_col = "round" if "round" in df.columns else ("server_round" if "server_round" in df.columns else None)
            for _, row in df.iterrows():
                payload = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in row.to_dict().items()}
                if step_col and step_col in payload:
                    try:
                        step_val = int(payload[step_col])
                    except Exception:
                        step_val = None
                    if step_val is not None:
                        wandb.log(payload, step=step_val)
                    else:
                        wandb.log(payload)
                else:
                    wandb.log(payload)

        # Upload final artifacts if present
        for fname in [
            "confusion_matrix.csv",
            "per_class_metrics.csv",
            "global_class_weights.json",
            "diagnostics_status.json",
            "sent_evaluate_config.json",
        ]:
            p = out_dir / fname
            if p.exists():
                wandb.save(str(p), base_path=str(out_dir))
    finally:
        run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
