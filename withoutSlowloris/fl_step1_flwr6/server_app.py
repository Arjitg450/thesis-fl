"""Flower server for the 6-class (without ddos_slowloris) variant.

We use Flower's FedAdam (server-side adaptive optimization) by default.
This mirrors what worked best in the 7-class setup.

Outputs:
- metrics_by_round.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from .common import ModelConfig, create_mlp, label_names, load_schema, per_class_from_confusion


def weighted_mean(metric_name: str, results: List[Tuple[int, Dict[str, float]]]) -> float:
    vals = []
    wts = []
    for n, m in results:
        if metric_name in m:
            vals.append(float(m[metric_name]))
            wts.append(int(n))
    if not wts or sum(wts) == 0:
        return float("nan")
    v = np.asarray(vals, dtype=np.float64)
    w = np.asarray(wts, dtype=np.float64)
    return float((v * w).sum() / w.sum())


class TqdmFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, rounds: int, out_dir: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rounds = int(rounds)
        self.out_dir = out_dir
        self.pbar = tqdm(total=self.rounds, desc="FL Rounds", dynamic_ncols=True)
        self.history_rows: List[Dict] = []

    def configure_evaluate(self, server_round, parameters, client_manager):  # type: ignore[override]
        # Ensure the client always receives round metadata.
        if self.fraction_evaluate == 0.0:
            return []

        config: Dict[str, fl.common.Scalar] = {
            "server_round": int(server_round),
            "is_final": int(int(server_round) == int(self.rounds)),
        }

        if self.on_evaluate_config_fn is not None:
            # Merge any user-provided config
            try:
                config.update(self.on_evaluate_config_fn(int(server_round)))
            except Exception:
                pass

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        evaluate_ins = fl.common.EvaluateIns(parameters, config)

        # Debug: record the config we intend to send (last one wins).
        try:
            (self.out_dir / "sent_evaluate_config.json").write_text(
                json.dumps({k: config[k] for k in sorted(config.keys())}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        metric_rows = []
        cms_local: List[np.ndarray] = []
        cms_central: List[np.ndarray] = []
        status_rows = []
        for client_proxy, r in results:
            numeric_metrics: Dict[str, float] = {}
            for k, v in r.metrics.items():
                if isinstance(v, (int, float, bool)):
                    numeric_metrics[k] = float(v)
            metric_rows.append((r.num_examples, numeric_metrics))

            # Record status and whether client included local/central diagnostics
            details = r.metrics.get("details_json")
            central_details = r.metrics.get("central_details_json")
            status_rows.append(
                {
                    "cid": getattr(client_proxy, "cid", ""),
                    "has_details_json": bool(isinstance(details, str) and len(details) > 0),
                    "has_central_details_json": bool(isinstance(central_details, str) and len(central_details) > 0),
                }
            )

            # Collect confusion matrices from both local diagnostics and central diagnostics
            if isinstance(details, str) and details:
                try:
                    payload = json.loads(details)
                    cm = np.asarray(payload.get("confusion_matrix"), dtype=np.int64)
                    if cm.ndim == 2:
                        cms_local.append(cm)
                except Exception:
                    pass
            if isinstance(central_details, str) and central_details:
                try:
                    payload = json.loads(central_details)
                    cm = np.asarray(payload.get("confusion_matrix"), dtype=np.int64)
                    if cm.ndim == 2:
                        cms_central.append(cm)
                except Exception:
                    pass

        f1 = weighted_mean("macro_f1", metric_rows)
        prec = weighted_mean("macro_precision", metric_rows)
        rec = weighted_mean("macro_recall", metric_rows)

        # Also compute aggregated central-test metrics if clients provided them
        central_f1 = weighted_mean("central_macro_f1", metric_rows)
        central_prec = weighted_mean("central_macro_precision", metric_rows)
        central_rec = weighted_mean("central_macro_recall", metric_rows)

        self.pbar.update(1)
        self.pbar.set_postfix({"F1": f"{f1:.3f}", "P": f"{prec:.3f}", "R": f"{rec:.3f}"})

        row = {
            "round": int(server_round),
            "macro_f1_weighted": f1,
            "macro_precision_weighted": prec,
            "macro_recall_weighted": rec,
        }
        # Add central aggregated metrics if present (NaN otherwise)
        row["central_macro_f1_weighted"] = central_f1
        row["central_macro_precision_weighted"] = central_prec
        row["central_macro_recall_weighted"] = central_rec
        self.history_rows.append(row)

        if server_round == self.rounds:
            (self.out_dir / "diagnostics_status.json").write_text(
                json.dumps(
                    {
                        "round": int(server_round),
                        "num_clients": int(len(results)),
                        "num_local_details_received": int(sum(1 for r in status_rows if r["has_details_json"])),
                        "num_central_details_received": int(sum(1 for r in status_rows if r["has_central_details_json"])),
                        "status_by_client": status_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            names = label_names()
            # Aggregate local diagnostics if present
            if cms_local:
                cm_sum_local = np.sum(np.stack(cms_local, axis=0), axis=0)
                cm_df = pd.DataFrame(cm_sum_local, index=names, columns=names)
                cm_df.to_csv(self.out_dir / "confusion_matrix.csv", index=True)
                per_class_df = per_class_from_confusion(cm_sum_local, class_names=names)
                per_class_df.to_csv(self.out_dir / "per_class_metrics.csv", index=False)

            # Aggregate central diagnostics if present
            if cms_central:
                cm_sum_central = np.sum(np.stack(cms_central, axis=0), axis=0)
                cm_df = pd.DataFrame(cm_sum_central, index=names, columns=names)
                cm_df.to_csv(self.out_dir / "confusion_matrix_central.csv", index=True)
                per_class_df = per_class_from_confusion(cm_sum_central, class_names=names)
                per_class_df.to_csv(self.out_dir / "per_class_metrics_central.csv", index=False)

        if server_round == self.rounds:
            self.pbar.close()
            pd.DataFrame(self.history_rows).to_csv(self.out_dir / "metrics_by_round.csv", index=False)

        return aggregated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--min-clients", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preproc-dir", type=Path, required=True)
    ap.add_argument("--central-test-csv", type=Path, default=None,
                    help="Optional central held-out test CSV to send to clients each evaluate round")

    ap.add_argument("--server-lr", type=float, default=0.01)
    ap.add_argument("--eta-l", type=float, default=1e-3)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    ap.add_argument("--use-layernorm", action="store_true", default=True)
    ap.add_argument("--no-layernorm", action="store_true", default=False)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    _, feature_cols = load_schema(args.preproc_dir)
    use_ln = bool(args.use_layernorm) and not bool(args.no_layernorm)

    model_cfg = ModelConfig(
        input_dim=len(feature_cols),
        num_classes=6,
        dropout=float(args.dropout),
        l2=float(args.l2),
        use_layernorm=use_ln,
        activation=str(args.activation),
    )
    model = create_mlp(model_cfg)
    initial_parameters = fl.common.ndarrays_to_parameters([w.astype(np.float32) for w in model.get_weights()])

    def on_eval_cfg(r: int) -> dict:
        cfg = {"server_round": r, "is_final": int(r == int(args.rounds))}
        if args.central_test_csv:
            # send path as string so clients can load it
            cfg["central_test_csv"] = str(args.central_test_csv)
        return cfg

    strategy = TqdmFedAdam(
        rounds=args.rounds,
        out_dir=args.out_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
        eta=float(args.server_lr),
        eta_l=float(args.eta_l),
        beta_1=float(args.beta1),
        beta_2=float(args.beta2),
        tau=float(args.tau),
        on_evaluate_config_fn=on_eval_cfg,
    )

    fl.server.start_server(
        server_address=args.server,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"Done. Wrote metrics to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
