"""Flower client for the 6-class (without ddos_slowloris) variant.

- 3 clients: client_1, client_2, client_3 (as created under withoutSlowloris/preprocessing/datasets)
- Uses global class weights computed by the local runner (passed via JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
import traceback

from .common import (
    ModelConfig,
    build_label_mapping,
    compute_confusion,
    create_mlp,
    eval_metrics,
    label_names,
    load_client_df,
    load_schema,
    per_class_from_confusion,
    split_client_data,
    encode_labels,
)


def load_class_weights_arg(arg: str) -> np.ndarray:
    """Load class weights from a JSON file path or inline JSON.

    Supported formats:
    - A list/array of length 6
    - A dict mapping label->weight
    - An object with key "label_weights" mapping label->weight
    """

    payload_str = arg
    p = Path(arg)
    if p.exists() and p.is_file():
        payload_str = p.read_text(encoding="utf-8")

    payload = json.loads(payload_str)

    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32)

    if isinstance(payload, dict) and "label_weights" in payload and isinstance(payload["label_weights"], dict):
        payload = payload["label_weights"]

    if isinstance(payload, dict):
        ordered_labels = label_names()
        weights = [float(payload[lbl]) for lbl in ordered_labels]
        return np.asarray(weights, dtype=np.float32)

    raise ValueError("Unsupported class weights JSON format")


def get_model_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    return [w.astype(np.float32) for w in model.get_weights()]


def set_model_parameters(model: tf.keras.Model, parameters: List[np.ndarray]) -> None:
    model.set_weights(parameters)


class TabularMLPClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_name: str,
        preproc_dir: Path,
        model_cfg: ModelConfig,
        test_size: float,
        seed: int,
        lr: float,
        batch_size: int,
        local_epochs: int,
        class_weights: np.ndarray,
    ) -> None:
        self.client_name = client_name
        self.preproc_dir = preproc_dir
        self.model_cfg = model_cfg
        self.test_size = test_size
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        label_col, feature_cols = load_schema(preproc_dir)
        label_mapping = build_label_mapping()

        df = load_client_df(preproc_dir, client_name)
        X_tr, y_tr, X_te, y_te = split_client_data(
            df,
            feature_cols=feature_cols,
            label_col=label_col,
            label_mapping=label_mapping,
            test_size=test_size,
            seed=seed,
        )

        self.X_train = X_tr
        self.y_train = y_tr
        self.X_test = X_te
        self.y_test = y_te

        cw = np.asarray(class_weights, dtype=np.float32)
        if cw.shape != (self.model_cfg.num_classes,):
            raise ValueError(f"class_weights must be shape ({self.model_cfg.num_classes},), got {cw.shape}")
        self.class_w = cw

        self.model = create_mlp(model_cfg)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )

    def get_parameters(self, config: Dict[str, str]):  # type: ignore[override]
        return get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        set_model_parameters(self.model, parameters)
        sample_w = self.class_w[self.y_train]

        self.model.fit(
            self.X_train,
            self.y_train,
            sample_weight=sample_w,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True,
        )

        return get_model_parameters(self.model), len(self.y_train), {}

    def evaluate(self, parameters, config):  # type: ignore[override]
        set_model_parameters(self.model, parameters)
        loss = float(self.model.evaluate(self.X_test, self.y_test, batch_size=4096, verbose=0))
        metrics = eval_metrics(self.model, self.X_test, self.y_test)

        cfg = config if isinstance(config, dict) else {}
        metrics["cfg_num_keys"] = float(len(cfg))
        try:
            metrics["cfg_is_final_raw"] = float(cfg.get("is_final", -1))
        except Exception:
            metrics["cfg_is_final_raw"] = -999.0
        try:
            metrics["cfg_server_round_raw"] = float(cfg.get("server_round", -1))
        except Exception:
            metrics["cfg_server_round_raw"] = -999.0

        is_final = int(cfg.get("is_final", 0))
        metrics["diag_is_final"] = float(is_final)
        if is_final == 1:
            probs = self.model.predict(self.X_test, batch_size=4096, verbose=0)
            y_pred = np.argmax(probs, axis=1).astype(np.int32)
            cm = compute_confusion(self.y_test.astype(np.int32), y_pred, num_classes=self.model_cfg.num_classes)
            per_class = per_class_from_confusion(cm, class_names=label_names())

            details = {
                "client": self.client_name,
                "num_test": int(len(self.y_test)),
                "labels": label_names(),
                "confusion_matrix": cm.tolist(),
                "per_class": per_class.to_dict(orient="records"),
            }
            details_json = json.dumps(details)
            metrics["diag_details_len"] = float(len(details_json))
            metrics["details_json"] = details_json

        # If a central held-out test CSV was provided in the evaluate config,
        # load it and evaluate the current model on that set as well.
        central_csv = cfg.get("central_test_csv")
        if central_csv:
            try:
                from pathlib import Path

                ct = Path(str(central_csv))
                if not ct.exists():
                    alt = self.preproc_dir / ct
                    if alt.exists():
                        ct = alt
                df_c = pd.read_csv(ct)
                label_col, feature_cols = load_schema(self.preproc_dir)
                Xc = df_c[feature_cols].to_numpy(dtype=np.float32)
                label_mapping = build_label_mapping()
                y_c = encode_labels(df_c[label_col].astype(str).to_numpy(), label_mapping)

                probs_c = self.model.predict(Xc, batch_size=4096, verbose=0)
                y_pred_c = np.argmax(probs_c, axis=1).astype(np.int32)
                cm_c = compute_confusion(y_c.astype(np.int32), y_pred_c, num_classes=self.model_cfg.num_classes)
                per_class_c = per_class_from_confusion(cm_c, class_names=label_names())

                central_details = {
                    "client": self.client_name,
                    "num_test": int(len(y_c)),
                    "labels": label_names(),
                    "confusion_matrix": cm_c.tolist(),
                    "per_class": per_class_c.to_dict(orient="records"),
                }
                central_json = json.dumps(central_details)
                metrics["central_diag_details_len"] = float(len(central_json))
                metrics["central_details_json"] = central_json

                # Also provide simple summary metrics for convenience
                try:
                    em = eval_metrics(self.model, Xc, y_c)
                    metrics.update({f"central_{k}": float(v) for k, v in em.items()})
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning: failed central-test evaluation for {self.client_name}: {e}")
        return loss, len(self.y_test), metrics

    def write_local_diagnostics(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        probs = self.model.predict(self.X_test, batch_size=4096, verbose=0)
        y_pred = np.argmax(probs, axis=1).astype(np.int32)
        cm = compute_confusion(self.y_test.astype(np.int32), y_pred, num_classes=self.model_cfg.num_classes)
        names = label_names()

        cm_df = pd.DataFrame(cm, index=names, columns=names)
        cm_df.to_csv(out_dir / f"confusion_matrix_{self.client_name}.csv", index=True)

        per_class_df = per_class_from_confusion(cm, class_names=names)
        per_class_df.insert(0, "client", self.client_name)
        per_class_df.to_csv(out_dir / f"per_class_metrics_{self.client_name}.csv", index=False)
        # Save the local model weights so the trained client model is preserved for inspection/SHAP.
        try:
            model_path = out_dir / f"client_model_{self.client_name}.keras"
            # Use TensorFlow's SavedModel / Keras native format
            # Save without unsupported kwargs for native format
            self.model.save(str(model_path))
        except Exception as e:
            # Best-effort: do not fail diagnostics if saving model fails
            # Log the exception so runs don't silently hide save failures
            print(f"Warning: failed to save client model to {model_path}: {e}")
            traceback.print_exc()
        # If a central test CSV path was attached to the client instance, evaluate and write central diagnostics.
        central_csv = getattr(self, "central_test_csv", None)
        if central_csv is not None:
            try:
                ct = Path(str(central_csv))
                if not ct.exists():
                    alt = self.preproc_dir / ct
                    if alt.exists():
                        ct = alt
                df_c = pd.read_csv(ct)
                label_col, feature_cols = load_schema(self.preproc_dir)
                Xc = df_c[feature_cols].to_numpy(dtype=np.float32)
                label_mapping = build_label_mapping()
                y_c = encode_labels(df_c[label_col].astype(str).to_numpy(), label_mapping)

                probs_c = self.model.predict(Xc, batch_size=4096, verbose=0)
                y_pred_c = np.argmax(probs_c, axis=1).astype(np.int32)
                cm_c = compute_confusion(y_c.astype(np.int32), y_pred_c, num_classes=self.model_cfg.num_classes)
                names = label_names()

                cm_df = pd.DataFrame(cm_c, index=names, columns=names)
                cm_df.to_csv(out_dir / f"confusion_matrix_central_{self.client_name}.csv", index=True)

                per_class_c = per_class_from_confusion(cm_c, class_names=names)
                per_class_c.insert(0, "client", self.client_name)
                per_class_c.to_csv(out_dir / f"per_class_metrics_central_{self.client_name}.csv", index=False)
            except Exception as e:
                print(f"Warning: failed to write central diagnostics for {self.client_name}: {e}")
                traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="0.0.0.0:8080")
    ap.add_argument("--client", type=str, required=True, choices=["client_1", "client_2", "client_3"])
    ap.add_argument("--preproc-dir", type=Path, required=True)

    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    ap.add_argument("--use-layernorm", action="store_true", default=True)
    ap.add_argument("--no-layernorm", action="store_true", default=False)

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--local-epochs", type=int, default=1)

    ap.add_argument(
        "--class-weights-json",
        type=str,
        required=True,
        help="JSON file path or inline JSON (list of 6 weights, or dict label->weight, or {label_weights:{...}})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, writes per-client diagnostics CSVs after training completes.",
    )
    ap.add_argument("--central-test-csv", type=Path, default=None,
                    help="Optional central held-out test CSV (will be evaluated at client exit)")

    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

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

    class_weights = load_class_weights_arg(args.class_weights_json)

    client = TabularMLPClient(
        client_name=args.client,
        preproc_dir=args.preproc_dir,
        model_cfg=model_cfg,
        test_size=float(args.test_size),
        seed=int(args.seed),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        local_epochs=int(args.local_epochs),
        class_weights=class_weights,
    )

    # Attach central test csv path to client instance for final diagnostics
    if args.central_test_csv is not None:
        setattr(client, "central_test_csv", args.central_test_csv)

    fl.client.start_numpy_client(server_address=args.server, client=client)
    if args.out_dir is not None:
        client.write_local_diagnostics(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
