"""Common utilities for the 6-class (without ddos_slowloris) Flower FL variant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    num_classes: int
    dropout: float = 0.3
    l2: float = 1e-4
    use_layernorm: bool = True
    activation: str = "relu"  # relu or gelu


def load_schema(preproc_dir: Path) -> Tuple[str, List[str]]:
    import json

    schema = json.loads((preproc_dir / "schema.json").read_text(encoding="utf-8"))
    return schema["label_col"], list(schema["feature_cols"])


def load_client_df(preproc_dir: Path, client_name: str) -> pd.DataFrame:
    return pd.read_csv(preproc_dir / "datasets" / f"{client_name}.csv", low_memory=False)


def build_label_mapping() -> Dict[str, int]:
    # 6-class mapping: slowloris removed
    classes = [
        "benign",
        "bruteForce",
        "ddos_tcp_ddos",
        "pfcp_session_deletion",
        "pfcp_session_establishment",
        "pfcp_session_modification",
    ]
    return {name: i for i, name in enumerate(classes)}


def label_names() -> List[str]:
    mapping = build_label_mapping()
    inv = {v: k for k, v in mapping.items()}
    return [inv[i] for i in range(len(inv))]


def encode_labels(y_str: np.ndarray, mapping: Dict[str, int]) -> np.ndarray:
    y = np.empty(len(y_str), dtype=np.int32)
    for i, s in enumerate(y_str):
        if s not in mapping:
            raise ValueError(f"Unknown label '{s}'. Expected one of: {sorted(mapping.keys())}")
        y[i] = mapping[s]
    return y


def split_client_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    label_mapping: Dict[str, int],
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = encode_labels(df[label_col].astype(str).to_numpy(), label_mapping)

    y_counts = np.bincount(y, minlength=len(label_mapping))
    can_stratify = np.all(y_counts[y_counts > 0] >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if can_stratify else None,
        shuffle=True,
    )
    return X_train, y_train, X_test, y_test


def create_mlp(cfg: ModelConfig) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(cfg.l2)
    act = tf.nn.gelu if cfg.activation == "gelu" else "relu"

    inputs = tf.keras.Input(shape=(cfg.input_dim,), name="features")
    x = inputs

    if cfg.use_layernorm:
        x = tf.keras.layers.LayerNormalization(name="ln_in")(x)

    x = tf.keras.layers.Dense(128, activation=act, kernel_regularizer=reg, name="dense_128")(x)
    x = tf.keras.layers.Dropout(cfg.dropout, name="dropout_1")(x)
    x = tf.keras.layers.Dense(64, activation=act, kernel_regularizer=reg, name="dense_64")(x)
    x = tf.keras.layers.Dropout(cfg.dropout, name="dropout_2")(x)
    x = tf.keras.layers.Dense(32, activation=act, kernel_regularizer=reg, name="dense_32")(x)
    x = tf.keras.layers.Dropout(cfg.dropout, name="dropout_3")(x)

    outputs = tf.keras.layers.Dense(cfg.num_classes, activation="softmax", name="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_tabular_6c")


def eval_metrics(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    probs = model.predict(X_test, batch_size=4096, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)
    present = np.unique(y_test)

    return {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=present, zero_division=0)),
        "macro_precision": float(
            precision_score(y_test, y_pred, average="macro", labels=present, zero_division=0)
        ),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", labels=present, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
        "n_classes_test": int(len(present)),
    }


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels).astype(np.int64)


def per_class_from_confusion(cm: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    # cm: rows=true, cols=pred
    num_classes = cm.shape[0]
    rows = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = float(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "label": class_names[i] if i < len(class_names) else str(i),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
    return pd.DataFrame(rows)
