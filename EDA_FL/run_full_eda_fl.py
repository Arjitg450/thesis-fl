"""FL-aware exhaustive EDA for stage2Data (7-class).

Writes ALL artifacts (tables, plots, report) into --out-dir.

This script is intentionally modular-in-one-file for reproducibility in thesis work.

Official references (APIs used):
- pandas: https://pandas.pydata.org/docs/
- scikit-learn feature selection (MI, ANOVA, chi2): https://scikit-learn.org/stable/modules/feature_selection.html
- scikit-learn PCA / t-SNE: https://scikit-learn.org/stable/modules/decomposition.html#pca ; https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- scipy KS test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- SHAP: https://shap.readthedocs.io/

Run:
  /home/arjit/thesis/.venv/bin/python /home/arjit/thesis/EDA_FL/run_full_eda_fl.py \
    --data-dir /home/arjit/thesis/stage2Data \
    --out-dir /home/arjit/thesis/EDA_FL/output
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, skew, kurtosis

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier


# ------------------------
#  Labeling (7 classes)
# ------------------------


def label_from_relpath(rel: Path) -> str:
    parts = list(rel.parts)
    if not parts:
        return "unknown"

    top = parts[0].lower()

    if top == "benign":
        return "benign"
    if top == "bruteforce":
        return "bruteForce"

    if top == "ddos":
        if len(parts) >= 2:
            subtype = parts[1].lower().replace(" ", "_")
            if "slowloris" in subtype:
                return "ddos_slowloris"
            if "tcp" in subtype:
                return "ddos_tcp_ddos"
        return "ddos"

    if top == "pfcp":
        if len(parts) >= 2:
            subtype = parts[1].lower()
            if "session_deletion" in subtype:
                return "pfcp_session_deletion"
            if "session_establishment" in subtype:
                return "pfcp_session_establishment"
            if "session_modification" in subtype:
                return "pfcp_session_modification"
        return "pfcp"

    return top


# ------------------------
#  IO + basic utilities
# ------------------------


@dataclass(frozen=True)
class FileRef:
    path: Path
    rel: Path
    label: str


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def discover_csvs(data_dir: Path) -> List[FileRef]:
    refs: List[FileRef] = []
    for p in sorted(data_dir.rglob("*.csv")):
        rel = p.relative_to(data_dir)
        label = label_from_relpath(rel)
        refs.append(FileRef(path=p, rel=rel, label=label))
    return refs


def load_all(refs: List[FileRef]) -> pd.DataFrame:
    dfs = []
    for r in refs:
        df = pd.read_csv(r.path, low_memory=False)
        df["__label"] = r.label
        df["__source_file"] = str(r.rel)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


# ------------------------
#  Integrity & missingness
# ------------------------


def missingness_tables(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_miss = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(6),
        "nunique_non_null": df.nunique(dropna=True),
        "dtype": df.dtypes.astype(str),
    }).sort_values(["missing_pct", "missing_count"], ascending=False)

    by_class = []
    for lbl, g in df.groupby(label_col, sort=True):
        s = (g.isna().mean() * 100).round(6)
        s.name = str(lbl)
        by_class.append(s)
    miss_by_class = pd.concat(by_class, axis=1) if by_class else pd.DataFrame()

    return global_miss, miss_by_class


def constant_and_near_constant(df: pd.DataFrame, exclude: set, near_const_thresh: float = 0.999) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rows_near = []
    n = len(df)
    for c in df.columns:
        if c in exclude:
            continue
        vc = df[c].value_counts(dropna=False)
        if vc.empty:
            continue
        top_frac = float(vc.iloc[0] / max(1, n))
        nunique = int(df[c].nunique(dropna=False))
        if nunique <= 1:
            rows.append({"feature": c, "top_fraction": top_frac, "nunique": nunique})
        elif top_frac >= near_const_thresh:
            rows_near.append({"feature": c, "top_fraction": top_frac, "nunique": nunique})
    return pd.DataFrame(rows).sort_values("feature"), pd.DataFrame(rows_near).sort_values("top_fraction", ascending=False)


def duplicate_checks(df: pd.DataFrame, exclude_cols: List[str]) -> Dict[str, int]:
    d0 = int(df.duplicated().sum())
    d1 = int(df.drop(columns=[c for c in exclude_cols if c in df.columns]).duplicated().sum())
    return {"exact_duplicate_rows": d0, "duplicate_rows_excluding_cols": d1}


# ------------------------
#  Sparsity (zeros)
# ------------------------


def zero_rate_tables(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame(), pd.DataFrame()

    zero_pct = (num == 0).mean().sort_values(ascending=False) * 100
    global_zero = pd.DataFrame({"zero_pct": zero_pct.round(6), "dtype": num.dtypes.astype(str)})

    per_class = []
    for lbl, g in df.groupby(label_col, sort=True):
        ng = g.select_dtypes(include=[np.number])
        zp = (ng == 0).mean() * 100
        zp.name = str(lbl)
        per_class.append(zp)
    zero_by_class = pd.concat(per_class, axis=1) if per_class else pd.DataFrame()

    return global_zero.sort_values("zero_pct", ascending=False), zero_by_class


# ------------------------
#  Robust stats (global/per-class)
# ------------------------


def robust_stats(df: pd.DataFrame, label_col: str, max_features: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    num_cols = [c for c in num_cols if c not in {label_col}]
    if max_features:
        num_cols = num_cols[:max_features]

    def _mad(x: np.ndarray) -> float:
        med = np.nanmedian(x)
        return float(np.nanmedian(np.abs(x - med)))

    # Global
    rows = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite_n = int(np.isfinite(x).sum())
        if finite_n == 0:
            rows.append({
                "feature": c,
                "count": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "var": float("nan"),
                "std": float("nan"),
                "skew": float("nan"),
                "kurtosis": float("nan"),
                "q1": float("nan"),
                "q3": float("nan"),
                "iqr": float("nan"),
                "mad": float("nan"),
            })
            continue
        q1 = float(np.nanpercentile(x, 25))
        q3 = float(np.nanpercentile(x, 75))
        rows.append({
            "feature": c,
            "count": finite_n,
            "mean": float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
            "var": float(np.nanvar(x)),
            "std": float(np.nanstd(x)),
            "skew": float(skew(x, nan_policy="omit")),
            "kurtosis": float(kurtosis(x, nan_policy="omit")),
            "q1": q1,
            "q3": q3,
            "iqr": float(q3 - q1),
            "mad": _mad(x),
        })
    global_stats = pd.DataFrame(rows).sort_values("feature")

    # Per-class (long format)
    rows = []
    for lbl, g in df.groupby(label_col, sort=True):
        for c in num_cols:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            finite_n = int(np.isfinite(x).sum())
            if finite_n == 0:
                rows.append({
                    "label": str(lbl),
                    "feature": c,
                    "count": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "iqr": float("nan"),
                    "mad": float("nan"),
                })
                continue
            q1 = float(np.nanpercentile(x, 25))
            q3 = float(np.nanpercentile(x, 75))
            rows.append({
                "label": str(lbl),
                "feature": c,
                "count": finite_n,
                "mean": float(np.nanmean(x)),
                "median": float(np.nanmedian(x)),
                "std": float(np.nanstd(x)),
                "iqr": float(q3 - q1),
                "mad": _mad(x),
            })
    per_class_stats = pd.DataFrame(rows)

    return global_stats, per_class_stats


# ------------------------
#  Relationships: correlation + MI
# ------------------------


def correlation_matrices(df: pd.DataFrame, max_cols: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num = df.select_dtypes(include=[np.number])
    cols = list(num.columns)[:max_cols]
    pearson = num[cols].corr(method="pearson").fillna(0)
    spearman = num[cols].corr(method="spearman").fillna(0)
    return pearson, spearman


def plot_heatmap(mat: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(mat, cmap="coolwarm", center=0, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def mi_feature_label(df: pd.DataFrame, label_col: str, random_state: int = 42) -> pd.Series:
    # Numeric only MI (robust and fast). For categorical, we use chi2 later.
    y = df[label_col].astype("category").cat.codes.to_numpy()
    num = df.select_dtypes(include=[np.number]).copy()
    if label_col in num.columns:
        num = num.drop(columns=[label_col])

    # Fill NaN with median (for MI computation only)
    X = num.copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    mi = mutual_info_classif(X.to_numpy(), y, discrete_features=False, random_state=random_state)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


# ------------------------
#  Separability geometry
# ------------------------


def pca_tsne_embeddings(df: pd.DataFrame, label_col: str, out_dir: Path, sample_n: int = 30000, random_state: int = 42) -> Dict:
    _mkdir(out_dir)

    # numeric-only + simple imputation
    num = df.select_dtypes(include=[np.number]).copy()
    if label_col in num.columns:
        num = num.drop(columns=[label_col])

    X = num.copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())

    y = df[label_col].astype("category")

    if len(X) > sample_n:
        sample = df[[label_col]].copy()
        sample_idx = sample.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max(1000, sample_n // df[label_col].nunique())), random_state=random_state)
        ).index
        Xs = X.loc[sample_idx]
        ys = y.loc[sample_idx]
    else:
        Xs, ys = X, y

    # PCA
    scaler = RobustScaler(with_centering=True)
    Xs_scaled = scaler.fit_transform(Xs)
    pca = PCA(n_components=10, random_state=random_state)
    X_pca = pca.fit_transform(Xs_scaled)

    pca_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_sum_2": float(pca.explained_variance_ratio_[:2].sum()),
    }

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=ys.astype(str), s=10, linewidth=0)
    plt.title("PCA (first 2 components)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_2d.png", dpi=160)
    plt.close()

    # t-SNE (on PCA-reduced for speed)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=random_state)
    X_tsne = tsne.fit_transform(X_pca[:, :10])

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=ys.astype(str), s=10, linewidth=0)
    plt.title("t-SNE (on PCA(10) features)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_2d.png", dpi=160)
    plt.close()

    return pca_info


# ------------------------
#  Importance (no final model)
# ------------------------


def build_preprocess(df: pd.DataFrame, label_col: str) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # Identify columns
    feature_cols = [c for c in df.columns if c not in {label_col, "__source_file"}]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    # Presence flags for sparse text fields: encode missingness explicitly
    sparse_text = [
        c for c in categorical_cols
        if c in {"user_agent", "content_type", "requested_server_name", "client_fingerprint", "server_fingerprint"}
    ]

    def _add_presence_flags(x: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        for c in sparse_text:
            if c in out.columns:
                out[c + "__present"] = out[c].notna().astype(int)
        return out

    # We apply presence flags outside sklearn pipeline, but we keep the list here.
    # (Simpler, and avoids custom transformers.)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_cols, categorical_cols


def chi2_categorical_proxy(df: pd.DataFrame, label_col: str, out_dir: Path) -> pd.DataFrame:
    # Chi-square requires non-negative X.
    # We compute chi2 ONLY on one-hot encoded categorical features (non-negative by construction).
    work = df.copy()
    y = work[label_col].astype("category").cat.codes.to_numpy()

    feature_cols = [c for c in work.columns if c not in {label_col, "__source_file"}]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(work[c])]

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    X = pre.fit_transform(work)

    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        feature_names = [f"cat_f{i}" for i in range(X.shape[1])]

    scores, pvals = chi2(X, y)
    res = pd.DataFrame({"feature": feature_names, "chi2": scores, "p_value": pvals}).sort_values("chi2", ascending=False)
    _save_df(res.head(500), out_dir / "chi2_top500.csv")
    return res


def anova_numeric(df: pd.DataFrame, label_col: str, out_dir: Path) -> pd.DataFrame:
    y = df[label_col].astype("category").cat.codes.to_numpy()
    num = df.select_dtypes(include=[np.number]).copy()
    if label_col in num.columns:
        num = num.drop(columns=[label_col])

    # Median impute
    X = num.copy()
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())

    f, p = f_classif(X.to_numpy(), y)
    res = pd.DataFrame({"feature": X.columns, "f_score": f, "p_value": p}).sort_values("f_score", ascending=False)
    _save_df(res, out_dir / "anova_f_scores.csv")
    return res


def proxy_model_importance(df: pd.DataFrame, label_col: str, out_dir: Path, sample_n: int = 120000, random_state: int = 42) -> Dict[str, str]:
    # Proxy model only for analysis (RF). Trains on a sample for speed.
    work = df.copy()

    if len(work) > sample_n:
        work = work.groupby(label_col, group_keys=False).apply(lambda g: g.sample(n=min(len(g), sample_n // work[label_col].nunique()), random_state=random_state))

    # Drop the artifact raw column named 'label' if present (leak from one file)
    if "label" in work.columns:
        work = work.drop(columns=["label"])

    y = work[label_col].astype("category").cat.codes

    # IMPORTANT:
    # - impurity-based feature_importances_ correspond to the *transformed* feature space
    # - permutation_importance(pipe, X, y) corresponds to the *input* columns of X
    feature_cols = [c for c in work.columns if c not in {label_col, "__source_file"}]

    pre, _, _ = build_preprocess(work, label_col)

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(work, y))
    X_train, X_test = work.iloc[train_idx], work.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train_feat = X_train[feature_cols]
    X_test_feat = X_test[feature_cols]

    pipe.fit(X_train_feat, y_train)
    y_pred = pipe.predict(X_test_feat)

    # Save report
    report = classification_report(y_test, y_pred, output_dict=False)
    (out_dir / "proxy_rf_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Proxy RF confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "proxy_rf_confusion_matrix.png", dpi=160)
    plt.close()

    # Feature importances (impurity) - map to names
    try:
        transformed_feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        transformed_feature_names = []

    n_model_features = int(pipe.named_steps["clf"].feature_importances_.shape[0])
    if len(transformed_feature_names) != n_model_features:
        transformed_feature_names = [f"tf{i}" for i in range(n_model_features)]

    fi = pd.DataFrame({"feature": transformed_feature_names, "importance": pipe.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False)
    _save_df(fi.head(500), out_dir / "proxy_rf_feature_importances_top500.csv")

    # Permutation importance on a small subset
    X_test_small = X_test_feat.sample(n=min(len(X_test_feat), 8000), random_state=random_state)
    y_test_small = y_test.loc[X_test_small.index]
    perm = permutation_importance(pipe, X_test_small, y_test_small, n_repeats=5, random_state=random_state, n_jobs=-1)

    perm_feature_names = list(X_test_small.columns)
    n_perm_features = int(perm.importances_mean.shape[0])
    if len(perm_feature_names) != n_perm_features:
        perm_feature_names = [f"x{i}" for i in range(n_perm_features)]

    perm_df = pd.DataFrame({
        "feature": perm_feature_names,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    })
    perm_df = perm_df.sort_values("perm_importance_mean", ascending=False)
    _save_df(perm_df.head(500), out_dir / "proxy_rf_permutation_importance_top500.csv")

    return {"proxy_report": "proxy_rf_classification_report.txt"}


def shap_tree_proxy(df: pd.DataFrame, label_col: str, out_dir: Path, random_state: int = 42) -> Optional[str]:
    # SHAP can be expensive; compute on a small stratified sample.
    try:
        import shap  # type: ignore
    except Exception:
        return None

    work = df.copy()
    if "label" in work.columns:
        work = work.drop(columns=["label"])

    # Use only numeric features for SHAP robustness and speed
    num = work.select_dtypes(include=[np.number]).copy()
    if label_col in num.columns:
        num = num.drop(columns=[label_col])

    # median impute
    for c in num.columns:
        num[c] = num[c].fillna(num[c].median())

    y = work[label_col].astype("category").cat.codes

    # stratified sample
    n = min(len(work), 12000)
    sample_idx = work.groupby(label_col, group_keys=False).apply(lambda g: g.sample(n=min(len(g), max(300, n // work[label_col].nunique())), random_state=random_state)).index
    Xs = num.loc[sample_idx]
    ys = y.loc[sample_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(Xs, ys)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    out_path = out_dir / "shap_summary.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    return str(out_path.name)


# ------------------------
#  FL-aware robustness
# ------------------------


def simulate_fl_clients(df: pd.DataFrame, label_col: str) -> Dict[str, List[str]]:
    return {
        "client_1": ["benign", "bruteForce"],
        "client_2": ["benign", "ddos_slowloris"],
        "client_3": ["benign", "ddos_tcp_ddos", "pfcp_session_deletion"],
        "client_4": ["benign", "pfcp_session_deletion", "pfcp_session_establishment", "pfcp_session_modification"],
    }


def drift_ks_vs_global(df: pd.DataFrame, label_col: str, client_map: Dict[str, List[str]], out_dir: Path, max_features: int = 80) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {label_col}]
    num_cols = num_cols[:max_features]

    # global numeric imputation (median)
    global_num = df[num_cols].copy()
    for c in num_cols:
        global_num[c] = pd.to_numeric(global_num[c], errors="coerce")
        global_num[c] = global_num[c].fillna(global_num[c].median())

    rows = []
    for client, labels in client_map.items():
        sub = df[df[label_col].isin(labels)]
        sub_num = sub[num_cols].copy()
        for c in num_cols:
            sub_num[c] = pd.to_numeric(sub_num[c], errors="coerce")
            sub_num[c] = sub_num[c].fillna(global_num[c].median())

        for c in num_cols:
            # KS test compares distributions (continuous); we report statistic only.
            try:
                stat = float(ks_2samp(global_num[c].values, sub_num[c].values).statistic)
            except Exception:
                stat = float("nan")
            rows.append({"client": client, "feature": c, "ks_stat": stat})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "fl_drift_ks.csv", index=False)

    # summarize: top drift features per client
    summary = out.sort_values(["client", "ks_stat"], ascending=[True, False]).groupby("client").head(25)
    summary.to_csv(out_dir / "fl_drift_ks_top25_per_client.csv", index=False)

    return out


# ------------------------
#  Domain-aware feature engineering
# ------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    def safe_div(a, b):
        return a / (b.replace(0, np.nan))

    # Common flow ratios
    if "bidirectional_bytes" in d.columns and "bidirectional_packets" in d.columns:
        d["fe_bytes_per_packet_bi"] = safe_div(d["bidirectional_bytes"], d["bidirectional_packets"]).fillna(0)

    if "src2dst_bytes" in d.columns and "src2dst_packets" in d.columns:
        d["fe_bytes_per_packet_s2d"] = safe_div(d["src2dst_bytes"], d["src2dst_packets"]).fillna(0)

    if "dst2src_bytes" in d.columns and "dst2src_packets" in d.columns:
        d["fe_bytes_per_packet_d2s"] = safe_div(d["dst2src_bytes"], d["dst2src_packets"]).fillna(0)

    if "src2dst_bytes" in d.columns and "dst2src_bytes" in d.columns:
        d["fe_byte_ratio_s2d_over_d2s"] = safe_div(d["src2dst_bytes"], d["dst2src_bytes"] + 1).fillna(0)

    if "src2dst_packets" in d.columns and "dst2src_packets" in d.columns:
        d["fe_packet_ratio_s2d_over_d2s"] = safe_div(d["src2dst_packets"], d["dst2src_packets"] + 1).fillna(0)

    # Time normalized metrics
    if "bidirectional_duration_ms" in d.columns and "bidirectional_packets" in d.columns:
        d["fe_packets_per_ms_bi"] = safe_div(d["bidirectional_packets"], d["bidirectional_duration_ms"] + 1).fillna(0)

    if "bidirectional_duration_ms" in d.columns and "bidirectional_bytes" in d.columns:
        d["fe_bytes_per_ms_bi"] = safe_div(d["bidirectional_bytes"], d["bidirectional_duration_ms"] + 1).fillna(0)

    # TCP-flag ratios (if present)
    for flag in ["syn", "rst", "fin", "ack", "psh"]:
        bi = f"bidirectional_{flag}_packets"
        if bi in d.columns and "bidirectional_packets" in d.columns:
            d[f"fe_{flag}_rate_bi"] = safe_div(d[bi], d["bidirectional_packets"] + 1).fillna(0)

    # UDPS normalized (PFCP signal per packet)
    if "udps.session_report_request_counter" in d.columns and "bidirectional_packets" in d.columns:
        d["fe_udps_session_report_req_per_packet"] = safe_div(d["udps.session_report_request_counter"], d["bidirectional_packets"] + 1).fillna(0)

    return d


# ------------------------
#  Report writer
# ------------------------


def write_report(out_dir: Path, sections: Dict[str, str]) -> None:
    lines = []
    lines.append("# FL-Aware Multi-Class EDA Report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    for k in [
        "1_integrity",
        "2_imbalance",
        "3_sparsity",
        "4_stats",
        "5_visuals",
        "6_corr",
        "7_geometry",
        "8_importance",
        "9_fl",
        "10_fe",
        "11_pipeline",
        "12_outputs",
    ]:
        lines.append(sections.get(k, f"## {k}\nMissing section."))
        lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------
#  Main
# ------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/home/arjit/thesis/stage2Data"))
    ap.add_argument("--out-dir", type=Path, default=Path("/home/arjit/thesis/EDA_FL/output"))
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    _mkdir(out_dir)

    refs = discover_csvs(data_dir)
    inv = pd.DataFrame([{"rel": str(r.rel), "path": str(r.path), "label": r.label, "size_mb": round(r.path.stat().st_size/(1024**2), 4)} for r in refs])
    inv.to_csv(out_dir / "INDEX.csv", index=False)

    df = load_all(refs)

    # Basic integrity
    label_counts = df["__label"].value_counts().sort_values(ascending=False)

    # Leakage artifact column present only in one file
    artifact_cols = []
    if "label" in df.columns:
        artifact_cols.append("label")

    # Missingness
    miss_global, miss_by_class = missingness_tables(df.drop(columns=[]), "__label")
    miss_global.to_csv(out_dir / "missingness_global.csv")
    miss_by_class.to_csv(out_dir / "missingness_by_class.csv")

    # Constant / near-constant
    const_df, near_const_df = constant_and_near_constant(df, exclude={"__label", "__source_file"})
    const_df.to_csv(out_dir / "constant_features.csv", index=False)
    near_const_df.to_csv(out_dir / "near_constant_features.csv", index=False)

    # Duplicates
    dup = duplicate_checks(df, exclude_cols=["id", "__source_file"])
    (out_dir / "duplicate_summary.json").write_text(json.dumps(dup, indent=2), encoding="utf-8")

    # Class imbalance plots
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, color="#4C72B0")
    plt.xticks(rotation=25, ha="right")
    plt.title("Class counts (7-class)")
    plt.tight_layout()
    plt.savefig(out_dir / "class_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=np.log10(label_counts.values + 1), color="#55A868")
    plt.xticks(rotation=25, ha="right")
    plt.title("Class counts (log10 scale)")
    plt.ylabel("log10(count+1)")
    plt.tight_layout()
    plt.savefig(out_dir / "class_counts_log.png", dpi=160)
    plt.close()

    # Sparsity (zeros)
    zero_global, zero_by_class = zero_rate_tables(df, "__label")
    zero_global.to_csv(out_dir / "zero_rates_global.csv")
    zero_by_class.to_csv(out_dir / "zero_rates_by_class.csv")

    # Stats
    gstats, cstats = robust_stats(df, "__label")
    gstats.to_csv(out_dir / "robust_stats_global.csv", index=False)
    cstats.to_csv(out_dir / "robust_stats_by_class_long.csv", index=False)

    # Correlations
    pearson, spearman = correlation_matrices(df, max_cols=60)
    pearson.to_csv(out_dir / "corr_pearson.csv")
    spearman.to_csv(out_dir / "corr_spearman.csv")
    plot_heatmap(pearson, out_dir / "corr_pearson_heatmap.png", "Pearson correlation (first 60 numeric features)")
    plot_heatmap(spearman, out_dir / "corr_spearman_heatmap.png", "Spearman correlation (first 60 numeric features)")

    # MI feature-label (numeric)
    mi = mi_feature_label(df.drop(columns=artifact_cols), "__label", random_state=args.random_state)
    mi.to_frame("mutual_info").to_csv(out_dir / "mi_numeric_feature_label.csv")

    plt.figure(figsize=(10, 8))
    top_mi = mi.head(25)
    sns.barplot(x=top_mi.values, y=top_mi.index, color="#C44E52")
    plt.title("Top-25 numeric features by mutual information with label")
    plt.tight_layout()
    plt.savefig(out_dir / "mi_top25.png", dpi=160)
    plt.close()

    # Geometry
    geom_dir = out_dir / "geometry"
    pca_info = pca_tsne_embeddings(df.drop(columns=artifact_cols), "__label", geom_dir, random_state=args.random_state)
    (geom_dir / "pca_info.json").write_text(json.dumps(pca_info, indent=2), encoding="utf-8")

    # Importance
    imp_dir = out_dir / "importance"
    _mkdir(imp_dir)
    anova = anova_numeric(df.drop(columns=artifact_cols), "__label", imp_dir)
    chi2_res = chi2_categorical_proxy(df.drop(columns=artifact_cols), "__label", imp_dir)
    proxy_model_importance(df.drop(columns=artifact_cols), "__label", imp_dir, random_state=args.random_state)
    shap_out = shap_tree_proxy(df.drop(columns=artifact_cols), "__label", imp_dir, random_state=args.random_state)

    # FL drift
    fl_dir = out_dir / "fl_simulation"
    _mkdir(fl_dir)
    client_map = simulate_fl_clients(df, "__label")
    (fl_dir / "client_splits.json").write_text(json.dumps(client_map, indent=2), encoding="utf-8")
    drift = drift_ks_vs_global(df.drop(columns=artifact_cols), "__label", client_map, fl_dir)

    # Feature engineering
    fe_dir = out_dir / "feature_engineering"
    _mkdir(fe_dir)
    df_fe = add_engineered_features(df.drop(columns=artifact_cols))
    # Save list of engineered features
    engineered = sorted([c for c in df_fe.columns if c.startswith("fe_")])
    (fe_dir / "engineered_features.json").write_text(json.dumps(engineered, indent=2), encoding="utf-8")

    # quick importance re-run on engineered features only (numeric MI)
    mi_fe = mi_feature_label(df_fe, "__label", random_state=args.random_state)
    mi_fe.to_frame("mutual_info").to_csv(fe_dir / "mi_numeric_with_engineering.csv")

    # ------------------------
    # Build the structured report
    # ------------------------

    sections: Dict[str, str] = {}

    mostly_missing = miss_global[miss_global["missing_pct"] >= 95].reset_index().rename(columns={"index": "feature"})
    always_zero = zero_global[zero_global["zero_pct"] >= 99.999].reset_index().rename(columns={"index": "feature"}) if not zero_global.empty else pd.DataFrame()

    # Leakage-prone inputs: strong identifiers + absolute timestamps.
    leakage_candidates = []
    for c in df.columns:
        cl = c.lower()
        if c in {"__label", "__source_file"}:
            continue
        if cl.endswith("_ip") or cl.endswith("_mac") or cl.endswith("_oui"):
            leakage_candidates.append(c)
        if cl.endswith("_first_seen_ms") or cl.endswith("_last_seen_ms"):
            leakage_candidates.append(c)
        if c in {"id", "expiration_id"}:
            leakage_candidates.append(c)
    leakage_candidates = sorted(set(leakage_candidates))

    sections["1_integrity"] = "\n".join([
        "## 1️⃣ Dataset Integrity & Sanity Checks",
        f"- Total rows: **{len(df)}**",
        f"- Total columns: **{df.shape[1]}** (includes `__label`, `__source_file`) ",
        f"- Files: **{len(refs)}** (see `INDEX.csv`)",
        f"- Duplicate rows: **{dup['exact_duplicate_rows']}** (exact)",
        f"- Duplicate rows excluding [id, __source_file]: **{dup['duplicate_rows_excluding_cols']}**",
        f"- Potential leakage/artifact columns detected: **{artifact_cols if artifact_cols else 'None'}**",
        "- Missingness outputs: `missingness_global.csv`, `missingness_by_class.csv`",
        "- Constant/near-constant outputs: `constant_features.csv`, `near_constant_features.csv`",
        "\nMostly-missing features (missing_pct ≥ 95%):",
        (mostly_missing[["feature", "missing_pct", "nunique_non_null", "dtype"]].head(10).to_markdown(index=False) if not mostly_missing.empty else "(none)"),
        "\nGlobally constant features (safe to drop from modeling unless required for schema):",
        (const_df.head(25).to_markdown(index=False) if not const_df.empty else "(none)"),
        "\nLeakage risk notes (do not blindly trust importance scores on these):",
        "- " + (", ".join(leakage_candidates[:25]) + (" ..." if len(leakage_candidates) > 25 else "") if leakage_candidates else "(none detected)"),
        "\nLeakage risk notes:",
        "- IP/MAC fields (`src_ip`, `dst_ip`, `src_mac`, `dst_mac`, OUI) are strong identifiers; can overfit under random splits.",
        "- Absolute timestamps (`*_first_seen_ms`, `*_last_seen_ms`) can encode collection time windows; treat carefully for FL/generalization.",
    ])

    sections["2_imbalance"] = "\n".join([
        "## 2️⃣ Class Distribution & Imbalance Analysis",
        "Class counts (7-class):",
        label_counts.to_frame("count").to_markdown(),
        "\nPlots:",
        "- `class_counts.png`",
        "- `class_counts_log.png`",
        "\nFL impact:",
        "- Non-IID clients may have minority classes completely absent; evaluation must include per-class recall and macro-averages.",
    ])

    sections["3_sparsity"] = "\n".join([
        "## 3️⃣ Feature Sparsity & Zero-Dominated Features",
        "Outputs:",
        "- `zero_rates_global.csv`",
        "- `zero_rates_by_class.csv`",
        "\nAlways/near-always zero features (zero_pct ≥ 99.999%):",
        (always_zero[["feature", "zero_pct", "dtype"]].head(20).to_markdown(index=False) if not always_zero.empty else "(none)"),
        "\nDecision rule (multiclass + FL-aware):",
        "- Drop features that are **globally always-zero** or **globally constant**.",
        "- Retain features that are sparse globally but **class-specific non-zero** (can be discriminative under partial label visibility).",
    ])

    sections["4_stats"] = "\n".join([
        "## 4️⃣ Statistical Feature Analysis (Global & Class-wise)",
        "Computed for numeric features:",
        "- mean, median, variance, std",
        "- skewness, kurtosis",
        "- IQR, MAD (robust)",
        "\nOutputs:",
        "- `robust_stats_global.csv`",
        "- `robust_stats_by_class_long.csv`",
        "\nInterpretation guidance:",
        "- High skew / kurtosis suggests heavy tails (consider log1p / RobustScaler).",
        "- Compare benign vs each attack class to identify sensitive features.",
    ])

    sections["5_visuals"] = "\n".join([
        "## 5️⃣ Feature Distribution Visualization",
        "Generated visuals (relationships-focused):",
        "- `mi_top25.png` (top MI numeric features)",
        "- `corr_pearson_heatmap.png`, `corr_spearman_heatmap.png`",
        "- `geometry/pca_2d.png`, `geometry/tsne_2d.png`",
        "- `importance/proxy_rf_confusion_matrix.png`",
        "- `importance/shap_summary.png` (if SHAP succeeded)",
        "\nFor deeper per-feature violins/boxplots, we can generate targeted plots for the top-K discriminative features once you confirm K.",
    ])

    sections["6_corr"] = "\n".join([
        "## 6️⃣ Correlation & Redundancy Analysis",
        "Outputs:",
        "- `corr_pearson.csv` + heatmap",
        "- `corr_spearman.csv` + heatmap",
        "- `mi_numeric_feature_label.csv`",
        "\nPruning strategy:",
        "- Identify highly correlated clusters; retain 1 representative per cluster chosen by MI/ANOVA stability across FL clients.",
    ])

    sections["7_geometry"] = "\n".join([
        "## 7️⃣ Class Separability & Geometry",
        "Outputs:",
        "- `geometry/pca_2d.png`",
        "- `geometry/tsne_2d.png`",
        "- `geometry/pca_info.json` (variance explained)",
        "\nInterpretation:",
        "- Overlap between benign and certain attacks indicates hard separation; prioritize robust engineered ratios and UDPS/TCP-flag features.",
    ])

    top_mi_tbl = mi.head(15).to_frame("mutual_info").reset_index().rename(columns={"index": "feature"}).to_markdown(index=False)
    top_anova_tbl = anova[["feature", "f_score", "p_value"]].head(15).to_markdown(index=False)

    sections["8_importance"] = "\n".join([
        "## 8️⃣ Feature Importance WITHOUT Final Model",
        "Computed:",
        "- Mutual information (numeric) → `mi_numeric_feature_label.csv`",
        "- ANOVA F-test (numeric) → `importance/anova_f_scores.csv`",
        "- Chi-square on one-hot encoded features (proxy) → `importance/chi2_top500.csv`",
        "- Proxy RandomForest importances + permutation importances → `importance/proxy_rf_*`",
        "- SHAP (tree proxy, numeric-only) → `importance/shap_summary.png` (if available)",
        "\nTop-15 by MI (numeric):",
        top_mi_tbl,
        "\nTop-15 by ANOVA (numeric):",
        top_anova_tbl,
        "\nImportant caution:",
        "- Extremely high importance for IP/MAC/timestamps can indicate leakage or collection-time artifacts (not robust in FL).",
    ])

    drift_top = drift.sort_values(["client", "ks_stat"], ascending=[True, False]).groupby("client").head(5)
    drift_tbl = drift_top.to_markdown(index=False)

    sections["9_fl"] = "\n".join([
        "## 9️⃣ FL-Aware Feature Robustness Analysis",
        "Client splits (as specified) saved in: `fl_simulation/client_splits.json`",
        "Drift metric:",
        "- KS statistic per numeric feature vs global → `fl_simulation/fl_drift_ks.csv`",
        "- Top-25 drift features per client → `fl_simulation/fl_drift_ks_top25_per_client.csv`",
        "\nTop-5 drift features per client (highest KS):",
        drift_tbl,
        "\nHow to use:",
        "- Prefer features with low drift across clients (more FL-robust).",
        "- Flag features with high drift (client-specific shortcuts).",
    ])

    top_fe_mi_tbl = mi_fe.head(15).to_frame("mutual_info").reset_index().rename(columns={"index": "feature"}).to_markdown(index=False)

    sections["10_fe"] = "\n".join([
        "## 🔟 Feature Engineering (Domain-Aware)",
        "Implemented engineered features (saved list): `feature_engineering/engineered_features.json`",
        "Examples:",
        "- bytes/packet, bytes/ms, packets/ms",
        "- directional ratios (s2d vs d2s)",
        "- TCP-flag rates (syn/rst/fin per packet)",
        "- UDPS normalized rates (per packet)",
        "\nImportance after engineering:",
        "- `feature_engineering/mi_numeric_with_engineering.csv`",
        "\nTop-15 MI after adding engineered features (numeric):",
        top_fe_mi_tbl,
    ])

    sections["11_pipeline"] = "\n".join([
        "## 1️⃣1️⃣ Preprocessing Pipeline Design",
        "Recommended (FL-aware) pipeline structure:",
        "- Numeric: median impute + RobustScaler",
        "- Categorical: most-frequent impute + OneHot (handle_unknown='ignore')",
        "- Add explicit missingness indicators for sparse text fields",
        "\nFL guidance:",
        "- Avoid fitting scalers on global pooled data in real FL; instead fit per-client robust scalers or use federated aggregation of summary stats where possible.",
        "- Keep feature set identical across clients (same columns), even if some classes are absent.",
    ])

    sections["12_outputs"] = "\n".join([
        "## 1️⃣2️⃣ Final Outputs",
        "All outputs are under this folder:",
        f"- `{out_dir}`",
        "\nKey files:",
        "- `REPORT.md` (this report)",
        "- `INDEX.csv`",
        "- `missingness_global.csv`, `missingness_by_class.csv`",
        "- `zero_rates_global.csv`, `zero_rates_by_class.csv`",
        "- `constant_features.csv`, `near_constant_features.csv`",
        "- `importance/` (ANOVA, chi2, proxy RF, SHAP)",
        "- `geometry/` (PCA, t-SNE)",
        "- `fl_simulation/` (drift tables)",
        "- `feature_engineering/`",
    ])

    write_report(out_dir, sections)

    print(f"Done. Report: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
