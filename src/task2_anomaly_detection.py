"""
Task 2 — Zero-touch fault / anomaly detection.

A ZTM control loop needs to flag QoS-degrading events (e.g. a sudden
throughput collapse / stall) from channel telemetry so it can trigger
autonomous remediation (handover, slice re-provisioning, etc.).

Ground truth here is rule-based (a common, defensible approach when no
manually-labelled fault dataset exists): a row is labelled anomalous
if throughput collapses to <10% of its own trailing 5-sample average,
after having been meaningfully active (trailing average > 500 kbps).
This captures real stall/interruption events visible in the raw KPI
traces.

Four detectors are compared:
    - LightGBM / CatBoost / Random Forest  -> supervised classifiers
      on the channel-quality + throughput-history features
    - NumPy LSTM residual score             -> reuses the forecaster
      from Task 1: a large gap between predicted and actual next
      throughput is itself evidence of an anomalous transition
      (classic forecast-residual anomaly detection).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostClassifier

from .data_loader import load_dataset
from .features import (
    TABULAR_FEATURE_COLS,
    add_lag_and_rolling_features,
    make_sequences,
    session_train_test_split,
)
from .numpy_lstm import NumpyLSTMRegressor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

ROLLMEAN_MIN = 500.0
COLLAPSE_RATIO = 0.10


def label_anomalies(df):
    df = df.copy()
    df["anomaly"] = (
        (df["DL_bitrate_rollmean5"] > ROLLMEAN_MIN)
        & (df["DL_bitrate"] < COLLAPSE_RATIO * df["DL_bitrate_rollmean5"])
    ).astype(int)
    return df


def _clf_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
        "positive_rate": float(np.mean(y_true)),
    }


def run(max_sessions: int | None = None) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    print("\n=== TASK 2: Anomaly / fault detection ===")
    raw = load_dataset(max_sessions=max_sessions)
    df = add_lag_and_rolling_features(raw)
    df = label_anomalies(df)
    train_df, test_df = session_train_test_split(df, test_frac=0.2)
    print(
        f"[task2] train rows={len(train_df)} (pos={train_df['anomaly'].mean():.2%})  "
        f"test rows={len(test_df)} (pos={test_df['anomaly'].mean():.2%})"
    )

    Xtr, ytr = train_df[TABULAR_FEATURE_COLS], train_df["anomaly"]
    Xte, yte = test_df[TABULAR_FEATURE_COLS], test_df["anomaly"]

    results = {}

    # ---- LightGBM ----
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    lgbm.fit(Xtr, ytr)
    score = lgbm.predict_proba(Xte)[:, 1]
    results["lightgbm"] = {**_clf_metrics(yte, (score > 0.5).astype(int), score),
                            "train_seconds": round(time.time() - t0, 2)}

    # ---- CatBoost ----
    t0 = time.time()
    cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=False)
    cat.fit(Xtr, ytr)
    score = cat.predict_proba(Xte)[:, 1]
    results["catboost"] = {**_clf_metrics(yte, (score > 0.5).astype(int), score),
                            "train_seconds": round(time.time() - t0, 2)}

    # ---- Random Forest ----
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    score = rf.predict_proba(Xte)[:, 1]
    results["random_forest"] = {**_clf_metrics(yte, (score > 0.5).astype(int), score),
                                 "train_seconds": round(time.time() - t0, 2)}

    # ---- NumPy LSTM residual anomaly score ----
    t0 = time.time()
    seq_cols = ["DL_bitrate", "RSRP", "RSRQ", "SNR", "CQI"]
    scaler = StandardScaler().fit(train_df[seq_cols])
    train_scaled, test_scaled = train_df.copy(), test_df.copy()
    train_scaled[seq_cols] = scaler.transform(train_df[seq_cols])
    test_scaled[seq_cols] = scaler.transform(test_df[seq_cols])

    window = 10
    Xtr_seq, _ = make_sequences(train_scaled, seq_cols, "DL_bitrate", window=window)
    Xte_seq, _ = make_sequences(test_scaled, seq_cols, "DL_bitrate", window=window)

    def aligned(colname, source_df):
        out = []
        for _, g in source_df.reset_index(drop=True).groupby("session_id"):
            g = g.reset_index(drop=True)
            vals = g[colname].to_numpy(dtype=np.float32)
            for t in range(window, len(g)):
                out.append(vals[t])
        return np.array(out, dtype=np.float32)

    ytr_next = aligned("target_next_bitrate", train_df)
    yte_next = aligned("target_next_bitrate", test_df)
    yte_anom = aligned("anomaly", test_df)

    target_scaler = StandardScaler().fit(ytr_next.reshape(-1, 1))
    ytr_next_scaled = target_scaler.transform(ytr_next.reshape(-1, 1)).reshape(-1).astype(np.float32)

    lstm = NumpyLSTMRegressor(n_features=len(seq_cols), hidden_size=24, seed=0)
    lstm.fit(Xtr_seq, ytr_next_scaled, epochs=10, batch_size=128, lr=0.01, verbose=False)
    pred_scaled = lstm.predict(Xte_seq)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    residual = np.abs(pred - yte_next)
    # normalize residual to a 0..1 anomaly score
    score = (residual - residual.min()) / (residual.max() - residual.min() + 1e-9)
    thresh = np.quantile(score, 1 - max(yte_anom.mean(), 0.05))
    pred_label = (score > thresh).astype(int)
    results["lstm_residual"] = {**_clf_metrics(yte_anom, pred_label, score),
                                 "train_seconds": round(time.time() - t0, 2)}

    (OUT_DIR / "task2_anomaly_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
