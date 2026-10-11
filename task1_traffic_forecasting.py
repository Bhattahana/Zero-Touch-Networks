"""
Task 1 — Zero-touch traffic forecasting.

Predicts next-step downlink throughput (DL_bitrate) from real 5G/LTE
client KPI traces, comparing four model families:

    - LightGBM        (gradient-boosted trees, tabular lag features)
    - CatBoost         (gradient-boosted trees, tabular lag features)
    - Random Forest    (bagged trees, tabular lag features)
    - NumPy LSTM       (from-scratch RNN, raw KPI sequences)

A ZTM controller uses this kind of forecast to anticipate congestion
and pre-emptively reconfigure RAN/slice resources before QoS degrades.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostRegressor

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


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run(max_sessions: int | None = None, lstm_epochs: int = 12) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    print("\n=== TASK 1: Traffic forecasting ===")
    raw = load_dataset(max_sessions=max_sessions)
    df = add_lag_and_rolling_features(raw)
    train_df, test_df = session_train_test_split(df, test_frac=0.2)
    print(f"[task1] train rows={len(train_df)}  test rows={len(test_df)}")

    Xtr, ytr = train_df[TABULAR_FEATURE_COLS], train_df["target_next_bitrate"]
    Xte, yte = test_df[TABULAR_FEATURE_COLS], test_df["target_next_bitrate"]

    results = {}

    # ---- LightGBM ----
    t0 = time.time()
    lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
    lgbm.fit(Xtr, ytr)
    pred = lgbm.predict(Xte)
    results["lightgbm"] = {**_metrics(yte, pred), "train_seconds": round(time.time() - t0, 2)}

    # ---- CatBoost ----
    t0 = time.time()
    cat = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=False)
    cat.fit(Xtr, ytr)
    pred = cat.predict(Xte)
    results["catboost"] = {**_metrics(yte, pred), "train_seconds": round(time.time() - t0, 2)}

    # ---- Random Forest ----
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    results["random_forest"] = {**_metrics(yte, pred), "train_seconds": round(time.time() - t0, 2)}

    # ---- NumPy LSTM (sequence model on raw KPI history) ----
    t0 = time.time()
    seq_cols = ["DL_bitrate", "RSRP", "RSRQ", "SNR", "CQI"]
    scaler = StandardScaler().fit(train_df[seq_cols])
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[seq_cols] = scaler.transform(train_df[seq_cols])
    test_scaled[seq_cols] = scaler.transform(test_df[seq_cols])

    window = 10
    Xtr_seq, _ = make_sequences(train_scaled, seq_cols, "DL_bitrate", window=window)
    Xte_seq, _ = make_sequences(test_scaled, seq_cols, "DL_bitrate", window=window)

    # target used for the LSTM is the *next* raw bitrate, aligned the
    # same way make_sequences aligns X, then scaled with its own
    # StandardScaler (fit on train only) so the loss/gradients operate
    # on a well-conditioned range instead of raw kbps magnitudes.
    def targets_next(df_src, window):
        ys = []
        for _, g in df_src.reset_index(drop=True).groupby("session_id"):
            g = g.reset_index(drop=True)
            vals = g["target_next_bitrate"].to_numpy(dtype=np.float32)
            for t in range(window, len(g)):
                ys.append(vals[t])
        return np.array(ys, dtype=np.float32)

    ytr_raw = targets_next(train_df, window)
    yte_raw = targets_next(test_df, window)

    target_scaler = StandardScaler().fit(ytr_raw.reshape(-1, 1))
    ytr_seq = target_scaler.transform(ytr_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    lstm = NumpyLSTMRegressor(n_features=len(seq_cols), hidden_size=24, seed=0)
    lstm.fit(Xtr_seq, ytr_seq, epochs=lstm_epochs, batch_size=128, lr=0.01, verbose=True)
    pred_scaled = lstm.predict(Xte_seq)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    results["numpy_lstm"] = {**_metrics(yte_raw, pred), "train_seconds": round(time.time() - t0, 2)}

    (OUT_DIR / "task1_forecasting_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
