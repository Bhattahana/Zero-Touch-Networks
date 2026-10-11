"""
Shared feature engineering for the tabular models (LightGBM, CatBoost,
Random Forest) and windowing for the NumPy LSTM.

Everything is computed per `session_id` so no lag/rolling window ever
crosses a session boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 5)
ROLL_WINDOWS = (5, 10)
BASE_FEATURES = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("session_id")["DL_bitrate"]

    for lag in LAGS:
        df[f"DL_bitrate_lag{lag}"] = g.shift(lag)

    for w in ROLL_WINDOWS:
        df[f"DL_bitrate_rollmean{w}"] = (
            g.shift(1).rolling(w, min_periods=1).mean().reset_index(drop=True)
        )
        df[f"DL_bitrate_rollstd{w}"] = (
            g.shift(1).rolling(w, min_periods=1).std().reset_index(drop=True).fillna(0)
        )

    for col in BASE_FEATURES:
        df[f"{col}_lag1"] = df.groupby("session_id")[col].shift(1)

    df["app_code"] = df["app"].astype("category").cat.codes
    df["mobility_code"] = df["mobility"].astype("category").cat.codes
    df["networkmode_code"] = df["NetworkMode"].astype("category").cat.codes

    # target: NEXT bitrate value (one-step-ahead forecast)
    df["target_next_bitrate"] = df.groupby("session_id")["DL_bitrate"].shift(-1)

    df = df.dropna(subset=[f"DL_bitrate_lag{max(LAGS)}", "target_next_bitrate"]).reset_index(drop=True)
    return df


TABULAR_FEATURE_COLS = (
    [f"DL_bitrate_lag{l}" for l in LAGS]
    + [f"DL_bitrate_rollmean{w}" for w in ROLL_WINDOWS]
    + [f"DL_bitrate_rollstd{w}" for w in ROLL_WINDOWS]
    + [f"{c}_lag1" for c in BASE_FEATURES]
    + ["app_code", "mobility_code", "networkmode_code"]
)


def session_train_test_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Splits by whole session (not by row) so no trace leaks between
    train and test.
    """
    rng = np.random.default_rng(seed)
    sessions = df["session_id"].unique()
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_frac))
    test_sessions = set(sessions[:n_test])
    test_mask = df["session_id"].isin(test_sessions)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


def make_sequences(df: pd.DataFrame, seq_cols: list[str], target_col: str, window: int = 10):
    """Builds sliding-window sequences per session for the LSTM:
    X: (N, window, n_features), y: (N,)
    """
    X_list, y_list = [], []
    for _, g in df.groupby("session_id"):
        g = g.reset_index(drop=True)
        values = g[seq_cols].to_numpy(dtype=np.float32)
        target = g[target_col].to_numpy(dtype=np.float32)
        for t in range(window, len(g)):
            X_list.append(values[t - window : t])
            y_list.append(target[t])
    if not X_list:
        return np.empty((0, window, len(seq_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list), np.array(y_list, dtype=np.float32)
