"""
Parses every raw KPI trace CSV from the 5G-production-dataset into a
single, clean, typed pandas DataFrame that the three downstream tasks
(forecasting, anomaly detection, RL resource allocation) all share.

Each source CSV is one continuous session (one video / download run).
We keep that boundary as `session_id` so that lag/rolling features and
train/test splits never leak information across sessions.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .download_data import EXTRACT_DIR, download_and_extract

NUMERIC_COLS = [
    "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI",
    "DL_bitrate", "UL_bitrate", "PINGAVG", "PINGMIN", "PINGMAX",
    "PINGSTDEV", "PINGLOSS",
]

KEEP_COLS = [
    "session_id", "app", "mobility", "step",
    "Timestamp", "Speed", "NetworkMode",
    "RSRP", "RSRQ", "SNR", "CQI", "RSSI",
    "DL_bitrate", "UL_bitrate", "State",
    "PINGAVG", "PINGLOSS",
]


def _infer_labels_from_path(path: str, root: Path) -> tuple[str, str]:
    """5G-production-dataset/<App>/<Mobility>/<Show>/<file>.csv"""
    rel = Path(path).relative_to(root)
    parts = rel.parts
    app = parts[0] if len(parts) > 0 else "unknown"
    mobility = parts[1] if len(parts) > 1 else "unknown"
    return app, mobility


def _load_one_csv(path: str, root: Path, session_id: int) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, na_values=["-"], low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[data_loader] skipping unreadable file {path}: {exc}")
        return None

    if "DL_bitrate" not in df.columns:
        return None

    app, mobility = _infer_labels_from_path(path, root)
    df["app"] = app
    df["mobility"] = mobility
    df["session_id"] = session_id
    df["step"] = np.arange(len(df))

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "NetworkMode" not in df.columns:
        df["NetworkMode"] = "unknown"
    if "State" not in df.columns:
        df["State"] = "unknown"

    return df[[c for c in KEEP_COLS if c in df.columns]]


def load_dataset(max_sessions: int | None = None, force_download: bool = False) -> pd.DataFrame:
    """Loads all (or up to `max_sessions`) real KPI traces into one
    long-format DataFrame, one row per measurement, grouped by
    `session_id`.
    """
    root = download_and_extract(force=force_download)
    csv_paths = sorted(glob.glob(str(root) + "/**/*.csv", recursive=True))
    if max_sessions is not None:
        csv_paths = csv_paths[:max_sessions]

    frames = []
    for i, path in enumerate(csv_paths):
        df = _load_one_csv(path, root, session_id=i)
        if df is not None and len(df) > 20:  # drop degenerate/near-empty traces
            frames.append(df)

    if not frames:
        raise RuntimeError("No usable CSV traces were parsed from the dataset.")

    full = pd.concat(frames, ignore_index=True)

    # Basic cleaning: forward-fill small channel-quality gaps within a
    # session (radio KPIs are reported at slightly irregular intervals),
    # then drop rows still missing the forecasting target.
    full = full.sort_values(["session_id", "step"]).reset_index(drop=True)
    channel_cols = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
    full[channel_cols] = full.groupby("session_id")[channel_cols].ffill().bfill()
    full["PINGLOSS"] = full["PINGLOSS"].fillna(0)
    full = full.dropna(subset=["DL_bitrate"]).reset_index(drop=True)

    print(
        f"[data_loader] Loaded {full['session_id'].nunique()} sessions, "
        f"{len(full)} total rows. Apps: {sorted(full['app'].unique())}"
    )
    return full


if __name__ == "__main__":
    df = load_dataset()
    print(df.head(10))
    print(df.describe(numeric_only=True))
