"""Rigorous model comparison (extends the baseline task1/task2 scripts).

Fixes / additions vs. the baseline implementation
  1. Leak-free features: rolling windows are computed PER SESSION (the
     baseline's rolling() ran over the whole concatenated column and so
     leaked values across session boundaries).
  2. True one-step-ahead: current value DL_bitrate[t] is a feature
     (baseline features start at lag1, so every model was really
     predicting 2 steps ahead; LSTM and trees were consistent, but
     persistence was impossible to express).
  3. Baselines: persistence (next = current) and Ridge. Any ML model must
     beat these or it adds nothing.
  4. GroupKFold over SESSIONS -> mean +/- std instead of one split.
  5. Anomaly threshold for the LSTM-residual detector is fixed on TRAIN
     data (baseline used the test-set positive rate = label leakage).
  6. Extra metrics: MAPE-free relative error (RMSE / mean), R2, and
     PR-AUC for the imbalanced anomaly task.

Usage:  python -m src.benchmark --folds 3 --lstm-epochs 6
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from .data_loader import load_dataset
from .numpy_lstm import NumpyLSTMRegressor

OUT = Path(__file__).resolve().parent.parent / "outputs"
LAGS = (0, 1, 2, 3, 5)           # 0 = current value (needed for true 1-step)
ROLLS = (5, 10)
CH = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed"]
SEQ = ["DL_bitrate", "RSRP", "RSRQ", "SNR", "CQI"]
WIN = 10


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["session_id", "step"]).reset_index(drop=True).copy()
    g = df.groupby("session_id")["DL_bitrate"]
    for l in LAGS:
        df[f"lag{l}"] = g.shift(l)
    for w in ROLLS:   # per-session, includes current value, no cross-session leak
        df[f"rm{w}"] = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"rs{w}"] = g.transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
    for c in CH:
        df[f"{c}_c"] = df[c]
        df[f"{c}_d1"] = df.groupby("session_id")[c].diff().fillna(0)
    for c in ("app", "mobility", "NetworkMode"):
        df[c + "_code"] = df[c].astype("category").cat.codes
    df["y_next"] = g.shift(-1)
    df["anomaly_next"] = (
        (df["rm5"] > 500) & (df["y_next"] < 0.10 * df["rm5"])).astype(int)
    return df.dropna(subset=["lag5", "y_next"]).reset_index(drop=True)


FEATS = ([f"lag{l}" for l in LAGS] + [f"rm{w}" for w in ROLLS] + [f"rs{w}" for w in ROLLS]
         + [f"{c}_c" for c in CH] + [f"{c}_d1" for c in CH]
         + ["app_code", "mobility_code", "NetworkMode_code"])


def sequences(d: pd.DataFrame, ycol: str):
    """Window ENDS at t (inclusive) and predicts y_next of row t."""
    X, y, idx = [], [], []
    for _, g in d.groupby("session_id"):
        v = g[SEQ].to_numpy(np.float32); t = g[ycol].to_numpy(np.float32); ix = g.index.to_numpy()
        for i in range(WIN - 1, len(g)):
            X.append(v[i - WIN + 1:i + 1]); y.append(t[i]); idx.append(ix[i])
    return np.stack(X), np.array(y), np.array(idx)


def reg_metrics(y, p):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, p))), mae=float(mean_absolute_error(y, p)),
                r2=float(r2_score(y, p)))


def run_lstm(tr, te, ycol, epochs):
    sc = StandardScaler().fit(tr[SEQ]); a, b = tr.copy(), te.copy()
    a[SEQ] = sc.transform(tr[SEQ]); b[SEQ] = sc.transform(te[SEQ])
    Xa, ya, _ = sequences(a, ycol); Xb, yb, ib = sequences(b, ycol)
    ys = StandardScaler().fit(ya.reshape(-1, 1))
    m = NumpyLSTMRegressor(len(SEQ), hidden_size=24, seed=0)
    m.fit(Xa, ys.transform(ya.reshape(-1, 1)).ravel().astype(np.float32), epochs=epochs,
          batch_size=128, lr=0.01, verbose=False)
    p = ys.inverse_transform(m.predict(Xb).reshape(-1, 1)).ravel()
    ptr = ys.inverse_transform(m.predict(Xa).reshape(-1, 1)).ravel()
    return p, yb, ib, ptr, ya


def forecasting(df, folds, epochs):
    rows = []
    for k, (tri, tei) in enumerate(GroupKFold(folds).split(df, groups=df.session_id)):
        tr, te = df.iloc[tri], df.iloc[tei]
        Xa, ya, Xb, yb = tr[FEATS], tr.y_next, te[FEATS], te.y_next.values
        rows.append(dict(fold=k, model="persistence", train_s=0, **reg_metrics(yb, te.lag0.values)))
        zoo = {"Ridge": lambda: __import__("sklearn.pipeline", fromlist=["x"]).make_pipeline(StandardScaler(), Ridge(1.0)),
               "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1),
               "CatBoost": lambda: CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=False),
               "RandomForest": lambda: RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=0)}
        for n, mk in zoo.items():
            t0 = time.time(); m = mk().fit(Xa, ya)
            rows.append(dict(fold=k, model=n, train_s=time.time() - t0, **reg_metrics(yb, m.predict(Xb))))
        t0 = time.time(); p, yy, _, _, _ = run_lstm(tr, te, "y_next", epochs)
        rows.append(dict(fold=k, model="NumpyLSTM", train_s=time.time() - t0, **reg_metrics(yy, p)))
        print(f"[forecast] fold {k+1}/{folds} done", flush=True)
    return pd.DataFrame(rows)


def clf_metrics(y, s, thr):
    pr = (s >= thr).astype(int)
    return dict(precision=precision_score(y, pr, zero_division=0), recall=recall_score(y, pr, zero_division=0),
                f1=f1_score(y, pr, zero_division=0), roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))


def best_thr(y, s):
    qs = np.quantile(s, np.linspace(0.5, 0.99, 50))
    return max(qs, key=lambda q: f1_score(y, (s >= q).astype(int), zero_division=0))


def anomaly(df, folds, epochs):
    rows = []
    for k, (tri, tei) in enumerate(GroupKFold(folds).split(df, groups=df.session_id)):
        tr, te = df.iloc[tri], df.iloc[tei]
        ya, yb = tr.anomaly_next, te.anomaly_next.values
        # "dumb" baseline: a drop-in persistence rule on the current value
        rule = lambda d: np.where(d.rm5.values > 500, -(d.lag0 / (d.rm5 + 1e-6)).values, -9.0)  # gated on activity
        thr0 = np.quantile(rule(tr), 1 - ya.mean())            # threshold from TRAIN only
        rows.append(dict(fold=k, model="rule(lag0/rm5)", **clf_metrics(yb, rule(te), thr0)))
        zoo = {"LogReg": lambda: __import__("sklearn.pipeline", fromlist=["x"]).make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")),
               "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1),
               "CatBoost": lambda: CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=False),
               "RandomForest": lambda: RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=0)}
        for n, mk in zoo.items():
            m = mk().fit(tr[FEATS], ya)
            thr = best_thr(ya.values, m.predict_proba(tr[FEATS])[:, 1])   # chosen on TRAIN only
            rows.append(dict(fold=k, model=n, **clf_metrics(yb, m.predict_proba(te[FEATS])[:, 1], thr)))
        # LSTM residual: threshold from TRAIN residual quantile = train anomaly rate
        p, _, ib, ptr, ytr_seq = run_lstm(tr, te, "y_next", epochs)
        cur_te = te.loc[ib, "lag0"].values
        rm_te = te.loc[ib, "rm5"].values
        res_te = np.where(rm_te > 500, (p - te.loc[ib, "y_next"].values) / (rm_te + 1), -9.0)  # signed: predicted-but-did-not-happen
        ib_tr = sequences(tr, "y_next")[2]
        rm_tr = tr.loc[ib_tr, "rm5"].values
        res_tr = np.where(rm_tr > 500, (ptr - tr.loc[ib_tr, "y_next"].values) / (rm_tr + 1), -9.0)
        thr = np.quantile(res_tr, 1 - ya.mean())
        rows.append(dict(fold=k, model="LSTM-residual", **clf_metrics(te.loc[ib, "anomaly_next"].values, res_te, thr)))
        print(f"[anomaly] fold {k+1}/{folds} done", flush=True)
    return pd.DataFrame(rows)


def summarize(d, cols):
    g = d.groupby("model")[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lstm-epochs", type=int, default=6)
    ap.add_argument("--task", choices=["forecast", "anomaly", "both"], default="both")
    ap.add_argument("--max-sessions", type=int, default=None)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True)
    df = build(load_dataset(max_sessions=a.max_sessions))
    print(f"[benchmark] rows={len(df)} sessions={df.session_id.nunique()} anomaly_rate={df.anomaly_next.mean():.2%}")
    pd.set_option("display.width", 200)
    if a.task in ("forecast", "both"):
        f = forecasting(df, a.folds, a.lstm_epochs); f.to_csv(OUT / "bench_forecast_folds.csv", index=False)
        s = summarize(f, ["rmse", "mae", "r2", "train_s"]).sort_values("rmse_mean"); s.to_csv(OUT / "bench_forecast.csv")
        print("\n=== FORECASTING (GroupKFold by session) ===\n", s.round(3).to_string())
    if a.task in ("anomaly", "both"):
        d = anomaly(df, a.folds, a.lstm_epochs); d.to_csv(OUT / "bench_anomaly_folds.csv", index=False)
        s = summarize(d, ["precision", "recall", "f1", "roc_auc", "pr_auc"]).sort_values("f1_mean", ascending=False)
        s.to_csv(OUT / "bench_anomaly.csv")
        print("\n=== ANOMALY (GroupKFold by session) ===\n", s.round(3).to_string())
