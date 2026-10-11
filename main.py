"""Zero-Touch Management backend - CLI.

  python main.py all                 # whole pipeline
  python main.py generate            # synthetic KPI data -> data/kpis.csv
  python main.py faults              # self-healing: fault detection comparison
  python main.py forecast            # self-optimisation: traffic forecasting comparison
  python main.py loop                # closed-loop autoscaling comparison
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ztm import data, features, models, closedloop

OUT = "results"; os.makedirs(OUT, exist_ok=True); os.makedirs("data", exist_ok=True)


def load():
    p = "data/kpis.csv"
    if not os.path.exists(p):
        data.generate().to_csv(p, index=False)
    return pd.read_csv(p)


def split_time(df, frac=0.7):
    cut = int(df.step.max() * frac)
    return df.step <= cut, df.step > cut


def run_faults(df):
    X = features.fault_features(df); y = (df.fault > 0).astype(int)
    tr, te = split_time(df)
    res, probs = [], {}
    for n, m in models.fault_models().items():
        r, p = models.eval_fault(n, m, X[tr], y[tr], X[te], y[te]); res.append(r); probs[n] = p
    r, p = models.eval_isolation(X[tr].values, y[tr].values, X[te].values, y[te].values); res.append(r)
    out = pd.DataFrame(res).sort_values("f1", ascending=False)
    out.to_csv(f"{OUT}/fault_detection.csv", index=False)
    print("\n=== SELF-HEALING: fault detection (time-based split) ===")
    print(out.round(3).to_string(index=False))
    ax = out.set_index("model")[["precision", "recall", "f1"]].plot.bar(figsize=(8, 4), title="Fault detection")
    plt.tight_layout(); plt.savefig(f"{OUT}/fault_detection.png", dpi=120); plt.close()
    return out


def run_forecast(df):
    ds = features.forecast_dataset(df)
    cut = int(df.step.max() * 0.7)
    tr, te = ds.step <= cut, ds.step > cut
    feats = [c for c in ds.columns if c not in ("cell", "step", "y", "now")]
    Xtr, ytr, Xte, yte = ds.loc[tr, feats], ds.y[tr], ds.loc[te, feats], ds.y[te].values
    res, preds = [], {}
    naive = ds.now[te].values
    res.append(dict(model="Naive(persistence)", MAE=np.abs(yte - naive).mean(),
                    RMSE=float(np.sqrt(((yte - naive) ** 2).mean())),
                    MAPE_pct=float(np.mean(np.abs(yte - naive) / np.maximum(yte, 1)) * 100), train_s=0))
    for n, m in models.forecast_models().items():
        r, p = models.eval_forecast(n, m, Xtr, ytr, Xte, yte); res.append(r); preds[n] = p
    out = pd.DataFrame(res).sort_values("RMSE")
    out.to_csv(f"{OUT}/forecast.csv", index=False)
    print("\n=== SELF-OPTIMISATION: traffic forecast, 30 min ahead ===")
    print(out.round(3).to_string(index=False))
    out.set_index("model")[["MAE", "RMSE"]].plot.bar(figsize=(8, 4), title="Forecast error (lower=better)")
    plt.tight_layout(); plt.savefig(f"{OUT}/forecast.png", dpi=120); plt.close()
    best = out[out.model.isin(preds)].iloc[0].model
    return out, best, ds, te, preds


def run_loop(df, ds, te, preds, best):
    rows = []
    tdf = ds[te].copy(); tdf["pred"] = preds[best]
    for pol in ["static", "reactive", "predictive"]:
        agg = []
        for c, d in tdf.groupby("cell"):
            d = d.sort_values("step")
            dem = d.now.values; fc = np.roll(d.pred.values, 6)  # forecast made 6 steps ago for now
            agg.append(closedloop.simulate(dem, fc, pol))
        a = pd.DataFrame(agg)
        rows.append(dict(policy=pol + (f"({best})" if pol == "predictive" else ""),
                         sla_violation_pct=a.sla_violation_pct.mean(), cost=a.cost.sum(),
                         actions=a.actions.sum(), escalated_to_human=a.escalated_to_human.sum()))
    out = pd.DataFrame(rows); out.to_csv(f"{OUT}/closed_loop.csv", index=False)
    print("\n=== CLOSED LOOP: autoscaling policies ===")
    print(out.round(2).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "generate", "faults", "forecast", "loop"])
    a = ap.parse_args()
    if a.cmd == "generate":
        data.generate().to_csv("data/kpis.csv", index=False); print("saved data/kpis.csv"); return
    df = load()
    if a.cmd == "faults": run_faults(df)
    elif a.cmd in ("forecast", "loop", "all"):
        if a.cmd == "all": run_faults(df)
        f, best, ds, te, preds = run_forecast(df)
        if a.cmd in ("loop", "all"): run_loop(df, ds, te, preds, best)


if __name__ == "__main__":
    main()
