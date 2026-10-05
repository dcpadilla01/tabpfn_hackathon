import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet").merge(agent_api.train_targets(), on=["household_key","snapshot_day"])
oof["pred"] = oof.oof_med
oof["resid"] = oof.future_spend_4w - oof.pred
# residual vs prediction decile
oof["pb"] = pd.qcut(oof.pred, 10, duplicates="drop")
t = oof.groupby("pb", observed=True).agg(n=("resid","size"), pred=("pred","mean"), tgt=("future_spend_4w","median"), med_resid=("resid","median"), mae=("resid", lambda s: s.abs().mean()))
print(t.round(1))
# per-day optimal shift
for d, gr in oof.groupby("snapshot_day"):
    r = gr.resid.values
    shifts = np.arange(-30,31,2)
    maes = [np.abs(r-s).mean() for s in shifts]
    b = shifts[int(np.argmin(maes))]
    print(d, "base MAE %.2f"%np.abs(r).mean(), "best shift", b, "MAE %.2f"%min(maes))
# binned median calibration lookup (train-like fit on first 3 oof days, eval on day 431)
tr = oof[oof.snapshot_day<=403]; te = oof[oof.snapshot_day==431]
bins = np.quantile(tr.pred, np.linspace(0,1,21))
tr["b"] = pd.cut(tr.pred, bins, include_lowest=True)
lut = tr.groupby("b", observed=True).future_spend_4w.median()
def calib(p):
    idx = pd.cut(pd.Series(p), bins, include_lowest=True).cat.categories
    lab = pd.cut(pd.Series(p), bins, include_lowest=True)
    return lab.map(lut).astype(float).fillna(p).values
pc = calib(te.pred.values)
print("day431 raw MAE %.3f -> calibrated MAE %.3f"%(np.abs(te.resid).mean(), np.abs(te.future_spend_4w-pc).mean()))
