import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
r = oof.future_spend_4w - oof.oof_med
print("Median-objective model: median residual by predicted-value decile")
bins = pd.qcut(oof.oof_med, 10, duplicates="drop")
d = pd.DataFrame({"bin":bins, "res":r, "pred":oof.oof_med, "y":oof.future_spend_4w}).groupby("bin", observed=True)
tab = d.agg(n=("res","size"), pred_med=("pred","median"), y_med=("y","median"), res_med=("res","median"), mae=("res", lambda s: s.abs().mean()))
print(tab.round(2).to_string())
print("\nMAE now:", round(np.abs(r).mean(),3))
# apply per-bin median shift
shift = d.res.median()
adj = r - oof.bin.map(shift) if False else r - bins.map(shift)
print("MAE after per-bin median shift:", round(np.abs(adj).mean(),3))
# also by snapshot day x bin for stability check
print("\nres_med median by day:", oof.groupby("snapshot_day").res_med.median().round(2).to_dict())
# zero-activity segment: households with oof_med == 0
z = oof[oof.oof_med==0]
print("\nrows with pred==0:", len(z), "their y median:", z.future_spend_4w.median(), "MAE contribution:", round(np.abs(z.future_spend_4w).mean(),2))
nz = oof[oof.oof_med>0]
print("rows pred>0:", len(nz), "res median:", round((nz.future_spend_4w-nz.oof_med).median(),2))
