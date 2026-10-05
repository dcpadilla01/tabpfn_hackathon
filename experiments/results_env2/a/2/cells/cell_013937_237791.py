import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet").merge(agent_api.train_targets(), on=["household_key","snapshot_day"])
oof["resid"] = oof.future_spend_4w - oof.oof_med
# 1) capping predictions
for cap in [400, 600, 800, 1000, 1200, 1500, np.inf]:
    print("cap", cap, "MAE %.3f"%np.minimum(oof.resid+0,0).add(0).pipe(lambda s: None) if False else np.abs(np.minimum(oof.oof_med, cap)-oof.future_spend_4w).mean())
# 2) binned-median calibration: fit on days<=403, eval 431
tr = oof[oof.snapshot_day<=403].copy(); te = oof[oof.snapshot_day==431].copy()
qs = np.quantile(tr.oof_med, np.linspace(0,1,26))
tr["b"] = pd.cut(tr.oof_med, qs, include_lowest=True)
lut = tr.groupby("b", observed=True).future_spend_4w.median()
te["b"] = pd.cut(te.oof_med, qs, include_lowest=True)
pc = te["b"].map(lut).astype(float).fillna(te.oof_med)
print("day431 raw MAE %.3f -> binned-median calib MAE %.3f"%(np.abs(te.resid).mean(), np.abs(te.future_spend_4w-pc).mean()))
# 3) blend of oof_med with slight weight on oof_sq / shrinkage of top
for w in [0,0.1,0.2,0.3]:
    p = (1-w)*oof.oof_med + w*oof.oof_sq
    print("blend w_sq=%.1f MAE %.3f"%(w, np.abs(p-oof.future_spend_4w).mean()))
# shrink top tail: p = med if med<=q else q + 0.8*(med-q)
for k in [0.8,0.9,1.0]:
    q = np.quantile(oof.oof_med, 0.95)
    p = np.where(oof.oof_med>q, q + k*(oof.oof_med-q), oof.oof_med)
    print("tail shrink k=%.1f MAE %.3f"%(k, np.abs(p-oof.future_spend_4w).mean()))
