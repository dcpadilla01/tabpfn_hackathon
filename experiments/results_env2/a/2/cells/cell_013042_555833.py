import agent_api, pandas as pd, numpy as np, time
f3 = agent_api.load_saved("feats_v3.parquet")
fs = agent_api.load_saved("feats_seasonal.parquet")
m = f3.merge(fs, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape, "dtypes non-numeric:", sum(m.dtypes != "number"))
agent_api.save_table(m, "feats_v6.parquet")
tt = agent_api.train_targets()
tr = m[m.snapshot_day.isin(agent_api.snapshot_days()["train"])].merge(tt, on=["household_key","snapshot_day"])
print("train rows:", tr.shape, "val rows:", m[m.snapshot_day>=459].shape)

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
FEATS = [c for c in m.columns if c not in ("household_key","snapshot_day")]
Xtr = tr[FEATS].values; ytr = tr.future_spend_4w.values
t0=time.time()
mdl = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=10,
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                       objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=0)
mdl.fit(Xtr, ytr)
print("xgb 800 fit time:", round(time.time()-t0,1))
t0=time.time()
h = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, loss="quantile", quantile=0.5, random_state=0)
h.fit(Xtr, ytr)
print("hgb 400 fit time:", round(time.time()-t0,1))
