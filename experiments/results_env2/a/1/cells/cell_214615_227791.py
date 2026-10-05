import numpy as np, pandas as pd, xgboost as xgb

f = agent_api.load_saved("e002_features.parquet")
t = agent_api.train_targets()
d = f.merge(t, on=["household_key","snapshot_day"])
feats = [c for c in f.columns if c not in ("household_key","snapshot_day")]
d[feats] = d[feats].astype(float)
val = f[f.snapshot_day>=459].copy()
val[feats] = val[feats].astype(float)

params = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              objective="reg:squarederror", tree_method="hist", n_jobs=4)
model = xgb.XGBRegressor(**params)
model.fit(d[feats], d.future_spend_4w, eval_set=[(val[feats], val.future_spend_4w if 'future_spend_4w' in val else np.zeros(len(val)))], verbose=False)
