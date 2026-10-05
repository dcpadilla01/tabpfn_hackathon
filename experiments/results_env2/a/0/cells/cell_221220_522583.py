import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
tr = df[df.snapshot_day <= 431].copy()
va = df[df.snapshot_day >= 459].copy()

PARAMS = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              tree_method="hist", n_jobs=4)
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m.fit(tr[FE], tr.future_spend_4w)
p = m.predict(va[FE])
old = agent_api.load_saved("pred_e005.parquet").sort_values(["household_key","snapshot_day"]).prediction.values
print("corr with old E005 preds:", np.corrcoef(p, old)[0,1], " mean|diff|:", np.abs(p-old).mean())

# target stats by snapshot day (train only)
print(df.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))

# local pseudo-val: train on <=403, eval on 431
tr2 = df[df.snapshot_day <= 403]
pv = df[df.snapshot_day == 431]
m2 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m2.fit(tr2[FE], tr2.future_spend_4w)
pp = m2.predict(pv[FE])
print("pseudo-val 431 MAE:", mean_absolute_error(pv.future_spend_4w, pp))
# naive baselines on 431
print("lag0 (spend_28) MAE:", mean_absolute_error(pv.future_spend_4w, pv.spend_28))
print("exp4w_blend MAE:", mean_absolute_error(pv.future_spend_4w, pv.exp4w_blend))
