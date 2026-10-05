import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
CAT = ["classification_1","classification_3","classification_4","classification_5","classification_2","homeowner","kids"]
for c in CAT: df[c] = df[c].astype("category")
tr = df[df.snapshot_day <= 431].copy()
va = df[df.snapshot_day >= 459].copy()
print("train rows", len(tr), "val rows", len(va), "nan cols:", [c for c in FE if tr[c].isna().any()])

PARAMS = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
              tree_method="hist", enable_categorical=True, n_jobs=4)

# reproduce E005: quantile 0.5 on full train
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **PARAMS)
m.fit(tr[FE], tr.future_spend_4w)
p = m.predict(va[FE])
old = agent_api.load_saved("pred_e005.parquet").sort_values(["household_key","snapshot_day"]).prediction.values
va_sorted = va.sort_values(["household_key","snapshot_day"])
print("MAE repro:", mean_absolute_error(va_sorted.future_spend_4w, p), " vs old:", mean_absolute_error(va_sorted.future_spend_4w, old), " corr:", np.corrcoef(p, old)[0,1])
