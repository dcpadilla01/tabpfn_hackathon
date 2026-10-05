import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error
feats = agent_api.load_saved("feats_v2.parquet").copy()
feats["cls2"] = feats["cls2"].astype(float)
t = agent_api.train_targets()
tr = feats.merge(t, on=["household_key","snapshot_day"])
drop = ["household_key","snapshot_day","future_spend_4w"]
val = tr[tr.snapshot_day==431]; trn = tr[tr.snapshot_day!=431]
m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, n_jobs=4)
m.fit(trn.drop(columns=drop), trn.future_spend_4w, eval_set=[(val.drop(columns=drop), val.future_spend_4w)], verbose=False)
p = m.predict(val.drop(columns=drop))
print("holdout 431 MAE", mean_absolute_error(val.future_spend_4w, p))
va = feats[feats.snapshot_day>=459].copy()
va["prediction"] = m.predict(va.drop(columns=drop))
out = va[["household_key","snapshot_day","prediction"]]
agent_api.save_table(out, "pred_e002.parquet")
imp = pd.Series(m.feature_importances_, index=trn.drop(columns=drop).columns).sort_values(ascending=False)
print(imp.head(20))
print("n val rows", len(va))
