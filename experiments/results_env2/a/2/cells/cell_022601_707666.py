import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb
print("xgb", xgb.__version__)

v3 = A.load_saved("feats_v3.parquet")
se = A.load_saved("feats_seasonal.parquet")
m = v3.merge(se.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else v3.merge(se, on=["household_key","snapshot_day"], how="inner", suffixes=("","_se"))
print("merged", m.shape)
feat_cols = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("n_feat", len(feat_cols))
print(m[feat_cols].dtypes.value_counts())
A.save_table(m, "feats_all_e016.parquet")

tt = A.train_targets()
tr = m.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("train rows w/ target:", tr.shape, "target mean", tr.future_spend_4w.mean().round(2), "median", tr.future_spend_4w.median())

# time one quantile fit
X = tr[feat_cols].values.astype(np.float32); y = tr.future_spend_4w.values.astype(np.float32)
t0=time.time()
mdl = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=10,
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                       objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=1)
mdl.fit(X, y)
print("300-tree quantile fit secs:", round(time.time()-t0,1))
t0=time.time()
p = mdl.predict(X[:5000])
print("predict secs:", round(time.time()-t0,2))
