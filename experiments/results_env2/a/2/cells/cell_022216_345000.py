import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
f3 = A.load_saved("feats_v3.parquet"); fs = A.load_saved("feats_seasonal.parquet")
pt = A.load_saved("past_targets.parquet")
F = f3.merge(fs, on=["household_key","snapshot_day"], how="left").merge(pt, on=["household_key","snapshot_day"], how="left")
tt = A.train_targets()
data = F.merge(tt, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
feat_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
X = data[feat_cols].astype(float).values
y = data.future_spend_4w.values
day = data.snapshot_day.values
# train on all train rows, predict val
t0=time.time()
m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1200, learning_rate=0.03,
    max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, random_state=7)
m.fit(X, y)
val = F[F.snapshot_day>=459].sort_values(["snapshot_day","household_key"])
Xv = val[feat_cols].astype(float).values
val = val.assign(prediction=m.predict(Xv))
p = A.save_table(val[["household_key","snapshot_day","prediction"]], "pred_pt.parquet")
print("saved", p, f"{time.time()-t0:.0f}s")
