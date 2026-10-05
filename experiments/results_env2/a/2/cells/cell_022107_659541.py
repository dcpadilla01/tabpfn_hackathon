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
oof_days=[347,375,403,431]
t0=time.time(); oof = np.full(len(data), np.nan)
for d in oof_days:
    tr = day < d; te = day == d
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1200, learning_rate=0.03,
        max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=7)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict(X[te])
    print(d, "done", f"{time.time()-t0:.0f}s", flush=True)
mae = np.nanmean(np.abs(oof-y))
print("med + past-target feats OOF MAE:", round(mae,3), "(harness med was 62.499)")
odf = data[["household_key","snapshot_day"]].copy(); odf["oof"]=oof
A.save_table(odf, "oof_pt.parquet")
