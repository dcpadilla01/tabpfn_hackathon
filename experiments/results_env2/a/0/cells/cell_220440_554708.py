import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f = A.load_saved("feats_v3.parquet")
print(sorted(f.snapshot_day.unique()))
t = A.train_targets()
df = f.merge(t, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day','index')]
va_days = [459,487,515,543]

Xtr, ytr = df[feat_cols], df.future_spend_4w.values
Xva = f[f.snapshot_day.isin(va_days)]
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, objective='reg:quantileerror',
    quantile_alpha=0.5, tree_method='hist', n_jobs=4)
m.fit(Xtr, ytr, verbose=False)
pred = np.clip(m.predict(Xva[feat_cols]), 0, None)
out = Xva[['household_key','snapshot_day']].copy()
out['prediction'] = pred
print(out.shape, out.snapshot_day.value_counts().to_dict())
p = A.save_table(out, "pred_e005.parquet")
print(p)
