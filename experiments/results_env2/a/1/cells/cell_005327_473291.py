import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = M[base_cols].values.astype(np.float32)
y = M[ycol].values.astype(float); d = M.snapshot_day.values
t0=time.time()
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, objective='reg:quantileerror',
    quantile_alpha=0.5, tree_method='hist', subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=-1)
mask = d < 431
w = 0.5**((431 - d[mask])/140.0)
m.fit(X[mask], y[mask], sample_weight=w)
print('fit 1200 rounds, 24k rows:', round(time.time()-t0,1),'s')
p = m.predict(X[~mask]); print('MAE431', round(np.abs(p-y[~mask]).mean(),3))
