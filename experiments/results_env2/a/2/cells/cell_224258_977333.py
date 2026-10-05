
import numpy as np, pandas as pd, xgboost as xgb, itertools
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
X = m[feat_cols].astype(float); y = m[agent_api.TARGET].values
tr = (m.snapshot_day <= 403).values; va = (m.snapshot_day == 431).values
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

def fit_pred(objective, logt=False, n=2500, lr=0.02, md=9, mcw=10, subs=0.8, col=0.8, seed=0):
    yt = np.log1p(ytr) if logt else ytr
    model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                             subsample=subs, colsample_bytree=col, objective=objective,
                             tree_method='hist', n_jobs=8, random_state=seed)
    if objective=='reg:quantileerror': model.set_params(quantile_alpha=0.5)
    model.fit(Xtr, yt)
    p = model.predict(Xva)
    if logt: p = np.expm1(p)
    return np.clip(p, 0, None)

preds={}
preds['q'] = fit_pred('reg:quantileerror')
preds['q2'] = fit_pred('reg:quantileerror', seed=1)
preds['logabs'] = fit_pred('reg:absoluteerror', logt=True)
preds['logq'] = fit_pred('reg:quantileerror', logt=True)
preds['sq'] = fit_pred('reg:squarederror')
for k,p in preds.items(): print(f'{k:8s} MAE={np.abs(p-yva).mean():.3f}')

# median-of-models vs mean
pm = np.mean([preds['q'],preds['q2'],preds['logabs'],preds['logq']],axis=0)
print('mean of 4 (no sq):', np.abs(pm-yva).mean())
pmed = np.median([preds['q'],preds['q2'],preds['logabs'],preds['logq']],axis=0)
print('median of 4:', np.abs(pmed-yva).mean())
pm5 = np.mean([preds['q'],preds['q2'],preds['logabs'],preds['logq'],preds['sq']],axis=0)
print('mean of 5 (with sq):', np.abs(pm5-yva).mean())
w = np.mean([preds['q'],preds['logq']],axis=0)
print('mean(q,logq):', np.abs(w-yva).mean())
