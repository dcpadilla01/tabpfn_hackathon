
import numpy as np, pandas as pd, xgboost as xgb
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
Xtr = m[feat_cols].astype(float).values; ytr = m[agent_api.TARGET].values
va = f3[f3.snapshot_day.isin(agent_api.snapshot_days()['validation'])].copy()
print('train rows', Xtr.shape, 'val rows', va.shape)

def fit(objective, logt=False, n=2500, lr=0.02, md=9, mcw=10, subs=0.8, col=0.8, seed=0):
    yt = np.log1p(ytr) if logt else ytr
    model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                             subsample=subs, colsample_bytree=col, objective=objective,
                             tree_method='hist', n_jobs=8, random_state=seed)
    if objective=='reg:quantileerror': model.set_params(quantile_alpha=0.5)
    model.fit(Xtr, yt)
    p = model.predict(va[feat_cols].astype(float).values)
    if logt: p = np.expm1(p)
    return np.clip(p, 0, None)

P = [fit('reg:quantileerror', seed=0), fit('reg:quantileerror', seed=1),
     fit('reg:absoluteerror', logt=True), fit('reg:quantileerror', logt=True)]
pred = np.median(P, axis=0)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = pred
print('pred stats', np.percentile(pred,[10,50,90]).round(2), pred.mean().round(2))
path = agent_api.save_table(out, 'pred_e005.parquet')
print(path)
