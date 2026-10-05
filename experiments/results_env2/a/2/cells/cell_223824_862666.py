
import numpy as np, pandas as pd, xgboost as xgb
print('xgb version', xgb.__version__)
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
X = m[feat_cols].astype(float)
y = m[agent_api.TARGET].values
tr = m.snapshot_day <= 403
va = m.snapshot_day == 431
Xtr, ytr, Xva, yva = X[tr.values], y[tr.values], X[va.values], y[va.values]
print('train rows', Xtr.shape, 'val rows', Xva.shape)

def fit_pred(objective, logt=False, n=1500, lr=0.03, md=7, mcw=10, subs=0.8, col=0.8):
    yt = np.log1p(ytr) if logt else ytr
    model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                             subsample=subs, colsample_bytree=col, objective=objective,
                             tree_method='hist', n_jobs=8, random_state=0)
    model.fit(Xtr, yt)
    p = model.predict(Xva)
    if logt: p = np.expm1(p)
    return np.clip(p, 0, None)

for name, kw in [('squared', dict(objective='reg:squarederror')),
                 ('abserror', dict(objective='reg:absoluteerror')),
                 ('quantile0.5', dict(objective='reg:quantileerror', n=1500)),
                 ('log1p+squared', dict(objective='reg:squarederror', logt=True))]:
    try:
        p = fit_pred(**kw)
        print(f'{name:14s} MAE={np.abs(p-yva).mean():.3f}  R2={1-((p-yva)**2).sum()/((yva-yva.mean())**2).sum():.4f}')
    except Exception as e:
        print(name, 'ERR', type(e).__name__, str(e)[:120])
