
import numpy as np, pandas as pd, xgboost as xgb
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
X = m[feat_cols].astype(float); y = m[agent_api.TARGET].values
tr = (m.snapshot_day <= 403).values; va = (m.snapshot_day == 431).values
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

def fit_pred(objective, logt=False, n=1500, lr=0.03, md=7, mcw=10, subs=0.8, col=0.8, qalpha=None):
    yt = np.log1p(ytr) if logt else ytr
    model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                             subsample=subs, colsample_bytree=col, objective=objective,
                             tree_method='hist', n_jobs=8, random_state=0)
    if qalpha is not None: model.set_params(quantile_alpha=qalpha)
    model.fit(Xtr, yt)
    p = model.predict(Xva)
    if logt: p = np.expm1(p)
    return np.clip(p, 0, None)

res={}
for name, kw in [('abs md7', dict(objective='reg:absoluteerror')),
                 ('abs md9 lr02 n2500', dict(objective='reg:absoluteerror', lr=0.02, n=2500, md=9)),
                 ('abs md5 lr03', dict(objective='reg:absoluteerror', md=5)),
                 ('quantile .5', dict(objective='reg:quantileerror', qalpha=0.5)),
                 ('quantile .5 deep', dict(objective='reg:quantileerror', qalpha=0.5, lr=0.02, n=2500, md=9)),
                 ('log+abs', dict(objective='reg:absoluteerror', logt=True)),
                 ('log+quantile', dict(objective='reg:quantileerror', qalpha=0.5, logt=True))]:
    try:
        p = fit_pred(**kw)
        res[name]=p
        print(f'{name:20s} MAE={np.abs(p-yva).mean():.3f}')
    except Exception as e:
        print(name,'ERR',type(e).__name__,str(e)[:100])

# blends
ps = list(res.items())
best=(None,1e9)
import itertools
for r in range(1,len(ps)+1):
    for combo in itertools.combinations(ps,r):
        pm = np.mean([p for _,p in combo],axis=0)
        v = np.abs(pm-yva).mean()
        if v<best[1]: best=([n for n,_ in combo],round(v,3))
print('best blend:', best)
