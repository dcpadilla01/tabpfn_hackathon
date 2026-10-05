
import numpy as np, pandas as pd
print(agent_api.snapshot_days())
s = agent_api.snapshot()
print('households type:', type(s.households), 'n=', len(s.households))
f3 = agent_api.load_saved('feats_v3.parquet')
print('f3 shape', f3.shape)
print('cols:', sorted(f3.columns.tolist()))
print(f3.groupby('snapshot_day').size().to_dict())
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
print('merged train rows', m.shape)
y = m[agent_api.TARGET]
print(y.describe())
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
corr = m[feat_cols].corrwith(y).reindex(m[feat_cols].corrwith(y).abs().sort_values(ascending=False).index)
print('top |corr| with target:'); print(corr.head(25))
sp28 = [c for c in feat_cols if 'spend' in c.lower() and '28' in c]
print('spend28 candidates:', sp28)
if sp28:
    p = m[sp28[0]].fillna(0).values
    print('naive trailing28 MAE (train):', np.abs(p-y).mean())
print('mean-only MAE (train):', np.abs(np.full(len(y), y.mean())-y).mean())
gm = y.mean(); best=(None,1e9)
for a in np.linspace(0,1.5,31):
    v = np.abs((a*p+(1-a)*gm)-y).mean()
    if v<best[1]: best=(round(a,2),round(v,3))
print('in-sample blend a*trailing28+(1-a)*mean:', best)


# ---- cell ----

import numpy as np, pandas as pd
f3 = agent_api.load_saved('feats_v3.parquet')
print('f3 shape', f3.shape)
print('cols:', sorted(f3.columns.tolist()))
print(f3.groupby('snapshot_day').size().to_dict())
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
print('merged train rows', m.shape)
y = m[agent_api.TARGET]
print(y.describe())
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
absc = m[feat_cols].corrwith(y).abs().sort_values(ascending=False)
print('top |corr| with target:'); print(absc.head(25))
sp28 = [c for c in feat_cols if 'spend' in c.lower() and '28' in c]
print('spend28 candidates:', sp28)
if sp28:
    p = m[sp28[0]].fillna(0).values
    print('naive trailing28 MAE (train):', np.abs(p-y).mean())
print('mean-only MAE (train):', np.abs(np.full(len(y), y.mean())-y).mean())
gm = y.mean(); best=(None,1e9)
for a in np.linspace(0,1.5,31):
    v = np.abs((a*p+(1-a)*gm)-y).mean()
    if v<best[1]: best=(round(a,2),round(v,3))
print('in-sample blend a*trailing28+(1-a)*mean:', best)


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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
