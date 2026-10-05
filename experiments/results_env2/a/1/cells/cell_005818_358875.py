import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values

def fit_pred(Xtr, ytr, w, Xte, seed, obj='q', rounds=1200, lr=0.03, depth=6):
    kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, tree_method='hist',
              subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    if obj=='q':
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, **kw)
    else:
        m = xgb.XGBRegressor(objective='reg:absoluteerror', **kw)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xte)

def cv(transform, obj='q', eval_days=(403,431), seeds=(7,17), inv=None):
    out={}
    for ed in eval_days:
        trm = (d < ed) & ~np.isnan(y); tem = (d == ed) & ~np.isnan(y)
        w = 0.5**((ed - d[trm])/140.0)
        yt = transform(y[trm])
        ps = np.zeros(tem.sum())
        for s in seeds:
            p = fit_pred(X[trm], yt, w, X[tem], s, obj=obj)
            ps += (inv(p) if inv else p)/len(seeds)
        out[ed]=round(np.abs(ps - y[tem]).mean(),3)
    return out

t0=time.time()
idn = lambda v: v
print('raw-q   ', cv(idn), round(time.time()-t0,1))
print('log-q   ', cv(np.log1p, inv=np.expm1), round(time.time()-t0,1))
print('raw-ae  ', cv(idn, obj='ae'), round(time.time()-t0,1))
