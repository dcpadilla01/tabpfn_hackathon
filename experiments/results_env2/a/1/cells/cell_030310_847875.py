import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(train_max, obj='reg:quantileerror', alpha=0.5, rounds=2400, lr=0.03, depth=6, decay=140, seed=1):
    tr = (days>=151)&(days<=train_max)&y.notna()
    w = 0.5**((train_max-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective=obj, quantile_alpha=alpha, tree_method='hist',
        max_depth=6, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m
def evaluate(te_day, m, tag=''):
    te=(days==te_day); yt=y[te].values; ft=allF[te]
    p=m.predict(X[te]); base=0.31*ft['spend_84'].values
    out={'model':np.abs(p-yt).mean()}
    for a in [0.5,0.55,0.6]:
        b=a*p+(1-a)*base
        out[f'b{a}']=np.abs(b-yt).mean()
        out[f'b{a}c']=np.abs(np.where(b<10,0,b)-yt).mean()
    print(te_day, tag, {k:round(v,2) for k,v in out.items()}, flush=True)
    return out
t0=time.time()
for s in [347, 375, 403]:
    evaluate(s, fit(s-28), 'q50')
print("%.0fs" % (time.time()-t0))
