import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(te_day, obj='reg:quantileerror', alpha=0.5, rounds=2400, lr=0.03, depth=6, decay=140, seed=1, logt=False):
    train_max = te_day-28
    tr = (days>=151)&(days<=train_max)&y.notna()
    w = 0.5**((train_max-days[tr].values)/decay)
    yy = np.log1p(y[tr]) if logt else y[tr]
    m = xgb.XGBRegressor(n_estimators=rounds, objective=obj, quantile_alpha=alpha, tree_method='hist',
        max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1, seed=seed)
    m.fit(X[tr], yy, sample_weight=w)
    return m
def evaluate(te_day, m, logt=False):
    te=(days==te_day); yt=y[te].values; ft=allF[te]
    p=m.predict(X[te])
    if logt: p=np.expm1(p)
    base=0.31*ft['spend_84'].values
    out={'model':np.abs(p-yt).mean()}
    for a in [0.5,0.55,0.6]:
        b=a*p+(1-a)*base
        out[f'blend{a}']=np.abs(b-yt).mean()
        bc=np.where(b<10,0,b)
        out[f'blend{a}clip']=np.abs(bc-yt).mean()
    return out
t0=time.time()
# squared-error model at 431
msq = fit(431, obj='reg:squarederror')
print("431 sq:", {k:round(v,3) for k,v in evaluate(431, msq).items()}, flush=True)
# log-target squared model at 431
mlog = fit(431, obj='reg:squarederror', logt=True)
print("431 log:", {k:round(v,3) for k,v in evaluate(431, mlog, logt=True).items()}, flush=True)
# quantile alpha 0.45 at 431
mq45 = fit(431, alpha=0.45)
print("431 q45:", {k:round(v,3) for k,v in evaluate(431, mq45).items()}, flush=True)
print("%.0fs" % (time.time()-t0))
