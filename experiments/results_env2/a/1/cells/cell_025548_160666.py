import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
def fit(train_max, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, seed=1):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m
res = {}
t0=time.time()
for s in [319, 347, 375, 403]:
    te = (days==s); yt = y[te].values; Xt = X[te]; ft = allF[te]
    m = fit(s-28)
    p = m.predict(Xt)
    base = 0.31*ft['spend_84'].values
    seqm = ft['seq_mean'].values
    mae_p = np.abs(p-yt).mean()
    maes = {f'blend{a}': np.abs(a*p+(1-a)*base-yt).mean() for a in [0.5,0.6,0.7]}
    maes['seqonly'] = np.abs(seqm-yt).mean()
    maes['med3'] = np.median(np.vstack([p,base,seqm]),axis=0).astype(float)
    maes['med3'] = np.abs(maes['med3']-yt).mean()
    maes['mean3'] = np.abs((p+base+seqm)/3-yt).mean()
    print(s, "model %.3f" % mae_p, {k:round(v,3) for k,v in maes.items()}, flush=True)
print("total %.0fs" % (time.time()-t0))
