import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
d431 = (days==431); y431 = y[d431].values; X431 = X[d431]; f = allF[d431]
def fit_pred(train_max=403, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, alpha=0.5, seed=1, ss=0.8, cs=0.8, mchild=1):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    m = xgb.XGBRegressor(n_estimators=rounds, objective='reg:quantileerror', quantile_alpha=alpha,
        tree_method='hist', max_depth=depth, learning_rate=lr, subsample=ss, colsample_bytree=cs,
        min_child_weight=mchild, nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    return m.predict(X431)
t0=time.time()
pA = fit_pred(depth=8, rounds=2400)
print("depth8: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pA-y431).mean(), pA.mean(), time.time()-t0), flush=True)
t0=time.time()
pB = fit_pred(depth=6, rounds=2400, decay=90)
print("decay90: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pB-y431).mean(), pB.mean(), time.time()-t0), flush=True)
t0=time.time()
pC = fit_pred(depth=6, rounds=2400, decay=220)
print("decay220: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pC-y431).mean(), pC.mean(), time.time()-t0), flush=True)
t0=time.time()
pD = fit_pred(depth=10, rounds=1600, lr=0.03)
print("depth10/1600: MAE %.3f predmean %.1f (%.0fs)" % (np.abs(pD-y431).mean(), pD.mean(), time.time()-t0), flush=True)
base = 0.31*f['spend_84'].values
for nm,p in [('A_d8',pA),('B_dc90',pB),('C_dc220',pC),('D_d10',pD)]:
    for a in [0.5,0.6,0.7]:
        print("%s blend a=%.1f: %.3f" % (nm,a,np.abs(a*p+(1-a)*base-y431).mean()))
# blend of models
pm = (pA+pB+pC+pD)/4
for a in [0.5,0.6,0.7]:
    print("model-avg blend a=%.1f: %.3f" % (a,np.abs(a*pm+(1-a)*base-y431).mean()))
