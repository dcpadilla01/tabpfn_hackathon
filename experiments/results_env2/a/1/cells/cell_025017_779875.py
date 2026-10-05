import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float)
y = allF['future_spend_4w'].astype(float)
days = allF['snapshot_day'].astype(int)
d431 = (days==431)
y431 = y[d431].values
X431 = X[d431]
f = allF[d431]
print("431 rows:", d431.sum(), "target mean %.2f median %.2f" % (np.nanmean(y431), np.nanmedian(y431)))
for c,s in [('spend_28',1.0),('spend_28',0.9),('spend_84',0.31),('seq_mean',1.0),('wk_mean_12',1.0)]:
    print("baseline %s*%.2f MAE@431: %.3f" % (c,s,np.abs(s*f[c].values-y431).mean()))
def fit_pred(train_max, snaps_min=151, rounds=2400, lr=0.03, depth=6, decay=140, alpha=0.5):
    tr = (days>=snaps_min)&(days<=train_max)&y.notna()
    w = 0.5**(((train_max)-days[tr].values)/decay)
    params = dict(objective='reg:quantileerror', quantile_alpha=alpha, tree_method='hist',
                  max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, nthread=-1)
    m = xgb.XGBRegressor(n_estimators=rounds, **params)
    t0=time.time()
    m.fit(X[tr], y[tr], sample_weight=w)
    p = m.predict(X431)
    mae = np.abs(p-y431).mean()
    print("snaps %d-%d rounds %d depth %d decay %d -> MAE@431 %.3f  predmean %.1f actmean %.1f  (%.0fs)" %
          (snaps_min, train_max, rounds, depth, decay, mae, p.mean(), np.nanmean(y431), time.time()-t0), flush=True)
    return p, mae
p0,m0 = fit_pred(403, 151)
p1,m1 = fit_pred(403, 207)
p2,m2 = fit_pred(403, 263)
for a in [0.3,0.5,0.7]:
    print("blend %.1f*model + %.1f*0.31*spend84: MAE@431 %.3f" % (a,1-a, np.abs(a*p0+(1-a)*0.31*f['spend_84'].values-y431).mean()))
