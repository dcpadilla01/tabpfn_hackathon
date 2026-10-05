
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')

FCOLS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]
hold  = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh   = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def xgb_fit(X, y, seed=0, obj='reg:squarederror', lr=0.02, n=2000, md=7, mcw=10, q=None):
    kw = dict(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
              subsample=0.8, colsample_bytree=0.8, objective=obj,
              random_state=seed, n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw)
    m.fit(X, y); return m

m1 = xgb_fit(Xtr, ytr, seed=1)
m2 = xgb_fit(Xtr, ytr, seed=2, obj='reg:quantileerror', q=0.5)
p1, p2 = m1.predict(Xh), m2.predict(Xh)
blend = 0.5*p1 + 0.5*p2
def mae(a,b): return np.mean(np.abs(np.asarray(a)-np.asarray(b)))
print("holdout(431) MAE sq:", round(mae(p1,yh),3), " med:", round(mae(p2,yh),3), " blend:", round(mae(blend,yh),3))
print("mean y:", round(yh.mean(),2), "mean blend:", round(blend.mean(),2),
      "median y:", round(np.median(yh),2), "median blend:", round(np.median(blend),2))
print("corr sq/med preds:", round(np.corrcoef(p1,p2)[0,1],4))
# bias structure
for nm,p in [('sq',p1),('med',p2),('blend',blend)]:
    print(nm, "MAE y==0 rows:", round(mae(p[yh==0], 0),2), " MAE y>0 rows:", round(mae(p[yh>0], yh[yh>0]),2))
