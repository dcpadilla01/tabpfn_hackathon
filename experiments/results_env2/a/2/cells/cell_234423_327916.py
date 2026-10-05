
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
tr = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
va = feats[~feats.set_index(['household_key','snapshot_day']).index.isin(
        tr.set_index(['household_key','snapshot_day']).index)].copy()
print('train rows', len(tr), 'val rows', len(va))

holdout = [347, 375, 403, 431]
inner = [d for d in sorted(tr.snapshot_day.unique()) if d not in holdout]
mi_tr = tr.snapshot_day.isin(inner).values
mi_ho = tr.snapshot_day.isin(holdout).values
Xa = tr.loc[mi_tr, feat_cols].astype(float).values
Xb = tr.loc[mi_ho, feat_cols].astype(float).values
ya = tr.loc[mi_tr,'future_spend_4w'].values
yb = tr.loc[mi_ho,'future_spend_4w'].values
def mae(p, y): return np.abs(np.clip(p,0,None)-y).mean()

cfg_med = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
               subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
               objective='reg:quantileerror', quantile_alpha=0.5)
cfg_sq  = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
               subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
               objective='reg:squarederror')
def histgb(seed):
    return HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=400,
        learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, random_state=seed)

t0=time.time()
# inner-holdout OOF for weight selection
o1 = xgb.XGBRegressor(**cfg_med).fit(Xa,ya).predict(Xb)
o2 = xgb.XGBRegressor(**cfg_sq).fit(Xa,ya).predict(Xb)
o3 = 0.5*(histgb(7).fit(Xa,ya).predict(Xb) + histgb(13).fit(Xa,ya).predict(Xb))
print('inner MAEs:', round(mae(o1,yb),3), round(mae(o2,yb),3), round(mae(o3,yb),3), round(time.time()-t0,1),'s')

best=(None,1e9)
for a in np.arange(0.15,0.71,0.05):
    for b in np.arange(0.15,0.71-a,0.05):
        c = 1-a-b
        if c < 0.149: continue
        e = mae(a*o1+b*o2+c*o3, yb)
        if e < best[1]: best=((round(a,2),round(b,2),round(c,2)), e)
w = best[0]; print('chosen weights (med,sq,hist):', w, 'inner MAE', round(best[1],3))

# refit on ALL train snapshots, predict validation
Xtr = tr[feat_cols].astype(float).values; ytr = tr['future_spend_4w'].values
Xva = va[feat_cols].astype(float).values
t0=time.time()
p1 = xgb.XGBRegressor(**cfg_med).fit(Xtr,ytr).predict(Xva)
p2 = xgb.XGBRegressor(**cfg_sq).fit(Xtr,ytr).predict(Xva)
p3 = 0.5*(histgb(7).fit(Xtr,ytr).predict(Xva) + histgb(13).fit(Xtr,ytr).predict(Xva))
pred = np.clip(w[0]*p1 + w[1]*p2 + w[2]*p3, 0, None)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = pred
print('pred stats: mean', round(pred.mean(),2), 'zero frac', round((pred==0).mean(),3), round(time.time()-t0,1),'s')
path = api.save_table(out, 'pred_e008')
print('saved', path, out.shape)
