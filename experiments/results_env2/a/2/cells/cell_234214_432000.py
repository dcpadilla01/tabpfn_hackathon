
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
def mae(p): return round(np.abs(np.clip(p,0,None)-yho).mean(),3)

# tune HistGB quantile
best=(None,1e9)
t0=time.time()
for mi in [400, 800, 1200]:
    for lr in [0.03, 0.06]:
        for msl in [25, 60]:
            h = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=mi, learning_rate=lr,
                 max_leaf_nodes=31, min_samples_leaf=msl, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
            e = mae(h.predict(Xho))
            if e < best[1]: best=((mi,lr,msl),e)
            print(mi,lr,msl,'->',e, round(time.time()-t0,1),'s')
print('BEST histgb', best)

# xgb components (fixed configs from E004/E005)
t0=time.time()
mx = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
     subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
     objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_xgb = mx.predict(Xho); print('xgb med', mae(p_xgb), round(time.time()-t0,1),'s')
t0=time.time()
ms = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
     subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
     objective='reg:squarederror').fit(Xtr,ytr)
p_sq = ms.predict(Xho); print('xgb sq', mae(p_sq), round(time.time()-t0,1),'s')

mi,lr,msl = best[0]
h1 = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=mi, learning_rate=lr,
     max_leaf_nodes=31, min_samples_leaf=msl, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h = h1.predict(Xho); print('histgb best', mae(p_h))

# weight search over 3 components
rows=[]
for a in np.arange(0,1.001,0.1):
    for b in np.arange(0,1.001-a,0.1):
        c = 1-a-b
        if c < -1e-9: continue
        rows.append((round(a,2),round(b,2),round(c,2), round(mae(a*p_xgb+b*p_sq+c*p_h),3)))
rows.sort(key=lambda r:r[3])
print('top (xgbmed,xgbsq,histgb):')
for r in rows[:10]: print(r)
