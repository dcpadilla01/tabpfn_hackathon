
import pandas as pd, numpy as np

e5 = load_saved('e005_longrun.parquet')
e6 = load_saved('e006_fwd_profile.parquet')
e7 = load_saved('e007_log.parquet')
tt = train_targets()

# check whether e007 logged the spend features
print('e5 spend_28:', e5['spend_28'].describe()[['mean','50%','max']].to_dict())
print('e7 spend_28:', e7['spend_28'].describe()[['mean','50%','max']].to_dict())
print('e7 x_s28:', e7['x_s28'].describe()[['mean','50%','max']].to_dict())
print('e7 wk0:', e7['wk0'].describe()[['mean','50%','max']].to_dict())
print('e7 recency:', e7['recency'].describe()[['mean','50%','max']].to_dict(), 'NaN:', e7['recency'].isna().mean())
print('e7 x_recency:', e7['x_recency'].describe()[['mean','50%','max']].to_dict())

m = tt.merge(e5, on=['household_key','snapshot_day'], how='left').merge(
    e6[['household_key','snapshot_day','fwd28_k1','fwd28_k2','fwd28_k3','fwd28_k4','fwd28_k5','fwd28_k6']],
    on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

def mae(p): return np.mean(np.abs(np.asarray(p)-y))

blocks = m[['fwd28_k1','fwd28_k2','fwd28_k3','fwd28_k4','fwd28_k5','fwd28_k6']].values
cands = {
 'zero': np.zeros(len(m)),
 'spend_28': m['spend_28'],
 'spend_56': m['spend_56'],
 'spend_84*4/3': m['spend_84']*4/3,
 'spend_112/2': m['spend_112']/2,
 'mean3blocks': blocks[:,:3].mean(1),
 'mean6blocks': np.nanmean(blocks,1),
 'ewma halflife1blk': sum(w*b for w,b in zip([0.5,0.3,0.2],[blocks[:,0],blocks[:,1],blocks[:,2]])),
}
for k,v in cands.items():
    print(f'{k:22s} MAE={mae(v):7.2f}')

# best scalar scaling (through origin) for spend_28
s = m['spend_28'].values
a = (s*y).sum()/ (s*s).sum()
print('spend_28 scaled best MAE:', mae(a*s), 'a=',a)

# in-sample linear fit (OLS) with e005 raw features as proxy for fixed model
X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
Xf = X.fillna(X.median()).clip(-1e6,1e6)
Xm = np.c_[np.ones(len(Xf)), Xf.values]
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
print('OLS in-sample MAE (e005 raw, 85 feat):', mae(Xm@beta))
# with log1p transforms of heavy features
Xl = Xf.copy()
for c in Xl.columns:
    if Xl[c].skew() > 2: Xl[c] = np.log1p(Xl[c].clip(lower=0))
Xm2 = np.c_[np.ones(len(Xl)), Xl.values]
beta2, *_ = np.linalg.lstsq(Xm2, y, rcond=None)
print('OLS in-sample MAE (log-skewed):', mae(Xm2@beta2))
print('y mean/med:', y.mean(), np.median(y), 'MAE of median:', mae(np.median(y)))
