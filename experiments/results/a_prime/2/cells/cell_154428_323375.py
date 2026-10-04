
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
print('target by snapshot day (train):')
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()]).round(1).to_string())
# tenure / zero rate by snapshot
print('\nzero% and mean spend_28 by snapshot (all rows):')
print(t.groupby('snapshot_day')[['spend_28','is_zero_28','tenure_days']].mean().round(1).to_string())

# local ridge proxy
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[cols].copy(); Xva = va[cols].copy()
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
for c in cols:
    if str(Xtr[c].dtype) == 'category':
        Xtr[c] = Xtr[c].astype(float); Xva[c] = Xva[c].astype(float)
Xtr = Xtr.astype(np.float64); Xva = Xva.astype(np.float64)
med = np.nanmedian(Xtr.values, axis=0)
Xtr = np.where(np.isnan(Xtr.values), med, Xtr.values)
Xva = np.where(np.isnan(Xva.values), med, Xva.values)
mu = Xtr.mean(0); sg = Xtr.std(0)+1e-9
Xtr_s = (Xtr-mu)/sg; Xva_s = (Xva-mu)/sg

def ridge_fit(X, y, lam):
    A = X.T@X + lam*np.eye(X.shape[1])
    return np.linalg.solve(A, X.T@y)

# pick alpha on inner split: train snapshots <=403 fit, 431 as inner val
inner_va = tr.snapshot_day==431
Xa, ya = Xtr_s[~inner_va.values], ytr[~inner_va.values]
Xb, yb = Xtr_s[inner_va.values], ytr[inner_va.values]
for lam in [1e-2,1e-1,1,10,100,1000]:
    w = ridge_fit(Xa, ya, lam)
    print('lam', lam, 'inner MAE', round(np.abs(Xb@w - yb).mean(),2))
