import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@ytr)
    return Zv@w, w
for cols in [['spend_28'],['spend_28','spend_84'],['spend_28','spend_84','spend_364']]:
    Xtr = ptr[cols].astype(float).values; Xpv = pv[cols].astype(float).values
    p, w = ridge_fit(Xtr, ptr[TARGET].values, Xpv, 1.0)
    print(cols, 'pv MAE', round(np.abs(p-pv[TARGET]).values.mean(),2), 'pred mean', round(p.mean(),1), 'coef', np.round(w,3))
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
p_in, _ = ridge_fit(Xtr, ytr, Xtr, 1.0)
print('IN-SAMPLE full ridge MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean:', round(p_in.mean(),1), 'pred std:', round(p_in.std(),1))
Z=(Xtr-Xtr.mean(0))/(Xtr.std(0)+1e-9)
print('Z std min:', round(Z.std(0).min(),6), 'cond:', round(np.linalg.cond(Z.T@Z),1))
# per-snapshot means of key features
for c in ['spend_28','fwd28_mean','wk0']:
    print(c, 'ptr mean', round(ptr[c].mean(),1), 'pv mean', round(pv[c].mean(),1))