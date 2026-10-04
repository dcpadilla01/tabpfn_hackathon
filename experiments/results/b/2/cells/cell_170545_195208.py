import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym = ytr.mean(); yc = ytr-ym
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@yc)
    return Zv@w + ym
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
Xpv = pv[feat].astype(float).fillna(med).values; ypv=pv[TARGET].values
p_in = ridge_fit(Xtr,ytr,Xtr,1.0)
print('in-sample MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean', round(p_in.mean(),1), 'std', round(p_in.std(),1))
pi = D[D.snapshot_day==403]; pt = D[D.snapshot_day<=375]
med2 = pt[feat].astype(float).median()
Xt2 = pt[feat].astype(float).fillna(med2).values; yt2=pt[TARGET].values
Xi2 = pi[feat].astype(float).fillna(med2).values; yi2=pi[TARGET].values
for a in [1,3,10,30,100]:
    p = ridge_fit(Xt2,yt2,Xi2,a); print('alpha',a,'MAE@403', round(np.abs(p-yi2).mean(),2))
p = ridge_fit(Xtr,ytr,Xpv,10.0)
print('PSEUDO-VAL 431 ridge MAE:', round(np.abs(p-ypv).mean(),2), 'bias:', round((p-ypv).mean(),2), 'pred std:', round(p.std(),1), 'y std:', round(ypv.std(),1))
for s in [95,207,319,403]:
    m_ = D[D.snapshot_day==s]; Xs = m_[feat].astype(float).fillna(med).values
    ps = ridge_fit(Xtr,ytr,Xs,10.0)
    print(f'  snap {s}: bias {np.round((ps-m_[TARGET]).mean(),2)} MAE {np.round(np.abs(ps-m_[TARGET]).mean(),2)}')
print('corr(pred,y) pv:', round(pd.Series(p).corr(pd.Series(ypv)),3))
Xtr2 = np.hstack([Xtr, ptr[['gbm_pred']].astype(float).values]); Xpv2 = np.hstack([Xpv, pv[['gbm_pred']].astype(float).values])
p2 = ridge_fit(Xtr2,ytr,Xpv2,10.0)
print('pv MAE + gbm_pred:', round(np.abs(p2-ypv).mean(),2))