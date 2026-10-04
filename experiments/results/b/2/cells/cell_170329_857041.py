import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
sd = snapshot_days(); trd = sd['train']
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
feat = [c for c in F.columns if c not in key+['gbm_pred']]
D = F.merge(tt, on=key)
print('n feat', len(feat))
# simple baselines on pseudo-val (snap 431)
pv = D[D.snapshot_day==431]; ptr = D[D.snapshot_day<431]
print('y mean/med pv:', pv[TARGET].mean().round(1), pv[TARGET].median().round(1), '| train:', ptr[TARGET].mean().round(1), ptr[TARGET].median().round(1))
for c in ['spend_28','ewma28_4w','fwd28_mean','pred_2p' ]:
    if c in D.columns:
        print(f'baseline {c}: pv MAE', round((pv[c]-pv[TARGET]).abs().mean(),2), '| scaled .9:', round((0.9*pv[c]-pv[TARGET]).abs().mean(),2))
print('baseline 0:', round(pv[TARGET].abs().mean(),2), 'median:', round((ptr[TARGET].median()-pv[TARGET]).abs().mean(),2))

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sg; Zv=(Xva-mu)/sg
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); b = Z.T@ytr
    w = np.linalg.solve(A,b)
    return Zv@w, w, mu, sg
Xtr = ptr[feat].astype(float).fillna(ptr[feat].astype(float).median()).values
ytr = ptr[TARGET].values; Xpv = pv[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; ypv = pv[TARGET].values
# pick alpha on inner split (train<=375, val 403)
pi = D[D.snapshot_day==403]; pt = D[D.snapshot_day<=375]
Xt2 = pt[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; yt2=pt[TARGET].values
Xi2 = pi[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; yi2=pi[TARGET].values
best=None
for a in [1,3,10,30,100,300,1000]:
    p,_,_,_ = ridge_fit(Xt2, yt2, Xi2, a)
    m = np.abs(p-yi2).mean()
    if best is None or m<best[1]: best=(a,m)
print('inner best alpha, MAE@403:', best)
p, w, mu, sg = ridge_fit(Xtr, ytr, Xpv, best[0])
print('pseudo-val 431 ridge MAE:', round(np.abs(p-ypv).mean(),2), 'bias:', round((p-ypv).mean(),2))
# per-snapshot residual bias on train fit
for s in [95,207,319,403]:
    m_ = D[D.snapshot_day==s]; Xs = m_[feat].astype(float).fillna(ptr[feat].astype(float).median()).values
    ps = (Xs-mu)/sg@w
    print(f'  snap {s}: bias {np.round((ps-m_[TARGET]).mean(),2)} MAE {np.round(np.abs(ps-m_[TARGET]).mean(),2)}')
# gbm_pred as feature?
if 'gbm_pred' in F.columns:
    g = F[key+['gbm_pred']].merge(tt, on=key)
    print('corr(gbm_pred,y) train:', round(g.gbm_pred.corr(g[TARGET]),3))
    Xtr2 = np.hstack([Xtr, ptr[['gbm_pred']].astype(float).values]); Xpv2 = np.hstack([Xpv, pv[['gbm_pred']].astype(float).values])
    p2,_,_,_ = ridge_fit(Xtr2, ytr, Xpv2, best[0])
    print('pseudo-val MAE with gbm_pred:', round(np.abs(p2-ypv).mean(),2))