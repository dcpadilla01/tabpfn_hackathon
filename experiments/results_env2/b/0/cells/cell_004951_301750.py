import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
for c in [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]:
    d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), 'NA'), prefix=c)
    X = pd.concat([X.drop(columns=[c]), d.astype(float)], axis=1)
X = X.astype(float)
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr_snaps = [s for s in np.sort(np.unique(day)) if s < 459]
tr_mask = day < 459
Xv = X.values
nan_mask = (~np.isfinite(Xv)).astype(float)

def fit_pred(Xtr, ytr, Xte, lam, missind=False):
    if missind:
        Xtr = np.hstack([Xtr, (~np.isfinite(Xtr)).astype(float)])
        Xte = np.hstack([Xte, (~np.isfinite(Xte)).astype(float)])
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)
    sd = np.where(~np.isfinite(sd)|(sd==0), 1.0, sd)
    Z = np.where(np.isfinite(Xtr), (Xtr-mu)/sd, 0.0)
    Zt = np.where(np.isfinite(Xte), (Xte-mu)/sd, 0.0)
    Z = np.hstack([Z, np.ones((len(Z),1))]); Zt = np.hstack([Zt, np.ones((len(Zt),1))])
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Z.T@ytr)
    return Zt@w

def cv(lam=10, missind=False, add_inter=False, topk=12, drop_top=0):
    Xw = Xv
    if add_inter:
        cs = pd.Series({c: np.corrcoef(X[c].values[tr_mask & np.isfinite(X[c].values)], y[tr_mask & np.isfinite(X[c].values)])[0,1]
                        for c in X.columns if np.isfinite(X[c].values).sum()>100})
        top = list(np.abs(cs).sort_values(ascending=False).index[:topk])
        idx = [list(X.columns).index(c) for c in top]
        inter = np.column_stack([Xv[:,i]*Xv[:,j] for ii,i in enumerate(idx) for j in idx[ii+1:]])
        Xw = np.hstack([Xv, np.where(np.isfinite(inter), inter, 0.0)])
    if drop_top:
        cs2 = pd.Series({c: np.corrcoef(X[c].values[tr_mask & np.isfinite(X[c].values)], y[tr_mask & np.isfinite(X[c].values)])[0,1]
                         for c in X.columns if np.isfinite(X[c].values).sum()>100})
        top2 = list(np.abs(cs2).sort_values(ascending=False).index[:drop_top])
        keep = [i for i,c in enumerate(X.columns) if c not in top2]
        Xw = Xw[:, keep]
    pred = np.full(len(y), np.nan)
    for s in tr_snaps:
        te = day == s
        pred[te] = fit_pred(Xw[tr_mask & ~te], y[tr_mask & ~te], Xw[te], lam, missind)
    return pred

variants = {}
variants['plain_lam10'] = cv(10)
variants['missind_lam10'] = cv(10, missind=True)
variants['inter_lam10'] = cv(10, add_inter=True)
variants['inter_lam100'] = cv(100, add_inter=True)
variants['notop5_lam10'] = cv(10, drop_top=5)
for k,p in variants.items():
    print('%-16s CV-MAE %.3f' % (k, np.mean(np.abs(p[tr_mask]-y[tr_mask]))))

p = variants['plain_lam10']
print('\nplain OOF: corr(y)=%.3f corr(dec_28)=%.3f corr(spend_84)=%.3f corr(z_rate84)=%.3f' % (
    np.corrcoef(p[tr_mask], y[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['dec_28'].values[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['spend_84'].values[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['z_rate84'].values[tr_mask])[0,1]))
# residual of the fixed model ~ unknown; check ridge pred vs best single feature complementarity
r = y - p
print('resid |mean| %.3f  std %.1f' % (np.abs(np.mean(r[tr_mask])), np.std(r[tr_mask])))
