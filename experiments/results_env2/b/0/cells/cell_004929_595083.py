import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
print('E013 columns (%d):' % (len(df.columns)-2))
print(list(df.columns))
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
obj_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in obj_cols:
    d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), 'NA'), prefix=c)
    X = pd.concat([X.drop(columns=[c]), d.astype(float)], axis=1)
X = X.astype(float)
nanrate = X.isna().mean()
print('\ntop NaN columns:'); print(nanrate.sort_values(ascending=False).head(8))
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr_snaps = [s for s in np.sort(np.unique(day)) if s < 459]
tr_mask = day < 459
Xv = X.values

# correlation of each feature with target (train rows only)
corr = {}
for c in X.columns:
    v = X[c].values
    ok = np.isfinite(v) & tr_mask
    if ok.sum() > 100 and np.std(v[ok]) > 0:
        corr[c] = np.corrcoef(v[ok], y[ok])[0,1]
cs = pd.Series(corr).sort_values(key=np.abs, ascending=False)
print('\ntop |corr| with target:'); print(cs.head(15).round(3))

def fit_pred(Xtr, ytr, Xte, lam, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)
    sd = np.where(~np.isfinite(sd)|(sd==0), 1.0, sd)
    Z = np.where(np.isfinite(Xtr), (Xtr-mu)/sd, 0.0)
    Zt = np.where(np.isfinite(Xte), (Xte-mu)/sd, 0.0)
    if cols is not None:
        mi_tr = (~np.isfinite(Xtr)).astype(float); mi_te = (~np.isfinite(Xte)).astype(float)
        Z = np.hstack([Z, mi_tr]); Zt = np.hstack([Zt, mi_te])
    Z = np.hstack([Z, np.ones((len(Z),1))]); Zt = np.hstack([Zt, np.ones((len(Zt),1))])
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Z.T@ytr)
    return Zt@w

def cv(base_cols=None, lam=10, add_inter=False, topk=12):
    pred = np.full(len(y), np.nan)
    Xw = Xv
    if add_inter:
        top = list(cs.index[:topk])
        idx = [list(X.columns).index(c) for c in top]
        inter = np.column_stack([Xv[:,i]*Xv[:,j] for ii,i in enumerate(idx) for j in idx[ii+1:]])
        Xw = np.hstack([Xv, np.where(np.isfinite(inter), inter, 0.0)])
    for s in tr_snaps:
        te = day == s
        pred[te] = fit_pred(Xw[tr_mask & ~te], y[tr_mask & ~te], Xw[te], lam, cols=base_cols)
    return pred

variants = {}
variants['plain_lam10'] = cv(None, 10)
variants['missind_lam10'] = cv('missind', 10)
variants['inter_lam10'] = cv(None, 10, add_inter=True)
variants['inter_lam100'] = cv(None, 100, add_inter=True)
for k,p in variants.items():
    print('%-16s CV-MAE %.3f' % (k, np.mean(np.abs(p[tr_mask]-y[tr_mask]))))

p = variants['plain_lam10']
print('\nplain OOF pred: corr(y)=%.3f  corr(dec_28)=%.3f  corr(spend_28)=%.3f  corr(spend_84)=%.3f' % (
    np.corrcoef(p[tr_mask], y[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['dec_28'].values[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['spend_28'].values[tr_mask])[0,1],
    np.corrcoef(p[tr_mask], X['spend_84'].values[tr_mask])[0,1]))
