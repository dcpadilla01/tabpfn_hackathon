import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
sk = agent_api.load_saved('cand_skew.parquet')
tt = agent_api.train_targets()
m = df.merge(sk, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='left')
skc = [c for c in sk.columns if c not in ('household_key','snapshot_day')]
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr = day < 459
for c in skc:
    v = m[c].values.astype(float); ok = tr & np.isfinite(v)
    print('%-22s corr_y %+.3f corr_dec28 %+.3f' % (c, np.corrcoef(v[ok], y[ok])[0,1], np.corrcoef(v[ok], m['dec_28'].values[ok])[0,1]))

feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
def prep(cols):
    X = m[cols].copy()
    for c in [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]:
        d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), 'NA'), prefix=c)
        X = pd.concat([X.drop(columns=[c]), d.astype(float)], axis=1)
    return X.astype(float).values
def cv(Xv, lam=10):
    pred = np.full(len(y), np.nan)
    for s in [s for s in np.sort(np.unique(day)) if s < 459]:
        te = day == s
        Xtr, ytr, Xte = Xv[tr & ~te], y[tr & ~te], Xv[te]
        mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0); sd = np.where(~np.isfinite(sd)|(sd==0),1.0,sd)
        Z = np.where(np.isfinite(Xtr),(Xtr-mu)/sd,0.0); Zt = np.where(np.isfinite(Xte),(Xte-mu)/sd,0.0)
        Z = np.hstack([Z,np.ones((len(Z),1))]); Zt = np.hstack([Zt,np.ones((len(Zt),1))])
        A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1] -= lam
        pred[te] = Zt@np.linalg.solve(A, Z.T@ytr)
    return pred
X0, X1 = prep(feats), prep(feats+skc)
p0, p1 = cv(X0), cv(X1)
print('\nridge CV-MAE  E013: %.3f   +skew: %.3f' % (np.mean(np.abs(p0[tr]-y[tr])), np.mean(np.abs(p1[tr]-y[tr]))))
for s in [s for s in np.sort(np.unique(day)) if s < 459]:
    te = day==s
    print('snap %3d base %.2f +skew %.2f' % (s, np.mean(np.abs(p0[te]-y[te])), np.mean(np.abs(p1[te]-y[te]))))
