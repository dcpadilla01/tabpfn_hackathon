import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
print('df', df.shape)
tt = agent_api.train_targets()
print('tt', tt.shape)
m = df.merge(tt, on=['household_key','snapshot_day'], how='left')
snaps = np.sort(m['snapshot_day'].unique())
print('snaps', snaps)
print('train rows', (m.snapshot_day<459).sum(), 'val rows', (m.snapshot_day>=459).sum(),
      'target NaN', m.future_spend_4w.isna().sum())

feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
obj_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
print('non-numeric feats:', obj_cols)
for c in obj_cols:
    card = X[c].nunique(dropna=True)
    if card <= 12:
        d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), 'NA'), prefix=c)
        X = pd.concat([X.drop(columns=[c]), d.astype(float)], axis=1)
    else:
        X = X.drop(columns=[c])
X = X.astype(float)
print('X', X.shape, 'NaN frac %.4f' % X.isna().mean().mean())

y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr_snaps = [s for s in snaps if s < 459]
Xv = X.values
tr_mask = day < 459

def fit_pred(Xtr, ytr, Xte, lam):
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)
    sd = np.where(~np.isfinite(sd)|(sd==0), 1.0, sd)
    Z = np.where(np.isfinite(Xtr), (Xtr-mu)/sd, 0.0)
    Zt = np.where(np.isfinite(Xte), (Xte-mu)/sd, 0.0)
    Z = np.hstack([Z, np.ones((len(Z),1))]); Zt = np.hstack([Zt, np.ones((len(Zt),1))])
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Z.T@ytr)
    return Zt@w

def cv_preds(lam, logt):
    yy = np.log1p(y) if logt else y
    pred = np.full(len(y), np.nan)
    for s in tr_snaps:
        te = day == s
        pred[te] = fit_pred(Xv[tr_mask & ~te], yy[tr_mask & ~te], Xv[te], lam)
    return pred

res = {}
for logt in (False, True):
    for lam in (1, 10, 100, 1000, 10000):
        p = cv_preds(lam, logt)
        pe = np.expm1(np.clip(p,0,20)) if logt else np.clip(p,0,None)
        mae = np.mean(np.abs(pe[tr_mask]-y[tr_mask]))
        res[(logt,lam)] = (mae, p)
        print('logt=%s lam=%6d  CV-MAE %.3f' % (logt, lam, mae))

# reference: single best trailing-spend column
cands = [c for c in feats if '28' in c][:10]
for c in cands:
    v = pd.to_numeric(m[c], errors='coerce').fillna(0).values
    print('base %-22s MAE %.3f' % (c, np.mean(np.abs(v[tr_mask]-y[tr_mask]))))

br = min([k for k in res if not k[0]], key=lambda k: res[k][0])
bl = min([k for k in res if k[0]], key=lambda k: res[k][0])
pr = res[br][1]; pl = np.expm1(np.clip(res[bl][1],0,20))
print('best raw', br, 'best log', bl)
for w in (0.0,0.25,0.5,0.75,1.0):
    pb = w*pr + (1-w)*pl
    print('blend w_raw=%.2f CV-MAE %.3f' % (w, np.mean(np.abs(pb[tr_mask]-y[tr_mask]))))
pbest = 0.5*pr + 0.5*pl
for s in tr_snaps:
    te = day==s
    print('snap %d n=%d MAE %.2f mean_y %.1f' % (s, te.sum(), np.mean(np.abs(pbest[te]-y[te])), y[te].mean()))
