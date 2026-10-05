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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    g = tx.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # aligned 28d blocks b1..b5: [s-27..s], [s-55..s-28], ...
    for k in range(1, 6):
        hi = snapshot_day - 28*(k-1); lo = hi - 27
        out['b%d' % k] = g.apply(lambda t, lo=lo, hi=hi: t.loc[(t.day>=lo)&(t.day<=hi),'sales_value'].sum()) 
    w = tx.copy(); w['wk'] = (w.day + 8)//7
    cur_wk = (snapshot_day + 8)//7
    gw = w.groupby(['household_key','wk'])['sales_value'].sum().reset_index()
    for nwk, tag in [(12,'12'), (26,'26')]:
        sub = gw[(gw.wk <= cur_wk) & (gw.wk > cur_wk - nwk)]
        full = pd.DataFrame(index=hh); full['s'] = 0.0
        piv = sub.groupby('household_key')['sales_value'].sum()
        # include zero weeks: reindex over all weeks in window
        def med(t, nwk=nwk):
            weeks = np.arange(cur_wk-nwk+1, cur_wk+1)
            s = t.set_index('wk')['sales_value'].reindex(weeks).fillna(0.0).values
            return np.median(s)
        out['med_week_%s' % tag] = hh.map(sub.groupby('household_key').apply(med)).fillna(0.0)
    # median basket value last 84d
    sub84 = tx[tx.day > snapshot_day-84]
    out['med_basket_84'] = hh.map(sub84.groupby(['household_key','basket_id'])['sales_value'].sum().groupby('household_key').median())
    # robust block combos
    b = out[['b1','b2','b3','b4']].values
    out['blk_med3'] = np.median(b[:, :3], axis=1)
    out['blk_med4'] = np.median(b, axis=1)
    out['blk_min12'] = np.minimum(b[:,0], b[:,1])
    out['blk_min123'] = np.min(b[:, :3], axis=1)
    out['blk_trim4'] = (np.sort(b, axis=1)[:,1] + np.sort(b, axis=1)[:,2]) / 2.0
    out['blk_wavg'] = 0.4*b[:,0] + 0.3*b[:,1] + 0.2*b[:,2] + 0.1*b[:,3]
    out['blk_iqr4'] = np.sort(b, axis=1)[:,2] - np.sort(b, axis=1)[:,1]
    out['blk_zero_share4'] = (b == 0).mean(axis=1)
    out = out.drop(columns=['b1','b2','b3','b4'])
    return out

cand = agent_api.build_features(fn)
print('cand', cand.shape)
agent_api.save_table(cand, 'cand_robust')
print(cand.head())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
cand = agent_api.load_saved('cand_robust')
tt = agent_api.train_targets()
m = df.merge(cand, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr = day < 459

# correlations of cand cols with target and with dec_28
for c in cand_cols:
    v = m[c].values.astype(float)
    ok = tr & np.isfinite(v)
    r_y = np.corrcoef(v[ok], y[ok])[0,1]
    r_d = np.corrcoef(v[ok], m['dec_28'].values[ok])[0,1]
    print('%-16s corr_y %+.3f  corr_dec28 %+.3f  nan%% %.2f' % (c, r_y, r_d, 100*m[c].isna().mean()))

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
        w = np.linalg.solve(A, Z.T@ytr)
        pred[te] = Zt@w
    return pred

X0 = prep(feats); X1 = prep(feats + cand_cols)
p0, p1 = cv(X0), cv(X1)
print('\nridge CV-MAE  E013-only: %.3f   +cand: %.3f' % (np.mean(np.abs(p0[tr]-y[tr])), np.mean(np.abs(p1[tr]-y[tr]))))
# per-snapshot delta
for s in [s for s in np.sort(np.unique(day)) if s < 459]:
    te = day==s
    print('snap %3d  base %.2f  +cand %.2f' % (s, np.mean(np.abs(p0[te]-y[te])), np.mean(np.abs(p1[te]-y[te]))))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
cand = agent_api.load_saved('cand_robust.parquet')
tt = agent_api.train_targets()
m = df.merge(cand, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr = day < 459

for c in cand_cols:
    v = m[c].values.astype(float)
    ok = tr & np.isfinite(v)
    r_y = np.corrcoef(v[ok], y[ok])[0,1]
    r_d = np.corrcoef(v[ok], m['dec_28'].values[ok])[0,1]
    print('%-16s corr_y %+.3f  corr_dec28 %+.3f  nan%% %.2f' % (c, r_y, r_d, 100*m[c].isna().mean()))

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
        w = np.linalg.solve(A, Z.T@ytr)
        pred[te] = Zt@w
    return pred

X0 = prep(feats); X1 = prep(feats + cand_cols)
p0, p1 = cv(X0), cv(X1)
print('\nridge CV-MAE  E013-only: %.3f   +cand: %.3f' % (np.mean(np.abs(p0[tr]-y[tr])), np.mean(np.abs(p1[tr]-y[tr]))))
for s in [s for s in np.sort(np.unique(day)) if s < 459]:
    te = day==s
    print('snap %3d  base %.2f  +cand %.2f' % (s, np.mean(np.abs(p0[te]-y[te])), np.mean(np.abs(p1[te]-y[te]))))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

df = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
for c in [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]:
    d = pd.get_dummies(X[c].astype('object').where(X[c].notna(), 'NA'), prefix=c)
    X = pd.concat([X.drop(columns=[c]), d.astype(float)], axis=1)
Xv = X.astype(float).values
y = m['future_spend_4w'].values.astype(float)
day = m['snapshot_day'].values
tr = day < 459
tr_snaps = [s for s in np.sort(np.unique(day)) if s < 459]

def cv(lam=10):
    pred = np.full(len(y), np.nan)
    for s in tr_snaps:
        te = day == s
        Xtr, ytr, Xte = Xv[tr & ~te], y[tr & ~te], Xv[te]
        mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0); sd = np.where(~np.isfinite(sd)|(sd==0),1.0,sd)
        Z = np.where(np.isfinite(Xtr),(Xtr-mu)/sd,0.0); Zt = np.where(np.isfinite(Xte),(Xte-mu)/sd,0.0)
        Z = np.hstack([Z,np.ones((len(Z),1))]); Zt = np.hstack([Zt,np.ones((len(Zt),1))])
        A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1] -= lam
        pred[te] = Zt@np.linalg.solve(A, Z.T@ytr)
    return pred

p = cv()
r = y - p
print('MAE %.3f  resid mean %.2f  resid median %.2f  y=0 share %.3f  y<10 share %.3f' % (
    np.mean(np.abs(r[tr])), np.mean(r[tr]), np.median(r[tr]), (y[tr]==0).mean(), (y[tr]<10).mean()))
print('MAE from y=0 rows: %.3f (share of total %.2f)' % (
    np.abs(r[tr][y[tr]==0]).sum()/len(y[tr]), np.abs(r[tr][y[tr]==0]).sum()/np.abs(r[tr]).sum()))

q = np.quantile(p[tr], np.linspace(0,1,11))
bins = np.clip(np.digitize(p[tr], q[1:-1]), 0, 9)
base_mae = np.mean(np.abs(r[tr]))
gain = 0.0
print('\ndec |   n  | mean_y | med_y | mean_p | oracle_shift | mae_base | mae_oracle')
for b in range(10):
    sel = (bins==b)
    n = sel.sum()
    mb = np.mean(np.abs(r[tr][sel]))
    sh = np.median(r[tr][sel])          # oracle constant shift = median residual
    mo = np.mean(np.abs(r[tr][sel]-sh))
    gain += (mb-mo)*n
    print('%2d  | %5d | %6.1f | %5.1f | %6.1f | %12.1f | %8.2f | %8.2f' % (
        b, n, y[tr][sel].mean(), np.median(y[tr][sel]), p[tr][sel].mean(), sh, mb, mo))
print('\noracle per-decile shift gain: %.3f MAE points (base %.3f -> %.3f)' % (gain/len(y[tr]), base_mae, base_mae-gain/len(y[tr])))

# global scale/shift
for a in (0.9,0.95,1.0,1.05,1.1):
    print('scale %.2f MAE %.3f' % (a, np.mean(np.abs(y[tr]-a*p[tr]))))

# does a "recent spike" flag identify rows where model overpredicts? (median resid by spike)
sp = (m['dec_28'].values / np.maximum(m['spend_84'].values/3.0, 1e-9))
ok = tr & np.isfinite(sp) & (m['spend_84'].values>0)
qs = np.quantile(sp[ok], [0.25,0.5,0.75,0.9])
b2 = np.digitize(sp[ok], qs)
for b in range(5):
    sel = b2==b
    print('spike quint %d: n %5d  med resid %7.1f  mean resid %7.1f' % (b, sel.sum(), np.median(r[tr][ok][sel]), np.mean(r[tr][ok][sel])))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    out = pd.DataFrame(index=hh)
    s = snapshot_day
    # weekly spend series, last 26 weeks
    w = tx[(tx.day > s - 26*7) & (tx.day <= s)].copy()
    w['wk'] = (w.day + 8)//7
    cur = (s + 8)//7
    weeks = np.arange(cur-25, cur+1)
    gws = w.groupby(['household_key','wk'])['sales_value'].sum()
    def wk_stats(t):
        v = t.reindex(pd.MultiIndex.from_product([t.index.get_level_values(0).unique(), weeks])).droplevel(0)
        v = v.groupby(level=0).sum() if False else v
        return v
    piv = gws.unstack('wk').reindex(columns=weeks, fill_value=0.0).reindex(hh).fillna(0.0)
    med = piv.median(axis=1).replace(0, np.nan)
    out['sk_wk_mean_over_med'] = (piv.mean(axis=1)/med).fillna(1.0).clip(0, 50)
    out['sk_wk_max_over_med'] = (piv.max(axis=1)/med).fillna(1.0).clip(0, 200)
    out['sk_wk_cv'] = (piv.std(axis=1)/piv.mean(axis=1).replace(0,np.nan)).fillna(0.0).clip(0, 10)
    out['sk_wk_zero_share'] = (piv == 0).mean(axis=1)
    # basket concentration, last 84d
    b = tx[(tx.day > s-84) & (tx.day <= s)]
    gb = b.groupby(['household_key','basket_id'])['sales_value'].sum()
    tot = gb.groupby('household_key').sum().reindex(hh).fillna(0.0)
    mx = gb.groupby('household_key').max().reindex(hh).fillna(0.0)
    medb = gb.groupby('household_key').median().reindex(hh)
    out['sk_basket_top_share'] = np.where(tot>0, mx/tot.replace(0,np.nan), 0.0).clip(0,1)
    out['sk_basket_mean_over_med'] = (gb.groupby('household_key').mean().reindex(hh)/medb).fillna(1.0).clip(0,50)
    out['sk_basket_n_per_wk'] = (gb.groupby('household_key').count().reindex(hh).fillna(0.0)/12.0)
    # aligned 4-week blocks, last 8 blocks: max/med, zero share
    blocks = np.zeros((len(hh), 8)); hh_idx = {h:i for i,h in enumerate(hh)}
    for k in range(8):
        hi = s - 28*k; lo = hi - 27
        sub = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        for h, v in sub.items():
            blocks[hh_idx[h], k] = v
    bm = pd.DataFrame(blocks, index=hh)
    medb8 = bm.median(axis=1).replace(0, np.nan)
    out['sk_blk_max_over_med'] = (bm.max(axis=1)/medb8).fillna(1.0).clip(0,200)
    out['sk_blk_zero_share'] = (bm==0).mean(axis=1)
    out['sk_blk_med'] = bm.median(axis=1)          # historical median of 4w-block spend
    out['sk_blk_mean'] = bm.mean(axis=1)
    out['sk_p_zero_next'] = (bm==0).mean(axis=1)   # same as zero share (historical P(block=0))
    # interaction candidates: median level x zero risk
    out['sk_med_x_zerorisk'] = out['sk_blk_med'] * out['sk_blk_zero_share']
    out['sk_med_x_wkzeros'] = out['sk_blk_med'] * out['sk_wk_zero_share']
    return out

sk = agent_api.build_features(fn)
print('skew feats', sk.shape)
agent_api.save_table(sk, 'cand_skew')
print(sk.describe().T[['mean','50%','max']].round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    out = pd.DataFrame(index=hh)
    s = snapshot_day
    w = tx[(tx.day > s - 26*7) & (tx.day <= s)].copy()
    w['wk'] = (w.day + 8)//7
    cur = (s + 8)//7
    weeks = np.arange(cur-25, cur+1)
    piv = w.groupby(['household_key','wk'])['sales_value'].sum().unstack('wk').reindex(columns=weeks, fill_value=0.0).reindex(hh).fillna(0.0)
    med = piv.median(axis=1).replace(0, np.nan)
    out['sk_wk_mean_over_med'] = (piv.mean(axis=1)/med).fillna(1.0).clip(0, 50)
    out['sk_wk_max_over_med'] = (piv.max(axis=1)/med).fillna(1.0).clip(0, 200)
    out['sk_wk_cv'] = (piv.std(axis=1)/piv.mean(axis=1).replace(0,np.nan)).fillna(0.0).clip(0, 10)
    out['sk_wk_zero_share'] = (piv == 0).mean(axis=1)
    b = tx[(tx.day > s-84) & (tx.day <= s)]
    gb = b.groupby(['household_key','basket_id'])['sales_value'].sum()
    tot = gb.groupby('household_key').sum().reindex(hh).fillna(0.0)
    mx = gb.groupby('household_key').max().reindex(hh).fillna(0.0)
    medb = gb.groupby('household_key').median().reindex(hh)
    out['sk_basket_top_share'] = np.where(tot>0, mx/tot.replace(0,np.nan), 0.0)
    out['sk_basket_mean_over_med'] = (gb.groupby('household_key').mean().reindex(hh)/medb).fillna(1.0).clip(0,50)
    out['sk_basket_n_per_wk'] = (gb.groupby('household_key').count().reindex(hh).fillna(0.0)/12.0)
    blocks = np.zeros((len(hh), 8))
    for k in range(8):
        hi = s - 28*k; lo = hi - 27
        sub = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        blocks[:, k] = sub.reindex(hh).fillna(0.0).values
    bm = pd.DataFrame(blocks, index=hh)
    medb8 = bm.median(axis=1).replace(0, np.nan)
    out['sk_blk_max_over_med'] = (bm.max(axis=1)/medb8).fillna(1.0).clip(0,200)
    out['sk_blk_zero_share'] = (bm==0).mean(axis=1)
    out['sk_blk_med'] = bm.median(axis=1)
    out['sk_blk_mean'] = bm.mean(axis=1)
    out['sk_med_x_zerorisk'] = out['sk_blk_med'] * out['sk_blk_zero_share']
    out['sk_med_x_wkzeros'] = out['sk_blk_med'] * out['sk_wk_zero_share']
    return out

sk = agent_api.build_features(fn)
print('skew feats', sk.shape)
agent_api.save_table(sk, 'cand_skew')
print(sk.describe().T[['mean','50%','max']].round(3))


# ---- cell ----
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
