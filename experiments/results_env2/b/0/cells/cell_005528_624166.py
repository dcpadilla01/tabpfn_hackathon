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
