import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and df[c].dtype.kind in 'ifb']
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
X = df[num_cols].astype(float)
bad = [c for c in num_cols if np.isinf(X[c].values).any()]
print('cols with inf:', bad)
for c in bad: print(c, df[c].describe())

Xv = X.values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days); iva = df.snapshot_day.isin(iva_days)
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr.values],0); sd = np.nanstd(Xv[itr.values],0); sd[sd==0]=1
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd)
Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr.values], Z[iva.values]
ytr, yva = y[itr.values], y[iva.values]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
print('mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
best=(None,1e9)
for lam in [30,100,300,1000,3000,10000]:
    w = ridge(Ztr, np.log1p(ytr), lam)
    f = lambda Z: np.expm1(np.clip(Z@w, -5, CAP))
    smear = np.mean(ytr/np.maximum(f(Ztr),1e-9))
    m = np.mean(np.abs(f(Zva)*smear-yva)); mt = np.mean(np.abs(f(Ztr)*smear-ytr))
    w2 = ridge(Ztr, ytr, lam)
    m2 = np.mean(np.abs(Zva@w2-yva))
    print('lam=%5d log: %.3f (tr %.3f) | raw: %.3f (tr %.3f)' % (lam, m, mt, m2, np.mean(np.abs(Ztr@w2-ytr))))
    best = min(best, [(lam,'log',m),(lam,'raw',m2)][np.argmin([m,m2])][::2]+[0]) if False else best
    if m<best[1]: best=(('log',lam),m)
    if m2<best[1]: best=(('raw',lam),m2)
print('best inner:', best)
