import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days).values; iva = df.snapshot_day.isin(iva_days).values
y = df['future_spend_4w'].values
# fill: column median on inner-train (all-nan cols -> 0)
med = np.nanmedian(np.where(itr[:,None], Xv, np.nan), axis=0)
med = np.where(np.isnan(med), 0.0, med)
Xf = np.where(np.isnan(Xv), med, Xv)
mu = Xf[itr].mean(0); sd = Xf[itr].std(0); sd[sd==0]=1
Z = (Xf-mu)/sd; Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr], Z[iva]; ytr, yva = y[itr], y[iva]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
print('mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
def evalset(cols_idx, tag):
    out=[]
    for lam in [30,100,300,1000,3000]:
        w = ridge(Ztr[:,list(cols_idx)+[-1]], np.log1p(ytr), lam)
        f = lambda Z: np.expm1(np.clip(Z@w,-5,CAP))
        smear = np.mean(ytr/np.maximum(f(Ztr[:,list(cols_idx)+[-1]]),1e-9))
        out.append((lam, np.mean(np.abs(f(Zva[:,list(cols_idx)+[-1]])*smear-yva))))
    lam,m = min(out, key=lambda t:t[1])
    print('%-22s best lam=%5d inner MAE %.3f' % (tag, lam, m))
    return m
allidx = list(range(len(num_cols)))
e017idx = [i for i,c in enumerate(num_cols) if not c.endswith('_d') and c not in [g for g in num_cols if g.startswith('g_')]]
grididx = [i for i,c in enumerate(num_cols) if c.startswith('g_')]
extra   = [i for i,c in enumerate(num_cols) if c.endswith('_d') and c not in [g for g in num_cols if g.startswith('g_')]]
m_all = evalset(allidx, 'ALL %d'%len(allidx))
m_17  = evalset(e017idx, 'E017 %d'%len(e017idx))
m_g   = evalset(grididx, 'grid only %d'%len(grididx))
m_17g = evalset(e017idx+grididx, 'E017+grid %d'%(len(e017idx)+len(grididx)))
m_all2= evalset(e017idx+grididx+extra, 'E017+grid+extra %d'%(len(e017idx)+len(grididx)+len(extra)))
