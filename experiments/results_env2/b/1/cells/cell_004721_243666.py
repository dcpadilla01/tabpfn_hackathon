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
print('nan y:', np.isnan(y).sum(), 'overlap itr&iva:', (itr&iva).sum(), 'neither:', (~itr&~iva).sum())
mu = np.nanmean(Xv[itr],0); sd = np.nanstd(Xv[itr],0); sd[sd==0]=1
print('nan in mu:', np.isnan(mu).sum())
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd)
print('nan in Z:', np.isnan(Z).sum())
Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr], Z[iva]
ytr, yva = y[itr], y[iva]
print('nan Zva:', np.isnan(Zva).sum(), 'nan yva:', np.isnan(yva).sum(), 'nan ytr:', np.isnan(ytr).sum())
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
w2 = ridge(Ztr, ytr, 1000)
print('nan w2:', np.isnan(w2).sum())
p = Zva@w2
print('nan p:', np.isnan(p).sum())
m = np.mean(np.abs(p-yva))
print('raw inner MAE:', m)
