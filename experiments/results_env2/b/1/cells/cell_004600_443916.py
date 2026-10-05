import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day',], how='outer', suffixes=('','_d'))
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
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days); iva = df.snapshot_day.isin(iva_days)
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr.values],0); sd = np.nanstd(Xv[itr.values],0); sd[sd==0]=1
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd); Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr.values], Z[iva.values]
ytr, yva = y[itr.values], y[iva.values]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
w = ridge(Ztr, np.log1p(ytr), 1000)
pred_log = np.expm1(np.clip(Zva@w, -5, CAP))
print('nan in Zva@w:', np.isnan(Zva@w).sum(), ' max log pred:', np.nanmax(Zva@w))
print('nan in pred_log:', np.isnan(pred_log).sum())
ly = Zva@w
print('finite?', np.isfinite(ly).sum(), '/', len(ly))
smear = np.mean(ytr/np.maximum(np.expm1(np.clip(Ztr@w,-5,CAP)),1e-9))
print('smear', smear)
m = np.mean(np.abs(pred_log*smear - yva))
print('log-target lam=1000 inner MAE:', m)
# where do nans come from?
if np.isnan(ly).any():
    idx = np.where(np.isnan(ly))[0][:5]
    print('bad rows:', iva.values.nonzero()[0][idx])
