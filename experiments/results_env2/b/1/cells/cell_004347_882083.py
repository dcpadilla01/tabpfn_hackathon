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

itr = df[df.snapshot_day.isin(itr_days)]; iva = df[df.snapshot_day.isin(iva_days)]
def prep(d):
    X = d[num_cols].astype(float).values
    return X, d['future_spend_4w'].values
Xtr,ytr = prep(itr); Xva,yva = prep(iva)
mu, sd = np.nanmean(Xtr,0), np.nanstd(Xtr,0); sd[sd==0]=1
Ztr = np.where(np.isnan(Xtr),0,(Xtr-mu)/sd); Zva = np.where(np.isnan(Xva),0,(Xva-mu)/sd)
Ztr = np.c_[Ztr, np.ones(len(Ztr))]; Zva = np.c_[Zva, np.ones(len(Zva))]
CAP = np.log1p(2000.0)
def ridge(Z,y,lam):
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Z.T@y)

print('baseline mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
for lam in [10,30,100,300,1000,3000]:
    w = ridge(Ztr, np.log1p(ytr), lam)
    f = lambda Z: np.expm1(np.clip(Z@w, -5, CAP))
    smear = np.mean(ytr/np.maximum(f(Ztr),1e-9))
    print('lam=%5d log-target: inner MAE %.3f | raw-target MAE %.3f' % (lam, np.mean(np.abs(f(Zva)*smear-yva)), np.mean(np.abs(f(Ztr)*smear-ytr))))
# raw target
for lam in [100,1000,3000,10000]:
    w = ridge(Ztr, ytr, lam)
    print('lam=%5d raw-target : inner MAE %.3f | train MAE %.3f' % (lam, np.mean(np.abs(Zva@w-yva)), np.mean(np.abs(Ztr@w-ytr))))
