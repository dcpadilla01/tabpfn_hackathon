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
# drop exact-duplicate columns (from the _d suffix merge)
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
print('num cols after dedup:', len(num_cols))

itr = df[df.snapshot_day.isin(itr_days)]; iva = df[df.snapshot_day.isin(iva_days)]
Xtr = itr[num_cols].astype(float).values; Xva = iva[num_cols].astype(float).values
ytr = itr['future_spend_4w'].values;     yva = iva['future_spend_4w'].values
mu, sd = np.nanmean(Xtr,0), np.nanstd(Xtr,0); sd[sd==0]=1
Ztr = np.where(np.isnan(Xtr),0,(Xtr-mu)/sd); Zva = np.where(np.isnan(Xva),0,(Xva-mu)/sd)
Ztr = np.c_[Ztr, np.ones(len(Ztr))]; Zva = np.c_[Zva, np.ones(len(Zva))]

def ridge_fit(Z,y,lam):
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Z.T@y)

# log-target ridge usually better for skewed spend; predict expm1 with smearing
ly = np.log1p(ytr)
best=None
for lam in [10,30,100,300,1000]:
    w = ridge_fit(Ztr, ly, lam)
    p = np.expm1(Zva@w); s = np.mean(np.exp(Ztr@w - (Ztr@w)))  # smearing ~1 on train
    m = np.mean(np.abs(p-yva))
    pw = np.expm1(Ztr@w); mw = np.mean(np.abs(pw-ytr))
    print('lam=%5d  inner-val MAE %.3f (train MAE %.3f)' % (lam, m, mw))
    if best is None or m<best[1]: best=(lam,m,w)
print('best lam', best[0])
