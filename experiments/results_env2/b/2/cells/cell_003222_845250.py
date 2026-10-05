
import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr = ~np.isin(days, [403,431]); va = np.isin(days, [403,431])
mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
Z = ((X-mu)/sd).fillna(0.0).values
Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[va], np.ones(va.sum())]
ytr = y[tr]
A = Ztr.T@Ztr + 1000*np.eye(Ztr.shape[1]); A[-1,-1]-=1000
w = np.linalg.solve(A, Ztr.T@ytr)
res = Zva@w - y[va]
print('baseline local MAE', np.abs(res).mean())
# top |corr| features with residual
Zv = Z[va][:,:len(feats)]
rv = res
cs = []
for j,f in enumerate(feats):
    zj = Zv[:,j]
    s = zj.std()
    if s < 1e-9: continue
    c = np.corrcoef(zj, rv)[0,1]
    cs.append((abs(c), c, f))
cs.sort(reverse=True)
for a,c,f in cs[:35]: print(f'{f:28s} r={c:+.3f}')
