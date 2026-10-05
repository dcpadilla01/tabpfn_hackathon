import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']
t = load_saved('e017_xsec_rank.parquet').merge(tt, on=KEY, how='left')
t = t[t['future_spend_4w'].notna()]
fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
X = t[fc].copy()
for c in fc:
    if X[c].dtype == bool: X[c] = X[c].astype(float)
    elif not np.issubdtype(X[c].dtype, np.number): X[c] = pd.Categorical(X[c]).codes.astype(float)
X = X.values.astype(float); y = t['future_spend_4w'].values.astype(float); d = t['snapshot_day'].values

tr = d <= 375; va = np.isin(d, (403,431))
mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
Z = np.nan_to_num((X-mu)/sd)
A = Z[tr].T@Z[tr] + 3000*np.eye(Z.shape[1])
w = np.linalg.solve(A, Z[tr].T@y[tr])
pred = Z@w

dfp = pd.DataFrame({'d':d, 'y':y, 'p':pred})
print(dfp.groupby('d').agg(y_mean=('y','mean'), p_mean=('p','mean'), mae=('p', lambda p: np.nan)).shape)
g = dfp.groupby('d').apply(lambda g: pd.Series({'y_mean':g.y.mean(),'p_mean':g.p.mean(),'mae':(g.p-g.y).abs().mean()}))
print(g.round(1))

# simple model: spend_28 alone
i = fc.index('spend_28')
x1 = np.nan_to_num((X[:,i]-mu[i])/sd[i])
A1 = x1[tr].T@x1[tr] + 3000
w1 = np.linalg.solve(A1, x1[tr]@y[tr])
p1 = x1*w1
dfp['p1'] = p1
g1 = dfp.groupby('d').apply(lambda g: pd.Series({'mae1':(g.p1-g.y).abs().mean(),'corr':g.p1.corr(g.y)}))
print(g1.round(2))
print('overall corr(spend_28, y) by day:')
print(t.assign(x=X[:,i]).groupby('snapshot_day').apply(lambda g: g['x'].corr(g['future_spend_4w'])).round(3))
