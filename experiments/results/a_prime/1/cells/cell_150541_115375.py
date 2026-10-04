import numpy as np, pandas as pd, agent_api
t1 = agent_api.load_saved('e001_txhist.parquet'); t3 = agent_api.load_saved('e003_catmix.parquet')
print('e001', t1.shape, 'e003', t3.shape)
print('E001 cols:', list(t1.columns))
print('E003 extra:', [c for c in t3.columns if c not in t1.columns])
tt = agent_api.train_targets()
d = t3.merge(tt, on=['household_key','snapshot_day'])
y = d['future_spend_4w']
print('target:'); print(y.describe())
print('zero share:', round((y==0).mean(),3))
print('mean target by snapshot_day:'); print(d.groupby('snapshot_day')['future_spend_4w'].mean().round(1))
num = [c for c in d.columns if d[c].dtype.kind in 'ifbu' and c not in ('snapshot_day','future_spend_4w')]
cor = d[num].corrwith(y)
print('top |corr| with target:'); print(cor.reindex(cor.abs().sort_values(ascending=False).index).head(25).round(3))
spend28 = [c for c in t1.columns if 'spend' in c.lower() and '28' in c]
print('spend28-like cols:', spend28)
for c in spend28[:4]:
    print(c, 'proxy MAE:', round(np.abs(d[c]-y).mean(),2))
c0 = spend28[0] if spend28 else None
if c0:
    print('quantile profile of', c0); print(d.groupby(pd.qcut(d[c0],10,duplicates='drop'), observed=True)['future_spend_4w'].agg(['mean','count']))
    print('spend28==0 rows:', round((d[c0]==0).mean(),3)); print(d.loc[d[c0]==0,'future_spend_4w'].describe())
