import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')

print('mean/zero-rate of target by snapshot day:')
g = m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda x:(x==0).mean(),'count'])
g.columns=['mean','median','zero_rate','n']; print(g.round(2))

# churn structure: rows with recent activity but zero future spend
m['spend84_0'] = m['spend_84']==0
print('\nby spend_84==0:')
print(m.groupby('spend84_0')['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

m['rec28'] = m['spend_28']==0
print('\nby spend_28==0:')
print(m.groupby('rec28')['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# zero-streak (consecutive trailing 28d zero blocks) vs target
print('\nby fwd28_zero_ct (count of trailing 28d blocks with zero spend, 0-6):')
print(m.groupby(m['fwd28_zero_ct'].fillna(6).astype(int))['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# recency buckets
print('\nby recency (days since last purchase):')
m['rec_b'] = pd.cut(m['recency'], [-1,7,14,28,56,112,10000])
print(m.groupby('rec_b', observed=True)['future_spend_4w'].agg(['mean','count',lambda x:(x==0).mean()]).round(2))

# error decomposition of current best proxy model on day 431
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
trd = [d for d in agent_api.snapshot_days()['train'] if d!=431]
tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day==431]
Xtr = tr[feats].astype(float).fillna(0).values; Xva = va[feats].astype(float).fillna(0).values
mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
w = np.linalg.solve(A.T@A+1.0*np.eye(A.shape[1]), A.T@tr['future_spend_4w'].values)
p = B@w; y = va['future_spend_4w'].values
err = p-y
print('\nday431 proxy MAE %.2f | by y==0: n=%d, mae=%.1f, meanpred=%.1f | by y>0: mae=%.1f' % (
    np.abs(err).mean(), (y==0).sum(), np.abs(err[y==0]).mean(), p[y==0].mean(), np.abs(err[y>0]).mean()))
print('share of total abs err from y==0 rows: %.1f%%' % (100*np.abs(err[y==0]).sum()/np.abs(err).sum()))
print('share from y>top-decile: %.1f%%' % (100*np.abs(err[y>=np.quantile(y,0.9)]).sum()/np.abs(err).sum()))