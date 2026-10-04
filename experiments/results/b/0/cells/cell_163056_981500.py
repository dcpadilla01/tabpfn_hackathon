import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
yv = df['future_spend_4w'].values.astype(float)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
tr403 = (df.snapshot_day<=403).values; va431=(df.snapshot_day==431).values
tr375 = (df.snapshot_day<=375).values; va403=(df.snapshot_day==403).values
mu = X[tr403].mean(); sd = X[tr403].std().replace(0,1.0)
Z = ((X-mu)/sd).fillna(0.0).values
def mae_for(a, trm, vam):
    A = Z[trm].T@Z[trm] + a*np.eye(Z.shape[1]); b = Z[trm].T@yv[trm]
    w = np.linalg.solve(A,b); p = Z[vam]@w
    return float(np.abs(p-yv[vam]).mean())
print('val431:', [(a, round(mae_for(a,tr403,va431),3)) for a in [3,10,30,100,300,1000,3000]])
print('val403:', [(a, round(mae_for(a,tr375,va403),3)) for a in [10,30,100,300,1000]])
print('mean-pred MAE431:', round(float(np.abs(yv[tr403].mean()-yv[va431]).mean()),3))
print('median-pred MAE431:', round(float(np.abs(np.median(yv[tr403])-yv[va431]).mean()),3))
print('target describe:', {k: round(float(v),2) for k,v in df.future_spend_4w.describe().items()})
print('zero share:', round(float((df.future_spend_4w==0).mean()),4))
print('rows/snap:', df.groupby('snapshot_day').size().to_dict())
v = agent_api.snapshot()
print('day', v.day, 'week', v.week)
print('households type:', type(v.households))
tx = v.transactions
print('tx', tx.shape)
print('neg sales lines:', int((tx.sales_value<0).sum()), 'neg qty:', int((tx.quantity<0).sum()))
print('sales describe:', {k: round(float(vv),2) for k,vv in tx.sales_value.describe().items()})
