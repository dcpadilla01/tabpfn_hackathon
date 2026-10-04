import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 250)

t = agent_api.load_saved('e011_price.parquet')
print('E011 shape:', t.shape)
print(t.dtypes.value_counts())
nonnum = [c for c in t.columns if not pd.api.types.is_numeric_dtype(t[c])]
print('non-numeric:', nonnum)
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))

tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', df.shape)
yv = df['future_spend_4w'].values.astype(float)
print('target describe:'); print(df['future_spend_4w'].describe())
print('zero share:', float((df.future_spend_4w==0).mean()))
print('rows per snapshot:'); print(df.groupby('snapshot_day').size().to_dict())

feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
print('overall NaN frac:', round(float(X.isna().mean().mean()),4))

tr403 = (df.snapshot_day<=403).values; va431 = (df.snapshot_day==431).values
tr375 = (df.snapshot_day<=375).values; va403 = (df.snapshot_day==403).values

mu = X[tr403].mean(); sd = X[tr403].std().replace(0,1.0)
Z = ((X-mu)/sd).fillna(0.0).values

def mae_for(a, trm, vam):
    A = Z[trm].T@Z[trm] + a*np.eye(Z.shape[1])
    b = Z[trm].T@yv[trm]
    w = np.linalg.solve(A,b)
    p = Z[vam]@w
    return float(np.abs(p-yv[vam]).mean())

print('--- ridge alpha sweep, val=431 (train<=403) ---')
for a in [1,3,10,30,100,300,1000,3000]:
    print(a, round(mae_for(a, tr403, va431),3))
print('--- val=403 (train<=375) ---')
for a in [10,30,100,300,1000]:
    print(a, round(mae_for(a, tr375, va403),3))

print('naive mean pred MAE431:', round(float(np.abs(yv[tr403].mean()-yv[va431]).mean()),3))

Xtr = X[tr403].copy(); ytr = pd.Series(yv[tr403])
corr = Xtr.corrwith(ytr)
top = corr.abs().sort_values(ascending=False).head(30)
print('top |corr| features:')
for c in top.index: print(f'  {c}: {corr[c]:+.3f}')

c28 = [c for c in X.columns if 'spend' in c.lower() and '28' in c]
print('spend28 candidates:', c28)
if c28:
    x = X[c28[0]].values
    m = tr403 & ~np.isnan(x)
    b_, a_ = np.polyfit(x[m], yv[m], 1)
    mv = va431 & ~np.isnan(x)
    p = a_ + b_*x[mv]
    print('lin spend28 MAE431:', round(float(np.abs(p-yv[mv]).mean()),3), 'slope', round(b_,3), 'icept', round(a_,2))

v = agent_api.snapshot()
txv = v.transactions
print('tx shape', txv.shape)
print(txv.head(3))
print('neg sales lines:', int((txv.sales_value<0).sum()), 'neg qty:', int((txv.quantity<0).sum()))
print('households attr:', type(v.households), (v.households.shape if hasattr(v.households,'shape') else len(v.households)))
print('day', v.day, 'week', v.week)
print('snapshot_days:', agent_api.snapshot_days())
