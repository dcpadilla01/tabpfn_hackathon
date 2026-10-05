import numpy as np, pandas as pd, time

t0 = time.time()
view = agent_api.snapshot(95)
d = 95
hh = view.households
print('households type:', type(hh))
if isinstance(hh, pd.DataFrame):
    hh = pd.Index(hh['household_key'].values) if 'household_key' in hh.columns else pd.Index(hh.index)
else:
    hh = pd.Index(np.asarray(hh).ravel())
print('n hh:', len(hh), 'time', round(time.time()-t0,1))
tx = view.table('transactions')
print('tx shape:', tx.shape, 'time', round(time.time()-t0,1))
t = pd.DataFrame({'hk': tx['household_key'].values, 'day': tx['day'].values,
                  'sv': tx['sales_value'].values, 'bid': tx['basket_id'].values})
feats = pd.DataFrame(index=hh)
def gsum(mask):
    return t.loc[mask].groupby('hk')['sv'].sum().reindex(hh).fillna(0.0)
def gcnt(mask):
    return t.loc[mask].groupby('hk')['bid'].nunique().reindex(hh).fillna(0.0)
blk = {}
for k in range(1, 14):
    lo, hi = d - 28*k + 1, d - 28*(k-1)
    m = (t['day'] >= lo) & (t['day'] <= hi)
    blk[k] = gsum(m)
    feats[f'blk_{k}'] = blk[k].values
print('blocks done', round(time.time()-t0,1))
for nm, L in [('s364', 364), ('s392', 392), ('s336', 336)]:
    lo, hi = d - L + 1, d - L + 28
    m = (t['day'] >= lo) & (t['day'] <= hi)
    feats[f'spend_{nm}'] = gsum(m).values
    feats[f'trips_{nm}'] = gcnt(m).values
print('seasonal done', round(time.time()-t0,1))
first_day = t.groupby('hk')['day'].min().reindex(hh).fillna(d)
tenure = (d - first_day + 1).clip(lower=1)
total = t.groupby('hk')['sv'].sum().reindex(hh).fillna(0.0)
avg28 = total / tenure * 28.0
feats['avg28_all'] = avg28.values
w = 0.85 ** np.arange(0, 13)
B = np.column_stack([blk[k].values for k in range(1, 14)])
decay_sum = B @ w
nb = np.maximum((tenure.values // 28).astype(int), 1)
norm = np.array([w[:min(n, 13)].sum() for n in nb])
feats['decay_mean'] = decay_sum / norm
feats['decay_sum'] = decay_sum
B6 = B[:, :6]
x = np.arange(1, 7, dtype=float); xm = x.mean()
slope = ((B6 - B6.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
feats['slope6'] = slope
feats['zero6'] = (B6 == 0).sum(axis=1).astype(float)
feats['max6'] = B6.max(axis=1)
feats['std6'] = B6.std(axis=1)
feats['ratio_b1_avg'] = (blk[1] / (avg28 + 1.0)).values
feats['ratio_b1_b2'] = (blk[1] / (blk[2] + 1.0)).values
feats['s364_ratio'] = (feats['spend_s364'] / (avg28.values + 1.0)).values
print('all done', round(time.time()-t0,1), feats.shape)
print(feats.describe().T[['mean','std']].round(2).to_string())
