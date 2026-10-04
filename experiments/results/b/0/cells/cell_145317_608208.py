import numpy as np, pandas as pd

# Test 1: can build_features fn use load_saved? trivial test on 2 snapshots via manual call pattern
saved = agent_api.load_saved('mkt_demo.parquet')

def test_fn(view, snapshot_day):
    hh = view.households
    sub = saved[saved.snapshot_day == snapshot_day][['household_key','spend28']]
    return hh[['household_key']].merge(sub, on='household_key', how='left').set_index('household_key')

# don't run full build_features (expensive); just check one snapshot manually
v = agent_api.snapshot(431)
out = test_fn(v, 431)
print('merge-in-fn OK:', out.shape, out.spend28.notna().mean())

# Test 2: time + sanity of new features at snapshot 431
v = agent_api.snapshot(431)
tx = v.transactions
prods = v.products[['product_id','department']]
tx = tx.merge(prods, on='product_id', how='left')
d0 = 431
tx28 = tx[tx.day > d0-28]
print('tx28 rows:', len(tx28))

# dept spend pivot
ds = tx28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
ds.columns = ['dsp_' + c for c in ds.columns]
print('dept spend cols:', ds.shape)

# weekly spend pattern last 28d
w = tx28.assign(wk=(d0 - tx28.day)//7)  # 0=most recent week
ws = w.groupby(['household_key','wk']).sales_value.sum().unstack(fill_value=0.0)
ws.columns = [f'spend_w{c}' for c in ws.columns]
print('weekly cols:', list(ws.columns))

# trip intervals last 112d
tx112 = tx[tx.day > d0-112]
g = tx112.groupby('household_key').day.apply(lambda s: np.diff(np.sort(s.unique())))
iv = pd.DataFrame({'hh': g.index, 'iv': g.values})
iv['mean_iv'] = iv.iv.apply(lambda a: a.mean() if len(a) else np.nan)
iv['std_iv'] = iv.iv.apply(lambda a: a.std() if len(a)>1 else 0.0)
iv['min_iv'] = iv.iv.apply(lambda a: a.min() if len(a) else np.nan)
print('interval stats sample:', iv[['mean_iv','std_iv','min_iv']].describe().round(1))

# basket stats 28d
bk = tx28.groupby(['household_key','basket_id']).sales_value.sum()
bs = bk.groupby('household_key').agg(bk_mean28='mean', bk_max28='max', bk_std28='std', nbaskets28='count')
print(bs.describe().round(1))

# recency-weighted spend 84d
tx84 = tx[tx.day > d0-84]
rw = tx84.assign(w=np.exp(-(d0-tx84.day)/28.0)).groupby('household_key').apply(lambda g: (g.sales_value*g.w).sum())
print('recency-weighted spend:', rw.describe().round(1))