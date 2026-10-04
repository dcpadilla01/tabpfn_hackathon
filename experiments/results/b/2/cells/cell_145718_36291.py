import agent_api as A, pandas as pd, numpy as np

v = A.snapshot()
tx0 = v.transactions
print('tx shape', tx0.shape)
print(tx0[['sales_value','coupon_disc','coupon_match_disc','retail_disc']].describe().loc[['mean','min','max']].to_string())
cr = v.coupon_redemptions
print('cr shape', cr.shape); print(cr.head(5).to_string())
h = v.households
print('hh container', repr(h)[:200], 'len', len(h))

def keys_of(view):
    h = view.households
    if isinstance(h, pd.DataFrame):
        return pd.Index(h['household_key'].values, name='household_key')
    return pd.Index(list(h), name='household_key')

def build(view, day):
    idx = keys_of(view)
    keyset = set(idx)
    tx = view.transactions
    tx = tx[tx.household_key.isin(keyset)]
    f = pd.DataFrame(index=idx)
    prev_hi = day
    cols = []
    for k in range(1, 7):
        lo = day - 28*k
        w = tx[(tx.day > lo) & (tx.day <= prev_hi)]
        c = f'fwd28_k{k}'
        f[c] = w.groupby('household_key')['sales_value'].sum()
        cols.append(c)
        prev_hi = lo
    V = f[cols]
    f['fwd28_mean'] = V.mean(axis=1)
    f['fwd28_median'] = V.median(axis=1)
    f['fwd28_std'] = V.std(axis=1)
    f['fwd28_min'] = V.min(axis=1)
    f['fwd28_max'] = V.max(axis=1)
    f['fwd28_zero_ct'] = (V == 0).sum(axis=1)
    f['fwd28_latest_ratio'] = f['fwd28_k1'] / f['fwd28_mean'].where(f['fwd28_mean'] > 0)
    dd = tx.groupby(['household_key', 'day'], as_index=False)['sales_value'].sum()
    for hl, cname in [(14, 'ewma14_4w'), (28, 'ewma28_4w')]:
        def wmean(g, hl=hl, day=day):
            age = day - g['day'].values
            wts = np.exp(-np.log(2.0) * age / hl)
            den = wts.sum()
            return (g['sales_value'].values * wts).sum() / den * 28.0 if den > 0 else np.nan
        f[cname] = dd.groupby('household_key').apply(wmean)
    return f

table = A.build_features(build)
print('table', table.shape)
print(table.columns.tolist())
b = A.baseline_features()
chk = b[['household_key','snapshot_day']].merge(table[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='outer', indicator=True)
print(chk['_merge'].value_counts())
print(table.drop(columns=['household_key','snapshot_day']).isna().mean().round(3).to_string())
path = A.save_table(table, 'e006_fwd_profile')
print('saved', path)
print(table.head(3).to_string())
