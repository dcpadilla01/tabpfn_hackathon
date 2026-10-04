import agent_api as A, pandas as pd, numpy as np

def keys_of(view):
    h = view.households
    if isinstance(h, pd.DataFrame):
        return pd.Index(h['household_key'].values, name='household_key')
    if isinstance(h, pd.Index):
        return pd.Index(h, name='household_key')
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
        c = 'fwd28_k%d' % k
        f[c] = w.groupby('household_key')['sales_value'].sum()
        cols.append(c)
        prev_hi = lo
    V = f[cols]
    f['fwd28_mean'] = V.mean(axis=1)
    f['fwd28_median'] = V.median(axis=1)
    f['fwd28_std'] = V.std(axis=1)
    f['fwd28_min'] = V.min(axis=1)
    f['fwd28_max'] = V.max(axis=1)
    f['fwd28_zero_ct'] = ((V == 0) & V.notna()).sum(axis=1)
    f['fwd28_latest_ratio'] = f['fwd28_k1'] / f['fwd28_mean'].where(f['fwd28_mean'] > 0)
    dd = tx.groupby(['household_key', 'day'], as_index=False)['sales_value'].sum()
    for hl, cname in [(14, 'ewma14_4w'), (28, 'ewma28_4w')]:
        def wmean(g, hl=hl, day=day):
            age = day - g['day'].values
            wts = np.exp(-np.log(2.0) * age / hl)
            den = wts.sum()
            return (g['sales_value'].values * wts).sum() / den * 28.0 if den > 0 else np.nan
        f[cname] = dd.groupby('household_key').apply(wmean)
    for c in ['fwd28_k1','fwd28_k2','fwd28_k3','fwd28_k4','fwd28_k5','fwd28_k6','ewma14_4w','ewma28_4w']:
        f['log_' + c] = np.log1p(f[c])
    return f

table = A.build_features(build)
print('table', table.shape)
print(table.columns.tolist())
nn = table.drop(columns=['household_key','snapshot_day']).isna().mean().round(3)
print(nn.to_string())
tt = A.train_targets()
m = table.merge(tt, on=['household_key','snapshot_day'])
print('merged train rows', len(m))
cors = m.drop(columns=['household_key','snapshot_day']).corr()['future_spend_4w'].drop('future_spend_4w').sort_values()
print(cors.round(3).to_string())
path = A.save_table(table, 'e006_fwd_profile')
print('saved', path)
