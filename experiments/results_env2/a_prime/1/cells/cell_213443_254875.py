import numpy as np, pandas as pd

_BASE = agent_api.load_saved('e002_mix.parquet')
print('base ncols:', len(_BASE.columns), 'rows:', len(_BASE))
print('has snapshot_day:', 'snapshot_day' in _BASE.columns)

def fn(view, d):
    try:
        base = _BASE
    except NameError:
        base = agent_api.load_saved('e002_mix.parquet')
    if 'snapshot_day' in base.columns:
        base_d = base[base['snapshot_day'] == d].set_index('household_key')
    else:
        base_d = base
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        hh = pd.Index(hh['household_key'].values) if 'household_key' in hh.columns else pd.Index(hh.index)
    else:
        hh = pd.Index(np.asarray(hh).ravel())
    tx = view.table('transactions')
    t = pd.DataFrame({'hk': tx['household_key'].values, 'day': tx['day'].values,
                      'sv': tx['sales_value'].values, 'bid': tx['basket_id'].values})
    feats = pd.DataFrame(index=hh)
    def gsum(mask):
        return t.loc[mask].groupby('hk')['sv'].sum().reindex(hh).fillna(0.0)
    def gcnt(mask):
        return t.loc[mask].groupby('hk')['bid'].nunique().reindex(hh).fillna(0.0)
    # non-overlapping 28-day blocks, k=1 most recent
    blk = {}
    for k in range(1, 14):
        lo, hi = d - 28*k + 1, d - 28*(k-1)
        m = (t['day'] >= lo) & (t['day'] <= hi)
        blk[k] = gsum(m)
        feats[f'blk_{k}'] = blk[k].values
    # seasonal lags of the target window [d+1, d+28]
    for nm, L in [('s364', 364), ('s392', 392), ('s336', 336)]:
        lo, hi = d - L + 1, d - L + 28
        m = (t['day'] >= lo) & (t['day'] <= hi)
        feats[f'spend_{nm}'] = gsum(m).values
        feats[f'trips_{nm}'] = gcnt(m).values
    # level / decay / trend
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
    return base_d.join(feats, how='left')

df = agent_api.build_features(fn)
print('shape:', df.shape)
tt = agent_api.train_targets()
m = tt.merge(df, on=['household_key', 'snapshot_day'], how='left')
newcols = [c for c in df.columns if c.startswith(('blk_', 'spend_s', 'trips_s', 'avg28', 'decay', 'slope6', 'zero6', 'max6', 'std6', 'ratio_'))]
print('n new cols:', len(newcols))
cor = m[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print(cor.round(3).to_string())
print('max nan frac:', df.isna().mean().max())
p = agent_api.save_table(df, 'e004_seasonal.parquet')
print('saved:', p)
