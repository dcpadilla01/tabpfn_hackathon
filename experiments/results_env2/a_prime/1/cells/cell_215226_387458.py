import numpy as np, pandas as pd, time

def get_hh(view):
    hh = view.households
    if hh is None:
        return pd.Index([], name='household_key')
    if isinstance(hh, pd.DataFrame):
        return pd.Index(hh['household_key'].values, name='household_key')
    if isinstance(hh, pd.Series):
        return pd.Index(hh.values, name='household_key')
    if isinstance(hh, pd.Index):
        return hh
    return pd.Index(np.asarray(hh).ravel(), name='household_key')

def fn_full(view, d):
    hh = get_hh(view)
    tx = view.table('transactions')
    g = tx.groupby('household_key').agg(fd=('day', 'min'), tot=('sales_value', 'sum'))
    sub = tx[tx['day'] >= d - 391].copy()
    sub['blk'] = ((d - sub['day']) // 28) + 1
    sp = sub.groupby(['household_key', 'blk'])['sales_value'].sum().unstack()
    tr = sub.groupby(['household_key', 'blk'])['basket_id'].nunique().unstack()
    feats = pd.DataFrame(index=hh)
    cols = {}
    for k in range(1, 15):
        if k in sp.columns:
            cols[k] = sp[k].reindex(hh).fillna(0.0).values
        else:
            cols[k] = np.zeros(len(hh))
        if k in tr.columns:
            cols[100 + k] = tr[k].reindex(hh).fillna(0.0).values
        else:
            cols[100 + k] = np.zeros(len(hh))
    for k in range(1, 14):
        feats[f'blk_{k}'] = cols[k]
    feats['spend_s336'] = cols[12]; feats['trips_s336'] = cols[112]
    feats['spend_s364'] = cols[13]; feats['trips_s364'] = cols[113]
    feats['spend_s392'] = cols[14]; feats['trips_s392'] = cols[114]
    fd = g['fd'].reindex(hh).fillna(d).values.astype(np.float64)
    tot = g['tot'].reindex(hh).fillna(0.0).values.astype(np.float64)
    tenure = np.clip(d - fd + 1.0, 1.0, None)
    avg28 = tot / tenure * 28.0
    feats['avg28_all'] = avg28
    wts = 0.85 ** np.arange(0, 13)
    Bm = np.column_stack([cols[k] for k in range(1, 14)])
    decay_sum = Bm @ wts
    nb = np.maximum((tenure // 28).astype(int), 1)
    norm = np.array([wts[:min(n, 13)].sum() for n in nb])
    feats['decay_mean'] = decay_sum / norm
    feats['decay_sum'] = decay_sum
    B6 = Bm[:, :6]
    x = np.arange(1, 7, dtype=float); xm = x.mean()
    feats['slope6'] = ((B6 - B6.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
    feats['zero6'] = (B6 == 0).sum(axis=1).astype(float)
    feats['max6'] = B6.max(axis=1)
    feats['std6'] = B6.std(axis=1)
    feats['ratio_b1_avg'] = cols[1] / (avg28 + 1.0)
    feats['ratio_b1_b2'] = cols[1] / (cols[2] + 1.0)
    feats['s364_ratio'] = cols[13] / (avg28 + 1.0)
    feats.index.name = 'household_key'
    return feats

t0 = time.time()
df = agent_api.build_features(fn_full)
print('BUILD OK', df.shape, round(time.time()-t0,1), 'sec; max nan:', df.isna().mean().max())
base = agent_api.load_saved('e002_mix.parquet')
merged = base.merge(df.reset_index(), on=['household_key','snapshot_day'], how='left')
print('merged:', merged.shape)
tt = agent_api.train_targets()
m = tt.merge(merged, on=['household_key','snapshot_day'], how='left')
newcols = list(df.columns)
cor = m[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print(cor.round(3).to_string())
p = agent_api.save_table(merged, 'e004_seasonal.parquet')
print('saved:', p)
