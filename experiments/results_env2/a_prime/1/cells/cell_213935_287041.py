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
    hk = tx['household_key'].values; day = tx['day'].values
    sv = tx['sales_value'].values; bid = tx['basket_id'].values
    g = pd.DataFrame({'hk': hk, 'day': day, 'sv': sv}).groupby('hk').agg(first_day=('day','min'), total=('sv','sum'))
    w = day >= d - 391
    sub = pd.DataFrame({'hk': hk[w], 'blk': ((d - day[w]) // 28) + 1, 'sv': sv[w], 'bid': bid[w]})
    spend_p = sub.groupby(['hk','blk'])['sv'].sum().unstack()
    trips_p = sub[['hk','blk','bid']].drop_duplicates().groupby(['hk','blk']).size().unstack()
    sp = spend_p.reindex(hh).fillna(0.0)
    tr = trips_p.reindex(hh).fillna(0.0)
    cols = {}
    for k in range(1, 15):
        cols[f'b{k}'] = sp[k].values if k in sp.columns else np.zeros(len(hh))
        cols[f't{k}'] = tr[k].values if k in tr.columns else np.zeros(len(hh))
    feats = pd.DataFrame(index=hh)
    for k in range(1, 14):
        feats[f'blk_{k}'] = cols[f'b{k}']
    feats['spend_s336'] = cols['b12']; feats['trips_s336'] = cols['t12']
    feats['spend_s364'] = cols['b13']; feats['trips_s364'] = cols['t13']
    feats['spend_s392'] = cols['b14']; feats['trips_s392'] = cols['t14']
    first_day = g['first_day'].reindex(hh).fillna(d)
    tenure = (d - first_day + 1).clip(lower=1)
    avg28 = (g['total'].reindex(hh).fillna(0.0) / tenure * 28.0).values
    feats['avg28_all'] = avg28
    wts = 0.85 ** np.arange(0, 13)
    Bm = np.column_stack([cols[f'b{k}'] for k in range(1, 14)])
    decay_sum = Bm @ wts
    nb = np.maximum((tenure.values // 28).astype(int), 1)
    norm = np.array([wts[:min(n, 13)].sum() for n in nb])
    feats['decay_mean'] = decay_sum / norm
    feats['decay_sum'] = decay_sum
    B6 = Bm[:, :6]
    x = np.arange(1, 7, dtype=float); xm = x.mean()
    feats['slope6'] = ((B6 - B6.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
    feats['zero6'] = (B6 == 0).sum(axis=1).astype(float)
    feats['max6'] = B6.max(axis=1)
    feats['std6'] = B6.std(axis=1)
    feats['ratio_b1_avg'] = cols['b1'] / (avg28 + 1.0)
    feats['ratio_b1_b2'] = cols['b1'] / (cols['b2'] + 1.0)
    feats['s364_ratio'] = cols['b13'] / (avg28 + 1.0)
    feats.index.name = 'household_key'
    return feats

t0 = time.time()
df = agent_api.build_features(fn_full)
print('FULL OK', df.shape, round(time.time()-t0,1), 'sec; max nan:', df.isna().mean().max())
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
