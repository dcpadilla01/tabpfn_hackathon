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
    t0 = time.time()
    hh = get_hh(view)
    tx = view.table('transactions')
    print('snap', d, 'rows', len(tx), 'nhh', len(hh), flush=True)
    hk_vals = tx['household_key'].values
    codes, uniq = pd.factorize(hk_vals)
    uniq = pd.Index(uniq)
    day = tx['day'].values.astype(np.int64)
    sv = tx['sales_value'].values.astype(np.float64)
    bid = tx['basket_id'].values
    g = pd.DataFrame({'c': codes, 'day': day, 'sv': sv}).groupby('c').agg(first_day=('day','min'), total=('sv','sum'))
    w = day >= d - 391
    cw = codes[w]; dw = day[w]; svw = sv[w]; bidw = bid[w]
    blk = ((d - dw) // 28) + 1
    key = cw * 14 + (blk - 1)
    S = np.bincount(key, weights=svw, minlength=len(uniq) * 14).reshape(len(uniq), 14)
    bc_codes, _ = pd.factorize(bidw)
    df_u = pd.DataFrame({'c': cw, 'b': blk, 'bc': bc_codes}).drop_duplicates()
    T = np.zeros(len(uniq) * 14, dtype=np.float64)
    np.add.at(T, df_u['c'].values * 14 + (df_u['b'].values - 1), 1.0)
    T = T.reshape(len(uniq), 14)
    pos = uniq.get_indexer(hh)
    has = pos >= 0
    posc = np.clip(pos, 0, None)
    def col(j, arr):
        out = np.zeros(len(hh)); out[has] = arr[posc[has], j]; return out
    feats = pd.DataFrame(index=hh)
    for k in range(1, 14):
        feats[f'blk_{k}'] = col(k-1, S)
    feats['spend_s336'] = col(11, S); feats['trips_s336'] = col(11, T)
    feats['spend_s364'] = col(12, S); feats['trips_s364'] = col(12, T)
    feats['spend_s392'] = col(13, S); feats['trips_s392'] = col(13, T)
    fd = g['first_day'].reindex(uniq).values.astype(np.float64)
    tot = g['total'].reindex(uniq).values.astype(np.float64)
    fd_hh = np.where(has, fd[posc], float(d))
    tot_hh = np.where(has, tot[posc], 0.0)
    tenure = np.clip(d - fd_hh + 1, 1, None)
    avg28 = tot_hh / tenure * 28.0
    feats['avg28_all'] = avg28
    wts = 0.85 ** np.arange(0, 13)
    Bm = np.column_stack([col(k-1, S) for k in range(1, 14)])
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
    b1 = col(0, S); b2 = col(1, S)
    feats['ratio_b1_avg'] = b1 / (avg28 + 1.0)
    feats['ratio_b1_b2'] = b1 / (b2 + 1.0)
    feats['s364_ratio'] = col(12, S) / (avg28 + 1.0)
    feats.index.name = 'household_key'
    print('snap', d, 'done', round(time.time()-t0, 2), 's', flush=True)
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
