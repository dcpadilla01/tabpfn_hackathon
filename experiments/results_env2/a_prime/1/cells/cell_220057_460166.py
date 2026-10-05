import numpy as np, pandas as pd, time

base_cols = list(agent_api.load_saved('e002_mix.parquet').columns)
print('base cols:', [c for c in base_cols if 'ten' in c.lower() or 'recen' in c.lower() or 'first' in c.lower()])

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
    t4 = tx[['household_key', 'day', 'sales_value', 'basket_id']]
    blkall = ((d - t4['day']) // 28) + 1
    SP = t4.groupby(['household_key', blkall])['sales_value'].sum().unstack().fillna(0.0)
    sub = t4[t4['day'] >= d - 391]
    blkw = ((d - sub['day']) // 28) + 1
    TR = sub.groupby(['household_key', blkw])['basket_id'].nunique().unstack().fillna(0.0)
    feats = pd.DataFrame(index=hh)
    def sc(k):
        return SP[k].reindex(hh).fillna(0.0).values if k in SP.columns else np.zeros(len(hh))
    def tc(k):
        return TR[k].reindex(hh).fillna(0.0).values if k in TR.columns else np.zeros(len(hh))
    for k in range(1, 14):
        feats[f'blk_{k}'] = sc(k)
    feats['spend_s336'] = sc(12); feats['trips_s336'] = tc(12)
    feats['spend_s364'] = sc(13); feats['trips_s364'] = tc(13)
    feats['spend_s392'] = sc(14); feats['trips_s392'] = tc(14)
    feats['total_all'] = SP.sum(axis=1).reindex(hh).fillna(0.0).values
    feats.index.name = 'household_key'
    print('snap', d, 'ok', flush=True)
    return feats

t0 = time.time()
df = agent_api.build_features(fn_full)
print('BUILD OK', df.shape, round(time.time()-t0,1), 'sec; max nan:', df.isna().mean().max())
base = agent_api.load_saved('e002_mix.parquet')
merged = base.merge(df.reset_index(), on=['household_key','snapshot_day'], how='left')
ten_col = 'tenure' if 'tenure' in merged.columns else None
print('tenure col:', ten_col)
tenure = merged[ten_col].values.astype(np.float64) if ten_col else np.full(len(merged), 84.0)
d_arr = merged['snapshot_day'].values.astype(np.float64)
tenure = np.clip(np.where(np.isnan(tenure), d_arr, tenure), 1.0, None)
avg28 = merged['total_all'].values / tenure * 28.0
merged['avg28_all'] = avg28
wts = 0.85 ** np.arange(0, 13)
Bm = merged[[f'blk_{k}' for k in range(1, 14)]].values
decay_sum = Bm @ wts
nb = np.maximum((tenure // 28).astype(int), 1)
norm = np.array([wts[:min(n, 13)].sum() for n in nb])
merged['decay_mean'] = decay_sum / norm
merged['decay_sum'] = decay_sum
B6 = Bm[:, :6]
x = np.arange(1, 7, dtype=float); xm = x.mean()
merged['slope6'] = ((B6 - B6.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
merged['zero6'] = (B6 == 0).sum(axis=1).astype(float)
merged['max6'] = B6.max(axis=1)
merged['std6'] = B6.std(axis=1)
merged['ratio_b1_avg'] = merged['blk_1'].values / (avg28 + 1.0)
merged['ratio_b1_b2'] = merged['blk_1'].values / (merged['blk_2'].values + 1.0)
merged['s364_ratio'] = merged['spend_s364'].values / (avg28 + 1.0)
print('merged:', merged.shape, 'max nan:', merged.isna().mean().max())
tt = agent_api.train_targets()
m = tt.merge(merged, on=['household_key','snapshot_day'], how='left')
newcols = [c for c in df.columns] + ['avg28_all','decay_mean','decay_sum','slope6','zero6','max6','std6','ratio_b1_avg','ratio_b1_b2','s364_ratio']
cor = m[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print(cor.round(3).to_string())
p = agent_api.save_table(merged, 'e004_seasonal.parquet')
print('saved:', p)
