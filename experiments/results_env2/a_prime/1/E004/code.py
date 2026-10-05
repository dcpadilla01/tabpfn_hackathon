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


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, d):
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
    blk = {}
    for k in range(1, 14):
        lo, hi = d - 28*k + 1, d - 28*(k-1)
        m = (t['day'] >= lo) & (t['day'] <= hi)
        blk[k] = gsum(m)
        feats[f'blk_{k}'] = blk[k].values
    for nm, L in [('s364', 364), ('s392', 392), ('s336', 336)]:
        lo, hi = d - L + 1, d - L + 28
        m = (t['day'] >= lo) & (t['day'] <= hi)
        feats[f'spend_{nm}'] = gsum(m).values
        feats[f'trips_{nm}'] = gcnt(m).values
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


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, d):
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
    blk = {}
    for k in range(1, 14):
        lo, hi = d - 28*k + 1, d - 28*(k-1)
        m = (t['day'] >= lo) & (t['day'] <= hi)
        blk[k] = gsum(m)
        feats[f'blk_{k}'] = blk[k].values
    for nm, L in [('s364', 364), ('s392', 392), ('s336', 336)]:
        lo, hi = d - L + 1, d - L + 28
        m = (t['day'] >= lo) & (t['day'] <= hi)
        feats[f'spend_{nm}'] = gsum(m).values
        feats[f'trips_{nm}'] = gcnt(m).values
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
    return feats

df = agent_api.build_features(fn)
print('new shape:', df.shape)
base = agent_api.load_saved('e002_mix.parquet')
print('base shape:', base.shape, 'has snapshot_day:', 'snapshot_day' in base.columns)
merged = base.merge(df.reset_index().rename(columns={'index': 'household_key'}), on=['household_key', 'snapshot_day'], how='left') if df.index.name != 'household_key' else base.merge(df.reset_index(), on=['household_key', 'snapshot_day'], how='left')
print('merged shape:', merged.shape)
tt = agent_api.train_targets()
m = tt.merge(merged, on=['household_key', 'snapshot_day'], how='left')
newcols = [c for c in df.columns]
cor = m[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print(cor.round(3).to_string())
print('max nan frac:', merged.isna().mean().max())
p = agent_api.save_table(merged, 'e004_seasonal.parquet')
print('saved:', p)


# ---- cell ----
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


# ---- cell ----
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

def fn_min(view, d):
    f = pd.DataFrame(0.0, index=get_hh(view), columns=['blk_1'])
    return f

t0 = time.time()
df_min = agent_api.build_features(fn_min)
print('MIN OK', df_min.shape, round(time.time()-t0,1), 'sec; index name:', df_min.index.name if df_min.index.name else 'reset?')
print(df_min.head(2))

def fn_full(view, d):
    hh = get_hh(view)
    tx = view.table('transactions')
    hk = tx['household_key'].values; day = tx['day'].values
    sv = tx['sales_value'].values; bid = tx['basket_id'].values
    # one pass over full history: first day, total spend
    g = pd.DataFrame({'hk': hk, 'day': day, 'sv': sv}).groupby('hk').agg(first_day=('day','min'), total=('sv','sum'))
    # window of 14 blocks: day >= d-391 ; block index k = (d-day)//28 + 1 in 1..14
    w = day >= d - 391
    sub = pd.DataFrame({'hk': hk[w], 'blk': ((d - day[w]) // 28) + 1, 'sv': sv[w], 'bid': bid[w]})
    spend_p = sub.groupby(['hk','blk'])['sv'].sum().unstack()
    trips_p = sub[['hk','blk','bid']].drop_duplicates().groupby(['hk','blk']).size().unstack()
    feats = pd.DataFrame(index=hh)
    B = np.column_stack([spend_p.reindex(hh).get(k, 0.0) if isinstance(spend_p.reindex(hh), pd.DataFrame) else 0.0 for k in range(1,15)])
    # simpler: build each column directly
    sp = spend_p.reindex(hh).fillna(0.0)
    tr = trips_p.reindex(hh).fillna(0.0)
    cols = {}
    for k in range(1, 15):
        cols[f'b{k}'] = sp[k].values if k in sp.columns else np.zeros(len(hh))
        cols[f't{k}'] = tr[k].values if k in tr.columns else np.zeros(len(hh))
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
print('FULL OK', df.shape, round(time.time()-t0,1), 'sec')
print('max nan:', df.isna().mean().max())
base = agent_api.load_saved('e002_mix.parquet')
merged = base.merge(df.reset_index(), on=['household_key','snapshot_day'], how='left')
print('merged:', merged.shape)
tt = agent_api.train_targets()
m = tt.merge(merged, on=['household_key','snapshot_day'], how='left')
newcols = [c for c in df.columns]
cor = m[newcols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print(cor.round(3).to_string())
p = agent_api.save_table(merged, 'e004_seasonal.parquet')
print('saved:', p)


# ---- cell ----
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


# ---- cell ----
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
    print('snap', d, 'rows', len(tx), flush=True)
    hk_vals = tx['household_key'].values
    codes, uniq = pd.factorize(hk_vals)
    day = tx['day'].values.astype(np.int64)
    sv = tx['sales_value'].values.astype(np.float64)
    bid = tx['basket_id'].values
    # full-history aggregates via integer groupby (low memory)
    g = pd.DataFrame({'c': codes, 'day': day, 'sv': sv}).groupby('c').agg(first_day=('day','min'), total=('sv','sum'))
    # 14-block window
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
    def col(j, arr):
        out = np.zeros(len(hh)); out[has] = arr[pos[has], j]; return out
    feats = pd.DataFrame(index=hh)
    for k in range(1, 14):
        feats[f'blk_{k}'] = col(k-1, S)
    feats['spend_s336'] = col(11, S); feats['trips_s336'] = col(11, T)
    feats['spend_s364'] = col(12, S); feats['trips_s364'] = col(12, T)
    feats['spend_s392'] = col(13, S); feats['trips_s392'] = col(13, T)
    fd = g['first_day'].reindex(uniq).values.astype(np.float64)
    tot = g['total'].reindex(uniq).values.astype(np.float64)
    fd_hh = np.where(has, fd[np.clip(pos, 0, None)], float(d))
    tot_hh = np.where(has, tot[np.clip(pos, 0, None)], 0.0)
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


# ---- cell ----
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


# ---- cell ----
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
    # aggregate once on the whole history
    g = tx.groupby('household_key').agg(fd=('day', 'min'), tot=('sales_value', 'sum'))
    # 14-block window only
    sub = tx[tx['day'] >= d - 391]
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


# ---- cell ----
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


# ---- cell ----
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
