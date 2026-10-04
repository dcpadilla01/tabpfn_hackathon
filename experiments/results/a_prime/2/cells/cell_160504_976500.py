
import agent_api, pandas as pd, numpy as np

t9 = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t9.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]

# --- diagnostics ---
print('corr(index, y) train:', round(np.corrcoef(tr['index'], tr['future_spend_4w'])[0,1], 4))
g = tr[tr.snapshot_day <= 403].groupby('household_key')['future_spend_4w'].mean()
ev = tr[tr.snapshot_day == 431]
pred = ev['household_key'].map(g).fillna(tr['future_spend_4w'].mean())
print('oracle household-mean MAE @431:', round(np.abs(pred - ev['future_spend_4w']).mean(), 2),
      '| frac hh w/o earlier snap:', round(ev['household_key'].isin(g.index).mean(), 3))

# --- prune list (train-determined) ---
feat_cols = [c for c in t9.columns if c not in ('household_key','snapshot_day')]
const_cols = [c for c in feat_cols if tr[c].nunique(dropna=True) <= 1]
allnan_cols = [c for c in feat_cols if tr[c].isna().all()]
drop = sorted(set(['index'] + const_cols + allnan_cols))
print('dropping', len(drop), 'cols; sample:', drop[:6])
e011 = t9.drop(columns=drop)
print('e011 shape', e011.shape)

# --- robust-distribution features (as-of safe) ---
def make_feats(view, snapshot_day):
    s = snapshot_day
    hh = pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.day > s - 364]
    out = pd.DataFrame(index=hh)
    z = np.zeros((len(hh), 26))
    T = np.zeros((len(hh), 13))
    if len(tx):
        h = tx.household_key.values; d = tx.day.values; v = tx.sales_value.values
        tmp = pd.DataFrame({'h': h, 'w': (s - d) // 7, 'v': v})
        piv = tmp.pivot_table(index='h', columns='w', values='v', aggfunc='sum').reindex(hh).reindex(columns=range(26)).fillna(0.0)
        W = piv.values
        tmp2 = pd.DataFrame({'h': h, 'k': (s - d) // 28, 'v': v})
        piv2 = tmp2.pivot_table(index='h', columns='k', values='v', aggfunc='sum').reindex(hh).reindex(columns=range(13)).fillna(0.0)
        T = piv2.values
        r28 = tx[tx.day > s - 28]
        out['rb_max28'] = r28.groupby('household_key')['sales_value'].max().reindex(hh).fillna(0.0).values
        r84 = tx[tx.day > s - 84]
        nb = r84.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0).values
        s84 = r84.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
        out['rb_max28_ratio'] = out['rb_max28'] / (s84 / np.maximum(nb, 1) + 1.0)
        out['rb_spend3'] = tx[tx.day > s - 3].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
        out['rb_trips7'] = tx[tx.day > s - 7].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0).values
    out['rb_wmed26'] = np.median(W, axis=1)
    out['rb_wq25'] = np.percentile(W, 25, axis=1)
    out['rb_wq75'] = np.percentile(W, 75, axis=1)
    out['rb_wiqr'] = out['rb_wq75'] - out['rb_wq25']
    zz = (W == 0)
    out['rb_wzstreak'] = np.cumprod(zz, axis=1).sum(axis=1)
    out['rb_wnz4'] = (~zz[:, :4]).sum(axis=1)
    out['rb_tmed13'] = np.median(T, axis=1)
    out['rb_tq25'] = np.percentile(T, 25, axis=1)
    out['rb_tq75'] = np.percentile(T, 75, axis=1)
    out['rb_tmin13'] = T.min(axis=1)
    out['rb_tzero13'] = (T == 0).sum(axis=1)
    out['rb_tcv13'] = T.std(axis=1) / (T.mean(axis=1) + 1.0)
    return out

feats = agent_api.build_features(make_feats)
fcols = [c for c in feats.columns if c.startswith('rb_')]
print('rb feats:', len(fcols), 'nan%:', round(float(feats[fcols].isna().mean().mean()), 4))
print(feats[fcols].describe().loc[['mean','50%']].round(1).to_string())

feats['household_key'] = feats['household_key'].astype(e011.household_key.dtype)
feats['snapshot_day'] = feats['snapshot_day'].astype(e011.snapshot_day.dtype)
e012 = e011.merge(feats[['household_key','snapshot_day'] + fcols], on=['household_key','snapshot_day'], how='left')
print('e012 shape', e012.shape, 'rb nan in e012:', int(e012[fcols].isna().sum().sum()))
p1 = agent_api.save_table(e011, 'e011_pruned.parquet')
p2 = agent_api.save_table(e012, 'e012_robust.parquet')
print('saved:', p1, '|', p2)
