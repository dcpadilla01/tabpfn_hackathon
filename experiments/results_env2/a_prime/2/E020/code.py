
import pandas as pd, numpy as np
e019 = agent_api.load_saved('e019_e017_plus_marketing.parquet')
print('e019 shape', e019.shape)
cols19 = [c for c in e019.columns if c not in ('household_key','snapshot_day')]
print('e019 n_feat', len(cols19))
e017 = agent_api.load_saved('e017_xsec_rank.parquet')
cols17 = [c for c in e017.columns if c not in ('household_key','snapshot_day')]
print('e017 n_feat', len(cols17))
adds = [c for c in cols19 if c not in cols17]
print('e019 adds vs e017:', adds)
e018 = agent_api.load_saved('e018_tree_feats.parquet')
cols18 = [c for c in e018.columns if c not in ('household_key','snapshot_day')]
print('e018 feats:', cols18)
print('e018 feats NOT in e019:', [c for c in cols18 if c not in cols19])
print('dtypes of adds:', {c: str(e019[c].dtype) for c in cols19 if e019[c].dtype == object})
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero share train:', (tt.future_spend_4w == 0).mean())
m = e019.merge(tt, on=['household_key', 'snapshot_day'])
print('merged', m.shape)
nan_share = m[cols19].isna().mean().sort_values(ascending=False)
print('worst NaN coverage:'); print(nan_share.head(12))
corr = m[cols19].corrwith(m['future_spend_4w']).abs().sort_values(ascending=False)
print('top |corr| with target:'); print(corr.head(25))
print('snapshot_days:', agent_api.snapshot_days())


# ---- cell ----

import pandas as pd, numpy as np
e017 = agent_api.load_saved('e017_xsec_rank.parquet')
e018 = agent_api.load_saved('e018_tree_feats.parquet')
cols17 = set(e017.columns) - {'household_key','snapshot_day'}
cols18 = set(e018.columns) - {'household_key','snapshot_day'}
missing = sorted(cols18 - cols17)
print('E018 feats missing from E017:', missing)
# check near-duplicates by name similarity
import re
for m in missing:
    stem = re.sub(r'(w\d+|28|84|364)$', '', m)
    sim = [c for c in cols17 if stem and stem in c]
    print(m, '-> similar in E017:', sim[:8])


# ---- cell ----

import pandas as pd, numpy as np

def new_feats(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)
    t28 = tx[tx.day > d - 28]
    t84 = tx[tx.day > d - 84]
    t364 = tx[tx.day > d - 364]

    # --- store loyalty ---
    g84 = t84.groupby('household_key')
    spend84 = g84.sales_value.sum()
    nb84 = g84.basket_id.nunique()
    out['x_n_stores_84'] = g84.store_id.nunique().reindex(hh)
    s_store = t84.groupby(['household_key','store_id']).sales_value.sum()
    tot = s_store.groupby(level=0).sum().replace(0, np.nan)
    out['x_modal_store_share_84'] = (s_store.groupby(level=0).max() / tot).reindex(hh)
    out['x_n_stores_364'] = t364.groupby('household_key').store_id.nunique().reindex(hh)

    # --- price tier ---
    units84 = g84.quantity.sum()
    ppu84 = spend84 / units84.replace(0, np.nan)
    out['x_ppu_84'] = ppu84.reindex(hh)
    g364 = t364.groupby('household_key')
    ppu364 = g364.sales_value.sum() / g364.quantity.sum().replace(0, np.nan)
    out['x_ppu_ratio_84_364'] = (ppu84 / ppu364).reindex(hh)

    # --- basket trajectory ---
    g28 = t28.groupby('household_key')
    spend28 = g28.sales_value.sum()
    nb28 = g28.basket_id.nunique()
    ab28 = spend28 / nb28.replace(0, np.nan)
    ab84 = spend84 / nb84.replace(0, np.nan)
    out['x_avg_basket_28'] = ab28.reindex(hh)
    out['x_basket_28_vs_84'] = (ab28 / ab84).reindex(hh)
    out['x_units_per_basket_84'] = (units84 / nb84.replace(0, np.nan)).reindex(hh)

    # --- timing patterns (84d, basket level) ---
    bb = t84.groupby(['household_key','basket_id']).agg(tt=('trans_time','max'), day=('day','max'))
    out['x_evening_share_84'] = (bb.tt >= 1700).groupby(level=0).mean().reindex(hh)
    ddf = pd.DataFrame({'hh': bb.index.get_level_values(0), 'dow': (bb.day.values % 7)})
    cnt = ddf.groupby(['hh','dow']).size()
    totc = cnt.groupby(level=0).sum().replace(0, np.nan)
    out['x_dow_conc_84'] = (cnt.groupby(level=0).max() / totc).reindex(hh)

    # --- coupon recency ---
    cr = view.table('coupon_redemptions')
    last = cr.groupby('household_key').day.max()
    out['x_days_since_redem'] = (d - last).reindex(hh)
    out['x_coup_trips_84'] = cr[cr.day > d - 84].groupby('household_key').day.nunique().reindex(hh)

    # --- lifecycle / activity breadth ---
    first = tx.groupby('household_key').day.min()
    out['x_tenure_days'] = (d - first).reindex(hh)
    wk = t84.assign(w=(t84.day + 8) // 7)
    out['x_active_wk_share_84'] = (wk.groupby('household_key').w.nunique() / 12.0).reindex(hh)
    return out

nf = agent_api.build_features(new_feats)
print('new feats table:', nf.shape)
print(nf.drop(columns=['household_key','snapshot_day']).describe().T[['count','mean','std','min','max']])

e019 = agent_api.load_saved('e019_e017_plus_marketing.parquet')
m = e019.merge(nf, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape, 'expected (36426,', e019.shape[1]-2+nf.shape[1]-2, '+2)')
dupcols = [c for c in m.columns if c.endswith('_x') or c.endswith('_y')]
print('dup-suffix cols:', dupcols)
tt = agent_api.train_targets()
mm = m.merge(tt, on=['household_key','snapshot_day'])
cor = mm[[c for c in m.columns if c.startswith('x_')]].corrwith(mm.future_spend_4w)
print('corr of new feats with target:'); print(cor.sort_values(key=abs, ascending=False))
path = agent_api.save_table(m, 'e020_final_dims.parquet')
print('saved:', path)


# ---- cell ----

import pandas as pd
m = agent_api.load_saved('e020_final_dims.parquet')
print('rows', len(m), 'cols', m.shape[1])
print('has index col:', 'index' in m.columns)
print('object cols:', [c for c in m.columns if m[c].dtype == object])
print('all-NaN cols:', [c for c in m.columns if m[c].isna().all()])
print('snapshot days:', sorted(m.snapshot_day.unique()))
print('n households:', m.household_key.nunique())
