
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
