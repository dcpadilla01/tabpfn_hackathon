import agent_api
import pandas as pd, numpy as np

e5 = agent_api.load_saved('e005_trend_season.parquet')

def feats(view, sd):
    hh = pd.Index(view.households)
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    gd = tx.groupby(['household_key','day'], as_index=False)['sales_value'].sum()
    out = pd.DataFrame(index=hh)
    age = (sd - gd['day']).astype(float)
    for hl in (7, 224):
        w = gd['sales_value'] * np.power(0.5, age / hl)
        out[f'z_dec_{hl}'] = w.groupby(gd['household_key']).sum().reindex(hh).fillna(0.0)
    for wnd in (14, 112, 182):
        m = gd[gd['day'] > sd - wnd].groupby('household_key').size()
        out[f'z_trips_{wnd}'] = m.reindex(hh).fillna(0.0)
    gwk = gd.assign(wk=((gd['day'] + 8) // 7).astype(int))
    cur_wk = (sd + 8) // 7
    for nw in (4, 26, 52):
        m = gwk[gwk['wk'] > cur_wk - nw].groupby('household_key')['wk'].nunique()
        out[f'z_active_wk_{nw}'] = m.reindex(hh).fillna(0.0)
    t112 = gd[gd['day'] > sd - 112].sort_values(['household_key','day'])
    g = t112.groupby('household_key')['day'].apply(lambda d: np.diff(np.sort(d.values)))
    gm = g.apply(lambda a: a.mean() if len(a) else np.nan)
    gx = g.apply(lambda a: a.max() if len(a) else np.nan)
    gs = g.apply(lambda a: a.std() if len(a) else np.nan)
    out['z_gap_mean'] = gm.reindex(hh)
    out['z_gap_max'] = gx.reindex(hh)
    out['z_gap_std'] = gs.reindex(hh)
    b364 = gd[gd['day'] > sd - 364].copy()
    b364['b'] = ((sd - b364['day']) // 28).astype(int)
    zb = b364.groupby('household_key')['b'].nunique() / 13.0
    out['z_zero_block_share'] = (1.0 - zb).reindex(hh).fillna(1.0)
    last = t112.groupby('household_key')['day'].max()
    ld = last.rename('ld')
    tmp = gd.merge(ld.reset_index(), on='household_key')
    last_basket = tmp[tmp['day'] == tmp['ld']].groupby('household_key')['sales_value'].sum()
    out['z_last_basket'] = last_basket.reindex(hh)
    dsl = (sd - last).reindex(hh)  # days since last trip
    out['z_recency_norm'] = (dsl / (out['z_gap_mean'].fillna(14) + 1.0)).clip(0, 10)
    rate84 = gd[gd['day'] > sd - 84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0) / 84.0
    out['z_rate84'] = rate84
    nt28 = (gd[gd['day'] > sd - 28].groupby('household_key').size().reindex(hh).fillna(0.0) == 0).astype(float)
    out['z_notrip28_x_rate'] = nt28 * rate84
    out['z_recency_x_rate'] = (dsl.fillna(112).clip(0,112) * rate84)
    out['z_dec7_x_act4'] = out['z_dec_7'] * (out['z_active_wk_4'] / 4.0)
    return out

new = agent_api.build_features(feats)
print('new', new.shape)
print(new.describe().T[['mean','std','min','max']])

m = e5.merge(new.reset_index().rename(columns={'index':'household_key'}), on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape, 'dups:', m.duplicated(['household_key','snapshot_day']).sum())
assert m.shape[0] == e5.shape[0]
p = agent_api.save_table(m, 'e006_zero_inflation')
print(p)
