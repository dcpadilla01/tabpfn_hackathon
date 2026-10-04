import numpy as np, pandas as pd, agent_api as api

def peers(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    def win(lo, hi):
        return spend[(spend.day > d-hi) & (spend.day <= d-lo)].groupby('household_key').sales_value.sum()
    r28 = win(0,28).reindex(hh).fillna(0)
    r56 = win(28,56).reindex(hh).fillna(0)
    r84 = win(56,84).reindex(hh).fillna(0)
    out = pd.DataFrame(index=hh)
    out['peer_recent28'] = r28.mean()
    out['peer_recent28_med'] = r28.median()
    out['peer_ratio'] = r28 / max(r56.mean(), 1e-9)
    out['peer_ratio2'] = r56 / max(r84.mean(), 1e-9)
    out['peer_p90'] = r28.quantile(.9)
    out['peer_p10'] = r28.quantile(.1)
    prior = d - 28
    sp = spend[(spend.day > prior-28) & (spend.day <= prior)]
    out['cohort_prior4w'] = sp.groupby('household_key').sales_value.sum().mean()
    ly = spend[(spend.day > d-363-28) & (spend.day <= d-336)]
    out['anchor_ly_pop'] = ly.sales_value.sum() if len(ly) else np.nan
    ly_hh = ly.groupby('household_key').sales_value.sum()
    out['spend_ly4w'] = ly_hh.reindex(hh).fillna(0)
    out['has_ly4w'] = ly_hh.reindex(hh).notna().astype(float)
    out['pop_ratio'] = r28.sum() / max(ly.sales_value.sum(), 1e-9) if len(ly) else np.nan
    return out.reset_index().rename(columns={'index':'household_key'})

tab3 = api.build_features(peers)
print('c3', tab3.shape)
e3 = api.load_saved('e003_catmix.parquet')
d3 = e3.merge(tab3, on=['household_key','snapshot_day'], how='left')
print('c3 merged', d3.shape, 'dups', d3.duplicated(['household_key','snapshot_day']).sum())
api.save_table(d3, 'e013_peers.parquet')
d2 = api.load_saved('e012_dorm.parquet')
print('c2 check', d2.shape, 'dups', d2.duplicated(['household_key','snapshot_day']).sum(), d2.snapshot_day.nunique())
print(d2[['gap_mean84','n_zero_l6','inactive_run84','active_week_share']].describe().round(3))