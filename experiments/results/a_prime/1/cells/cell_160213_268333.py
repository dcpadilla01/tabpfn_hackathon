import numpy as np, pandas as pd, agent_api as api

def dorm(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    trips = w.groupby(['household_key','day']).size().reset_index()[['household_key','day']].sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key').day.diff()
    gg = trips.groupby('household_key').gap
    out = pd.DataFrame(index=hh)
    out['gap_mean84'] = gg.mean()
    out['gap_max84'] = gg.max()
    out['gap_std84'] = gg.std()
    out['gap_last'] = gg.last()
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        s = spend[(spend.day > d-28*(k+1)) & (spend.day <= d-28*k)].groupby('household_key').sales_value.sum()
        cnt = cnt + (s.reindex(hh).fillna(0) <= 0).astype(float)
    out['n_zero_l6'] = cnt
    last = trips.groupby('household_key').day.max()
    out['inactive_run84'] = np.maximum(out['gap_max84'].fillna(84), (d - last).where(last.notna(), 84))
    ten = tx.groupby('household_key').day.min()
    out['tenure'] = (d - ten).clip(lower=1)
    wk_act = tx.assign(wk=(tx.day+8)//7).groupby(['household_key','wk']).size()
    n_act_weeks = wk_act.groupby('household_key').size()
    n_weeks_ten = ((d - ten) // 7).clip(lower=1)
    out['active_week_share'] = (n_act_weeks / n_weeks_ten).clip(0,1)
    out['zero_frac_full'] = 1 - out['active_week_share']
    return out.reset_index().rename(columns={'index':'household_key'})

tab2 = api.build_features(dorm)
e3 = api.load_saved('e003_catmix.parquet')
d2 = e3.merge(tab2, on=['household_key','snapshot_day'], how='left')
print('c2', d2.shape, d2.household_key.nunique())
assert d2.household_key.duplicated(['household_key','snapshot_day']).sum()==0
api.save_table(d2, 'e012_dorm.parquet')

# --- candidate 3: peer anchors ---
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
    # prior-snapshot cohort mean (train snapshots are every 28d)
    prior = d - 28
    sp = spend[(spend.day > prior-28) & (spend.day <= prior)]
    out['cohort_prior4w'] = sp.groupby('household_key').sales_value.sum().mean()
    # seasonal anchor: same 4w window one year earlier (population level)
    ly = spend[(spend.day > d-363-28) & (spend.day <= d-336)]
    out['anchor_ly_pop'] = ly.sales_value.sum() if len(ly) else np.nan
    # per-household last-year same-window spend
    ly_hh = ly.groupby('household_key').sales_value.sum()
    out['spend_ly4w'] = ly_hh.reindex(hh).fillna(0)
    out['has_ly4w'] = ly_hh.reindex(hh).notna().astype(float)
    # population growth factor recent vs ly
    out['pop_ratio'] = r28.sum() / max(ly.sales_value.sum(), 1e-9) if len(ly) else np.nan
    return out.reset_index().rename(columns={'index':'household_key'})

tab3 = api.build_features(peers)
print('c3', tab3.shape)
d3 = e3.merge(tab3, on=['household_key','snapshot_day'], how='left')
print('c3 merged', d3.shape)
api.save_table(d3, 'e013_peers.parquet')
print(tab3.head())