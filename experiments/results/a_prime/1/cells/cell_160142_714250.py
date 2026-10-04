import numpy as np, pandas as pd, agent_api as api

# --- candidate 1: demographics on top of E003 ---
e3 = api.load_saved('e003_catmix.parquet')
b = api.baseline_features()
demo_cols = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc','has_demographics']
d1 = e3.merge(b[['household_key','snapshot_day']+demo_cols], on=['household_key','snapshot_day'], how='left')
print('c1', d1.shape)
api.save_table(d1, 'e011_demo.parquet')

# --- candidate 2: dormancy / gap-structure features on top of E003 ---
def dorm(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    g = w.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # trip-day gaps per household
    trips = w.groupby(['household_key','day']).size().reset_index()[['household_key','day']]
    trips = trips.sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key').day.diff()
    gg = trips.groupby('household_key').gap
    out['gap_mean84'] = gg.mean()
    out['gap_max84'] = gg.max()
    out['gap_std84'] = gg.std()
    out['gap_last'] = trips.groupby('household_key').gap.last()
    # zero 28d windows among l1..l6
    sp = tx.groupby('household_key').day.apply(lambda s: None)  # placeholder
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    def win_zero(lo, hi):
        s = spend[(spend.day > d-hi) & (spend.day <= d-lo+0)].groupby('household_key').sales_value.sum()
        return s
    z = 0; cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        s = spend[(spend.day > d-28*(k+1)) & (spend.day <= d-28*k)].groupby('household_key').sales_value.sum()
        cnt = cnt + (s.reindex(hh).fillna(0) <= 0).astype(float)
    out['n_zero_l6'] = cnt
    # longest inactive run in last 84d (max gap between consecutive trip days, incl. edges)
    first = trips.groupby('household_key').day.min(); last = trips.groupby('household_key').day.max()
    out['edge_gap84'] = (last - first).where(first.notna())
    out['inactive_run84'] = np.maximum(out['gap_max84'].fillna(84), (d - last).where(last.notna(), 84))
    # full-tenure zero-window fraction
    allsp = tx.groupby('household_key').sales_value.sum()
    ten = tx.groupby('household_key').day.agg(['min','max'])
    out['tenure'] = (d - ten['min']).clip(lower=1)
    out['zero_frac_full'] = cnt / 6.0  # cheap proxy; refine below with full history
    # full-history 28d windows with zero spend
    fh = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    nw = (out['tenure'] // 28).clip(lower=1)
    # count zero windows over full tenure via weekly activity
    wk_act = tx.assign(wk=(tx.day+8)//7).groupby(['household_key','wk']).size()
    n_act_weeks = wk_act.groupby('household_key').size()
    n_weeks_ten = ((d - ten['min']) // 7).clip(lower=1)
    out['active_week_share'] = (n_act_weeks / n_weeks_ten).clip(0,1)
    return out

tab2 = api.build_features(lambda view, sd: dorm(view, sd).reset_index().rename(columns={'index':'household_key'}))
print('c2 built', tab2.shape, tab2.snapshot_day.nunique())
d2 = e3.merge(tab2.drop(columns=['snapshot_day']), on='household_key', how='left')
print('c2', d2.shape, list(d2.columns[-10:]))
api.save_table(d2, 'e012_dorm.parquet')