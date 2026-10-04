
import agent_api, pandas as pd, numpy as np
e1 = agent_api.load_saved('e001_history.parquet')
print(e1.shape)
print(list(e1.columns))
print(e1.head(3))
print(agent_api.snapshot_days())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e1 = agent_api.load_saved('e001_history.parquet')

def fn(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    df = pd.DataFrame(index=hh)

    # --- campaign targeting (view.campaigns already filtered to started campaigns) ---
    ct = view.campaign_targets
    camp = view.campaigns
    cmap = camp.set_index('campaign')['description']
    ct2 = ct.copy()
    ct2['desc'] = ct2['campaign'].map(cmap)
    g = ct2.groupby('household_key')['desc']
    n_t = g.count()
    df['n_campaigns'] = n_t.reindex(hh).fillna(0)
    for t in ['TypeA','TypeB','TypeC']:
        cnt = ct2[ct2['desc']==t].groupby('household_key').size()
        df['targ_'+t] = cnt.reindex(hh).fillna(0)

    # --- coupon redemptions ---
    cr = view.coupon_redemptions
    cr = cr[cr['day'] <= snapshot_day]
    for w in [28, 56, 84, 365]:
        c = cr[cr['day'] > snapshot_day - w].groupby('household_key').size()
        df['cred_'+str(w)] = c.reindex(hh).fillna(0)
    last = cr.groupby('household_key')['day'].max()
    df['days_since_coupon'] = (snapshot_day - last).reindex(hh).fillna(9999)
    ncr = cr.groupby('household_key').size()
    df['cred_all'] = ncr.reindex(hh).fillna(0)
    df['ever_redeemed'] = (df['cred_all'] > 0).astype(int)

    # --- merge E001 features for this snapshot ---
    base = e1[e1['snapshot_day'] == snapshot_day].set_index('household_key')
    out = df.join(base.drop(columns=['household_key','snapshot_day'], errors='ignore'), how='left')
    return out

tab = agent_api.build_features(fn)
print(tab.shape, tab['snapshot_day'].nunique())
# sanity: should match e1 exactly on e1 columns
m = tab.merge(e1, on=['household_key','snapshot_day'], suffixes=('','_e1'))
diffs = {c: (m[c].fillna(-9e9) - m[c+'_e1'].fillna(-9e9)).abs().max() for c in ['spend_28','spend_365','nb_28','days_since_last']}
print(diffs)
print(tab[['n_campaigns','cred_28','cred_all','ever_redeemed','days_since_coupon']].describe())
path = agent_api.save_table(tab, 'e002_marketing.parquet')
print(path)
