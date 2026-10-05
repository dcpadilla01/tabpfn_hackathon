import numpy as np, pandas as pd
from agent_api import build_features, save_table

def fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby('household_key')
    day = view.day

    def win(d0, d1):
        return tx[(tx.day >= day-d0) & (tx.day <= day-d1)]

    f = pd.DataFrame(index=hh)
    for name,(d0,d1) in {'w28':(28,1),'w84':(84,1),'w364':(364,1)}.items():
        t = win(d0,d1)
        f['spend_'+name] = t.groupby('household_key').sales_value.sum()
        f['baskets_'+name] = t.groupby('household_key').basket_id.nunique()
    t28 = win(28,1)
    f['dsl'] = day - g.day.max()
    f['max_basket_28'] = t28.groupby(['household_key','basket_id']).sales_value.sum().groupby('household_key').max()
    f['zero_wk_share_12'] = 1 - (win(84,1).groupby('household_key').week_no.nunique()/12.0)

    ct = view.table('campaign_targets')
    ct = ct[ct.household_key.isin(hh)]
    desc = ct.groupby('household_key').description.apply(lambda s: ' '.join(sorted(set(s))))
    for typ in ['TypeA','TypeB','TypeC']:
        f['camp_'+typ] = desc.str.contains(typ).astype(float)
    f['camp_n'] = ct.groupby('household_key').campaign.nunique()
    cr = view.table('coupon_redemptions')
    cr = cr[cr.household_key.isin(hh)]
    f['days_since_redem'] = day - cr.groupby('household_key').day.max()
    f['coup_trips_84'] = win(84,1).query('coupon_disc != 0').groupby('household_key').basket_id.nunique()

    f['s28_x_dsl'] = f['spend_w28'].fillna(0) * f['dsl'].fillna(999).clip(0,120)
    f['s28_x_campA'] = f['spend_w28'].fillna(0) * f['camp_TypeA']
    f['ew28'] = 0.0
    for d0,d1,hw in [(7,1,1.0),(14,8,0.5),(28,15,0.25),(56,29,0.125),(84,57,0.0625)]:
        f['ew28'] += hw * win(d0,d1).groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    f['ew28_x_dsl'] = f['ew28'] * f['dsl'].fillna(999).clip(0,120)

    wk = (day+8)//7
    dm = view.table('display_mailer')
    t84 = win(84,1)
    top_store = t84.groupby('household_key').store_id.agg(lambda s: s.value_counts().index[0]).reindex(hh)
    prods = t84.groupby('household_key').product_id.agg(set).reindex(hh).apply(lambda v: v if isinstance(v,set) else set())
    dm4 = dm[(dm.week_no > wk-4) & (dm.week_no <= wk)]
    dm4 = dm4[dm4.display.notna() | dm4.mailer.notna()]
    dmg = dm4.groupby('store_id').product_id.agg(set)
    f['disp_exposure'] = [len(prods.get(h,set()) & dmg.get(st,set())) if pd.notna(st) else 0 for h,st in top_store.items()]
    return f.reindex(hh)

tab = build_features(fn)
print(tab.shape)
fc = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
print(tab[fc].isna().mean().round(3).to_string())
p = save_table(tab, 'e018_tree_feats')
print(p)
