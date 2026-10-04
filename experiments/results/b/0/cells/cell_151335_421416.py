import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def build(view, s):
    hh = view.households
    hhset = set(hh)
    tx = view.transactions
    tx = tx[tx.household_key.isin(hhset)]
    out = pd.DataFrame(index=hh)
    # A: decay-weighted spend last 56d
    t = tx[tx.day > s-56]
    dec = np.exp(-(s - t.day)/14.0)
    out['dec_spend14'] = (t.sales_value*dec).groupby(t.household_key).sum()
    dec7 = np.exp(-(s - t.day)/7.0)
    out['dec_spend7'] = (t.sales_value*dec7).groupby(t.household_key).sum()
    bt = t.drop_duplicates('basket_id')[['household_key','day']]
    out['dec_trips'] = bt.assign(w=np.exp(-(s-bt.day)/14.0)).groupby('household_key').w.sum()
    # B: trip cadence last 112d
    t2 = tx[tx.day > s-112]
    b2 = t2.drop_duplicates('basket_id')[['household_key','day']].sort_values(['household_key','day'])
    b2['gap'] = b2.groupby('household_key').day.diff()
    gs = b2.groupby('household_key').gap.agg(['mean','std','median'])
    out['gap_mean']=gs['mean']; out['gap_std']=gs['std']; out['gap_med']=gs['median']
    out['ntrip112'] = b2.groupby('household_key').size()
    # C: brand/deal traits last 112d
    prod = view.products[['product_id','brand']]
    m = t2.merge(prod, on='product_id', how='left')
    sp = m.groupby('household_key').sales_value.sum()
    pl = m[m.brand=='Private'].groupby('household_key').sales_value.sum()
    out['pl_share'] = (pl/sp).fillna(0)
    out['disc_share2'] = (-(t2.coupon_disc+t2.coupon_match_disc+t2.retail_disc).groupby(t2.household_key).sum()/sp.clip(lower=0.01))
    out['qty112'] = t2.groupby('household_key').quantity.sum()
    out['unit_price'] = sp/t2.groupby('household_key').quantity.sum().clip(lower=1)
    # D: macro trend
    allt = view.transactions
    w4 = allt[allt.day > s-28].sales_value.sum()/28.0
    w52 = allt[(allt.day > s-364)&(allt.day<=s-28)].sales_value.sum()/336.0
    out['macro_ratio'] = w4/(w52+1e-9)
    out['snapshot_day']=s
    return out.reset_index().rename(columns={'index':'household_key'})

cand = agent_api.build_features(build)
print(cand.shape)
print(cand.head(3))
agent_api.save_table(cand, 'cand1')
