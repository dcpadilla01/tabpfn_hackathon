import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

snaps = agent_api.snapshot_days()['train'] + agent_api.snapshot_days()['validation']
print(snaps)

def build(view, s):
    hh = view.households
    tx = view.transactions
    tx = tx[tx.household_key.isin(hh)]
    out = pd.DataFrame(index=hh)
    # A: decay-weighted spend/trips last 56d
    t = tx[(tx.day > s-56)]
    g = t.groupby('household_key')
    dec = np.exp(-(s - t.day)/14.0)
    out['dec_spend14'] = (t.sales_value*dec).groupby(t.household_key).sum()
    dec7 = np.exp(-(s - t.day)/7.0)
    out['dec_spend7'] = (t.sales_value*dec7).groupby(t.household_key).sum()
    out['dec_trips'] = t.drop_duplicates('basket_id').assign(d=dec7.groupby(t.loc[t.drop_duplicates('basket_id').index,'household_key']).sum().reindex(t.drop_duplicates('basket_id').household_key).values).groupby('household_key').d.sum() if False else np.nan
    # simpler: recency-weighted trips
    bt = t.drop_duplicates('basket_id')[['household_key','day']]
    out['dec_trips'] = bt.assign(w=np.exp(-(s-bt.day)/14.0)).groupby('household_key').w.sum()
    # B: trip cadence last 112d
    t2 = tx[(tx.day > s-112)]
    b2 = t2.drop_duplicates('basket_id')[['household_key','day']].sort_values(['household_key','day'])
    b2['gap'] = b2.groupby('household_key').day.diff()
    gapstat = b2.groupby('household_key').gap.agg(['mean','std','median'])
    out['gap_mean'] = gapstat['mean']; out['gap_std'] = gapstat['std']; out['gap_med'] = gapstat['median']
    out['ntrip112'] = b2.groupby('household_key').size()
    # C: brand/deal traits last 112d
    prod = view.products
    m = t2.merge(prod[['product_id','brand']], on='product_id', how='left')
    sp = m.groupby('household_key').sales_value.sum()
    pl = m[m.brand=='Private'].groupby('household_key').sales_value.sum()
    out['pl_share'] = (pl/sp).fillna(0)
    out['disc_share2'] = (-(t2.coupon_disc+t2.coupon_match_disc+t2.retail_disc).groupby(t2.household_key).sum() / sp.clip(lower=0.01))
    # quantity and unit price
    out['qty112'] = t2.groupby('household_key').quantity.sum()
    out['unit_price'] = (sp / t2.groupby('household_key').quantity.sum().clip(lower=1))
    # D: macro trend: retailer-wide avg weekly spend per active hh, last 4w vs prior 52w
    allt = view.transactions
    w4 = allt[(allt.day > s-28)].sales_value.sum()/28.0
    w52 = allt[(allt.day > s-364)&(allt.day<=s-28)].sales_value.sum()/336.0
    out['macro_ratio'] = w4/(w52+1e-9)
    # E: interactions
    out['inter_spend_season'] = out.index.map(lambda k: np.nan)  # placeholder, add later in merge
    return out

frames=[]
for s in snaps:
    v = agent_api.snapshot(s)
    f = build(v, s)
    f['snapshot_day']=s
    frames.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(frames, ignore_index=True)
print(cand.shape)
print(cand.head())
agent_api.save_table(cand, 'cand1')
