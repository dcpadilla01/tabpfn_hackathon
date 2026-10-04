import numpy as np, pandas as pd
from agent_api import load_saved, build_features, save_table, snapshot_days

base = load_saved('hist_v1.parquet')
print('base', base.shape)
print(base.columns.tolist())
print(snapshot_days())

def feats(view, day):
    tx = view.transactions
    hh = pd.Index(view.households, name='household_key')
    tx = tx[tx.household_key.isin(hh)]
    def win(lo, hi=None):
        m = tx.day >= day - lo + 1
        if hi is not None:
            m &= tx.day <= day - hi
        d = tx[m]
        return d.groupby('household_key').agg(spend=('sales_value','sum'),
                                              nb=('basket_id','nunique'),
                                              qty=('quantity','sum'))
    wins = {'4':(28,None), '8':(56,None), '12':(84,None), '24':(168,None), '52':(364,None),
            'p1':(56,28), 'p2':(84,56)}
    W = {k: win(lo,hi) for k,(lo,hi) in wins.items()}
    f = pd.DataFrame(index=hh)
    for k,a in W.items():
        f['x_spend_'+k] = a['spend'].reindex(hh).fillna(0)
        f['x_nb_'+k]    = a['nb'].reindex(hh).fillna(0)
    f['x_qty_4'] = W['4']['qty'].reindex(hh).fillna(0)
    f['x_bv_4']  = (f['x_spend_4']/f['x_nb_4'].replace(0,np.nan)).fillna(0)
    g = tx.groupby('household_key')
    f['x_recency'] = (day - g.day.max().reindex(hh)).fillna(0)
    f['x_tenure']  = (day - g.day.min().reindex(hh) + 1).fillna(1)
    tot = g.sales_value.sum().reindex(hh).fillna(0)
    f['x_exp4w'] = tot / f['x_tenure'].clip(lower=1) * 4          # long-run avg spend per 4w
    f['x_r_4_8']   = f['x_spend_4'] / (f['x_spend_8']/2 + 1)
    f['x_r_8_24']  = f['x_spend_8'] / (f['x_spend_24']/3 + 1)
    f['x_r_p1_p2'] = f['x_spend_p1'] / (f['x_spend_p2'] + 1)
    d84 = tx[tx.day >= day-83]
    f['x_n_stores'] = d84.groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    f['x_n_prods']  = d84.groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    disc = d84.assign(dd=d84.retail_disc.fillna(0)+d84.coupon_disc.fillna(0)+d84.coupon_match_disc.fillna(0))
    dd = disc.groupby('household_key').agg(dsum=('dd','sum'), ssum=('sales_value','sum')).reindex(hh).fillna(0)
    f['x_disc_rate'] = dd.dsum/(dd.ssum+dd.dsum+1)
    prods = view.products[['product_id','department']]
    m = d84.merge(prods, on='product_id', how='left')
    m['department'] = m['department'].fillna('UNK')
    pv = m.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum').fillna(0)
    top = pv.sum().sort_values(ascending=False).head(9).index
    for c in top:
        f['x_dep_'+str(c)] = pv[c].reindex(hh).fillna(0)
    rest = [c for c in pv.columns if c not in top]
    f['x_dep_other'] = pv[rest].sum(axis=1).reindex(hh).fillna(0) if rest else 0.0
    cr = view.coupon_redemptions
    cr = cr[cr.household_key.isin(hh) & (cr.day >= day-83)]
    f['x_n_coupons'] = cr.groupby('household_key').day.count().reindex(hh).fillna(0)
    return f

tab = build_features(feats)
print('built', tab.shape, tab.columns.tolist()[:8])
merged = base.merge(tab, on=['household_key','snapshot_day'], how='left')
print('merged', merged.shape, 'nan cols:', merged.isna().any().sum())
path = save_table(merged, 'hist_v2.parquet')
print(path)
