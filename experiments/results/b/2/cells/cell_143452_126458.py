import numpy as np, pandas as pd
from agent_api import build_features, save_table

TOP_DEPTS = ['GROCERY','DRUG GM','PRODUCE','MEAT-PCKGD','MEAT','DELI','PASTRY',
             'NUTRITION','KIOSK-GAS','SEAFOOD-PCKGD']
SHORT = {'GROCERY':'gro','DRUG GM':'drug','PRODUCE':'prod','MEAT-PCKGD':'meatp','MEAT':'meat',
         'DELI':'deli','PASTRY':'pastr','NUTRITION':'nutri','KIOSK-GAS':'gas','SEAFOOD-PCKGD':'sea'}

def make_features(view, sd):
    hh = view.households          # Index of household_key
    tx = view.transactions
    tx = tx[tx.day > sd - 112]
    prod = view.products[['product_id','department','brand']]
    t = tx.merge(prod, on='product_id', how='left')
    t['dept'] = t.department.astype(str).fillna('OTHER')
    t['is_priv'] = (t.brand.astype(str) == 'Private').astype(float)
    t['disc'] = -(t.retail_disc + t.coupon_disc + t.coupon_match_disc)
    t['evening'] = (t.trans_time >= 1700).astype(float)

    b = t.groupby('basket_id').agg(hh=('household_key','first'), val=('sales_value','sum'),
                                   day=('day','first'), store=('store_id','first'))
    out = pd.DataFrame(index=hh)
    g = t.groupby('household_key')
    out['spend_112'] = g.sales_value.sum().reindex(out.index).fillna(0)
    t84 = t[t.day > sd-84]
    g84 = t84.groupby('household_key')
    s84 = g84.sales_value.sum().reindex(out.index).fillna(0)
    out['spend_84'] = s84

    gs = t84.groupby(['household_key','dept']).sales_value.sum()
    sh = (gs / s84).unstack()
    for d in TOP_DEPTS:
        col = sh[d] if d in sh.columns else pd.Series(0.0, index=sh.index)
        out['sh_'+SHORT[d]] = col.reindex(out.index).fillna(0)
    other = [c for c in sh.columns if c not in TOP_DEPTS]
    out['sh_other'] = sh[other].sum(axis=1).reindex(out.index).fillna(0) if other else 0.0
    for d in ['GROCERY','PRODUCE','MEAT-PCKGD','DELI','DRUG GM']:
        col = gs.xs(d, level=1) if d in sh.columns else pd.Series(0.0, index=sh.index)
        out['sp_'+SHORT[d]] = col.reindex(out.index).fillna(0)

    out['priv_share'] = t84.assign(sv=t84.sales_value*t84.is_priv).groupby('household_key').sv.sum().reindex(out.index).fillna(0)/s84.replace(0,np.nan)
    out['disc_share'] = (g84.disc.sum()/(g84.sales_value.sum()+g84.disc.sum())).reindex(out.index).fillna(0)
    out['n_stores'] = g84.store_id.nunique().reindex(out.index).fillna(0)
    ms = b.groupby('hh').store.agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
    out['main_store_share'] = ms.reindex(out.index)
    out['n_prods'] = g84.product_id.nunique().reindex(out.index).fillna(0)
    out['rep_ratio'] = (g84.size()/g84.product_id.nunique()).reindex(out.index).fillna(1)

    b84 = b[b.day > sd-84]
    gb = b84.groupby('hh')
    out['trips_84'] = gb.size().reindex(out.index).fillna(0)
    out['basket_med'] = gb.val.median().reindex(out.index)
    out['basket_max'] = gb.val.max().reindex(out.index)
    out['basket_std'] = gb.val.std().reindex(out.index)
    out['evening_share'] = t84.assign(ev=t84.evening).groupby('household_key').ev.mean().reindex(out.index)

    a = t[t.day > sd-56]; bb = t[t.day <= sd-56]
    sa = a.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(columns=TOP_DEPTS).fillna(0)
    sb = bb.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(columns=TOP_DEPTS).fillna(0)
    sa = sa.div(sa.sum(axis=1).replace(0,np.nan), axis=0); sb = sb.div(sb.sum(axis=1).replace(0,np.nan), axis=0)
    out['mix_shift'] = (sa-sb).abs().sum(axis=1).reindex(out.index).fillna(0)
    return out

df = build_features(make_features)
print(df.shape, df.columns.tolist())
print(df.isna().sum().sort_values(ascending=False).head(8))
path = save_table(df, 'e003_product_mix')
print(path)
