import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, KEYS, TARGET, snapshot_days

e = load_saved('e001_recent_spend.parquet')
print(e.shape)
print(e.columns.tolist())
tt = train_targets()
print(tt[TARGET].describe())
print('zero frac:', (tt[TARGET]==0).mean())
m = tt.merge(e, on=KEYS, how='left')
print(m.shape, 'missing:', m[e.columns.tolist()[2:]].isna().sum().sum())
num = m.select_dtypes(include=[np.number])
corr = num.corr()[TARGET].drop(TARGET).sort_values()
print(corr)
print(snapshot_days())


# ---- cell ----
from agent_api import snapshot
v = snapshot()
print([a for a in dir(v) if not a.startswith('_')])
print(v.products.columns.tolist() if hasattr(v,'products') else 'no products')
print(v.transactions.shape)


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import snapshot
v = snapshot()
tx = v.transactions
print('tx days', tx.day.min(), tx.day.max())
# spend by department overall
p = v.products
t = tx.merge(p[['product_id','department','brand']], on='product_id', how='left')
print(t.department.value_counts().head(15))
print('null dept frac', t.department.isna().mean())
# discount cols
print(tx[['coupon_match_disc','coupon_disc','retail_disc','quantity','sales_value']].describe())
# per-household dept shares
g = t.groupby(['household_key','department']).sales_value.sum()
hh = t.groupby('household_key').sales_value.sum()
sh = (g/hh).unstack()
print(sh.shape)
print(sh.head())


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import build_features, save_table, snapshot

TOP_DEPTS = ['GROCERY','DRUG GM','PRODUCE','MEAT-PCKGD','MEAT','DELI','PASTRY',
             'NUTRITION','KIOSK-GAS','SEAFOOD-PCKGD']

def make_features(view, sd):
    hh = view.households
    tx = view.transactions
    tx = tx[tx.day > sd - 112]  # trailing 112 days
    prod = view.products[['product_id','department','brand']]
    t = tx.merge(prod, on='product_id', how='left')
    t['dept'] = t.department.fillna('OTHER')
    t['is_priv'] = (t.brand == 'Private').astype(float)
    t['disc'] = -(t.retail_disc + t.coupon_disc + t.coupon_match_disc)
    t['weekend'] = ((t.day % 7) >= 5).astype(float)   # rough weekend flag
    t['evening'] = (t.trans_time >= 1700).astype(float)

    # basket-level aggregates
    b = t.groupby('basket_id').agg(hh=('household_key','first'), val=('sales_value','sum'),
                                   day=('day','first'), store=('store_id','first'))
    g = t.groupby('household_key')
    out = pd.DataFrame(index=hh.index)
    tot112 = g.sales_value.sum()
    out['spend_112'] = tot112
    tot84 = t[t.day > sd-84].groupby('household_key').sales_value.sum()
    out['spend_84'] = tot84.reindex(out.index).fillna(0)

    # dept shares & spend (84d)
    t84 = t[t.day > sd-84]
    gs = t84.groupby(['household_key','dept']).sales_value.sum()
    sh = (gs / t84.groupby('household_key').sales_value.sum()).unstack()
    for d in TOP_DEPTS:
        out[f'sh_{d[:6].replace(" ","")}'] = sh[d].reindex(out.index).fillna(0) if d in sh.columns else 0.0
    other = [c for c in sh.columns if c not in TOP_DEPTS]
    out['sh_other'] = sh[other].sum(axis=1).reindex(out.index).fillna(0) if other else 0.0
    for d in ['GROCERY','PRODUCE','MEAT-PCKGD','DELI','DRUG GM']:
        out[f'sp_{d[:6].replace(" ","")}'] = (gs.xs(d, level=1) if d in sh.columns else 0).reindex(out.index).fillna(0)

    # brand / promo / channel mix (84d)
    g84 = t84.groupby('household_key')
    s84 = g84.sales_value.sum().reindex(out.index).fillna(0)
    out['priv_share'] = (g84.apply(lambda x: (x.sales_value*x.is_priv).sum(), include_groups=False)
                         if False else t84.assign(sv=t84.sales_value*t84.is_priv).groupby('household_key').sv.sum()).reindex(out.index).fillna(0)/s84.replace(0,np.nan)
    out['disc_share'] = (g84.disc.sum()/ (g84.sales_value.sum()+g84.disc.sum())).reindex(out.index).fillna(0)
    out['n_stores'] = g84.store_id.nunique().reindex(out.index).fillna(0)
    ms = b.groupby('hh').store.agg(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
    out['main_store_share'] = ms.reindex(out.index)
    out['n_prods'] = g84.product_id.nunique().reindex(out.index).fillna(0)
    out['rep_ratio'] = (g84.size()/g84.product_id.nunique()).reindex(out.index).fillna(1)

    # basket shape (84d)
    b84 = b[b.day > sd-84]
    gb = b84.groupby('hh')
    out['trips_84'] = gb.size().reindex(out.index).fillna(0)
    out['basket_med'] = gb.val.median().reindex(out.index)
    out['basket_max'] = gb.val.max().reindex(out.index)
    out['basket_std'] = gb.val.std().reindex(out.index)
    out['weekend_share'] = gb.day.apply(lambda s: ((s%7)>=5).mean()).reindex(out.index)
    out['evening_share'] = t84.assign(ev=t84.evening).groupby('household_key').ev.mean().reindex(out.index)

    # dept share stability: corr between 56d and prior 56d shares for GROCERY
    a = t[(t.day > sd-56)]; bb = t[(t.day <= sd-56) & (t.day > sd-112)]
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


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import build_features, save_table

TOP_DEPTS = ['GROCERY','DRUG GM','PRODUCE','MEAT-PCKGD','MEAT','DELI','PASTRY',
             'NUTRITION','KIOSK-GAS','SEAFOOD-PCKGD']
SHORT = {'GROCERY':'gro','DRUG GM':'drug','PRODUCE':'prod','MEAT-PCKGD':'meatp','MEAT':'meat',
         'DELI':'deli','PASTRY':'pastr','NUTRITION':'nutri','KIOSK-GAS':'gas','SEAFOOD-PCKGD':'sea'}

def make_features(view, sd):
    hh = view.households
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
    out = pd.DataFrame(index=hh.index)
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

    a = t[t.day > sd-56]; bb = t[(t.day <= sd-56)]
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


# ---- cell ----
from agent_api import snapshot
v = snapshot()
print(type(v.households))
print(v.households[:5] if hasattr(v.households,'__getitem__') else v.households.head())


# ---- cell ----
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
