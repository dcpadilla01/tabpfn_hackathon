import agent_api as A
for name in ['e006_zero_inflation.parquet','e006_newblock.parquet']:
    df = A.load_saved(name)
    print(name, df.shape)
    print(sorted(df.columns)[:40])
    print('...')
    print(sorted(df.columns)[40:])
    print()


# ---- cell ----
import agent_api as A
snap = A.snapshot()
tx = snap.transactions
print('stores:', tx.store_id.nunique(), 'products:', tx.product_id.nunique(), 'hh:', tx.household_key.nunique())
print(tx[['sales_value','quantity']].describe())
print('coupon cols sum:', tx[['coupon_match_disc','coupon_disc','retail_disc']].sum())
# campaign types
print(snap.campaign_targets.description.value_counts())
print(snap.campaigns.head())
print('n campaigns started by 459:', len(snap.campaigns))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
snap = A.snapshot()
tx = snap.transactions
# store diversity: how many stores per household in trailing 84d
g = tx[tx.day > 459-84].groupby('household_key')
stores_hh = g.store_id.nunique()
print('stores per hh (84d):', stores_hh.describe())
print('top_store_share_84 already in E006')
# check quantity vs sales correlation, and big-quantity rows
print('corr qty-sales:', tx.quantity.corr(tx.sales_value))
# zero/negative sales lines
print('zero sales lines:', (tx.sales_value<=0).mean())
# baskets per household 84d
b = tx[tx.day>375].groupby('household_key').basket_id.nunique()
print('baskets 84d:', b.describe())
# check E006 target relationship quickly
e6 = A.load_saved('e006_zero_inflation.parquet')
print(e6.shape, e6.snapshot_day.unique())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, re

def entropy(s):
    s = s[s > 0]
    if len(s) == 0: return np.nan
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())

def fn(view, snapshot_day):
    day = view.day
    hh = pd.Index(np.asarray(view.households).ravel(), name='household_key')
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)
    t84 = tx[tx.day > day - 84]; t28 = tx[tx.day > day - 28]; t7 = tx[tx.day > day - 7]
    t26w = tx[tx.day > day - 182]

    # store behaviour
    out['s_nstores_84'] = t84.groupby('household_key').store_id.nunique()
    out['s_nstores_28'] = t28.groupby('household_key').store_id.nunique()
    out['s_nstores_364'] = tx[tx.day > day-364].groupby('household_key').store_id.nunique()
    se = t84.groupby(['household_key','store_id']).sales_value.sum()
    out['s_store_ent_84'] = se.groupby(level=0).apply(entropy)
    top84 = se.groupby(level=0).max(); tot84 = se.groupby(level=0).sum()
    sh84 = top84/tot84
    se28 = t28.groupby(['household_key','store_id']).sales_value.sum()
    sh28 = se28.groupby(level=0).max()/se28.groupby(level=0).sum()
    out['s_topstore_spend_share_84'] = sh84
    out['s_loyalty_shift'] = sh28 - sh84

    # commodity mix
    prod = view.table('products')[['product_id','commodity_desc']]
    m84 = t84.merge(prod, on='product_id', how='left')
    m84['commodity_desc'] = m84['commodity_desc'].fillna('UNK')
    mall = tx.merge(prod, on='product_id', how='left')
    mall['commodity_desc'] = mall['commodity_desc'].fillna('UNK')
    topc = mall.groupby('commodity_desc').sales_value.sum().nlargest(15).index
    cspend = m84.groupby(['household_key','commodity_desc']).sales_value.sum().unstack(fill_value=0.0)
    totc = cspend.sum(axis=1)
    for c in topc:
        if c in cspend.columns:
            out['c_share_' + re.sub(r'\W+','_',str(c))[:22]] = cspend[c]/totc
    out['c_ncomm_84'] = m84.groupby('household_key').commodity_desc.nunique()
    out['c_comm_ent_84'] = cspend.apply(entropy, axis=1)

    # basket variability
    b84 = t84.groupby(['household_key','basket_id']).sales_value.sum()
    g84 = b84.groupby(level=0)
    out['b_std_84'] = g84.std(); out['b_max_84'] = g84.max()
    out['b_cv_84'] = g84.std()/g84.mean(); out['b_max_share_84'] = g84.max()/g84.sum()
    out['b_ntrips_7'] = t7.groupby('household_key').basket_id.nunique()
    out['b_active_days_28'] = t28.groupby('household_key').day.nunique()

    # weekly CV over 26 weeks
    wk = t26w.assign(w=(t26w.day+8)//7)
    ws = wk.groupby(['household_key','w']).sales_value.sum()
    gw = ws.groupby(level=0)
    out['w_cv_26'] = gw.std()/gw.mean()
    out['w_zero_share_26'] = 1 - gw.count()/26.0

    # discounts per trip
    disc28 = t28.groupby('household_key')[['coupon_disc','coupon_match_disc','retail_disc']].sum().sum(axis=1)
    trips28 = t28.groupby('household_key').basket_id.nunique()
    out['d_disc_per_trip_28'] = -disc28/trips28

    # time of day
    tt = pd.to_numeric(t84.trans_time, errors='coerce')
    t84m = t84.assign(morn=(tt<1200), aft=(tt>=1200)&(tt<1700))
    out['t_morning_share_84'] = t84m.groupby('household_key').morn.mean()
    out['t_afternoon_share_84'] = t84m.groupby('household_key').aft.mean()

    return out.reindex(hh)

v = A.snapshot(459)
res = fn(v, 459)
print(res.shape)
print(res.isna().mean().sort_values(ascending=False).head(10))
print(res.dropna(axis=1, how='all').describe().T[['mean','std','min','max']].round(3).to_string())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, re

def entropy(s):
    s = s[s > 0]
    if len(s) == 0: return np.nan
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())

def fn(view, snapshot_day):
    day = view.day
    hh = pd.Index(np.asarray(view.households).ravel(), name='household_key')
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)
    t84 = tx[tx.day > day - 84]; t28 = tx[tx.day > day - 28]; t7 = tx[tx.day > day - 7]
    t26w = tx[tx.day > day - 182]

    # store behaviour
    out['s_nstores_84'] = t84.groupby('household_key').store_id.nunique()
    out['s_nstores_28'] = t28.groupby('household_key').store_id.nunique()
    out['s_nstores_364'] = tx[tx.day > day-364].groupby('household_key').store_id.nunique()
    se = t84.groupby(['household_key','store_id']).sales_value.sum()
    out['s_store_ent_84'] = se.groupby(level=0).apply(entropy)
    sh84 = se.groupby(level=0).max()/se.groupby(level=0).sum()
    se28 = t28.groupby(['household_key','store_id']).sales_value.sum()
    sh28 = se28.groupby(level=0).max()/se28.groupby(level=0).sum()
    out['s_topstore_spend_share_84'] = sh84
    out['s_loyalty_shift'] = sh28 - sh84

    # commodity mix
    prod = view.table('products')[['product_id','commodity_desc']].copy()
    prod['commodity_desc'] = prod['commodity_desc'].astype(str).fillna('UNK')
    m84 = t84.merge(prod, on='product_id', how='left')
    m84['commodity_desc'] = m84['commodity_desc'].fillna('UNK')
    mall = tx.merge(prod, on='product_id', how='left')
    mall['commodity_desc'] = mall['commodity_desc'].fillna('UNK')
    topc = mall.groupby('commodity_desc').sales_value.sum().nlargest(15).index
    cspend = m84.groupby(['household_key','commodity_desc']).sales_value.sum().unstack(fill_value=0.0)
    totc = cspend.sum(axis=1)
    for c in topc:
        if c in cspend.columns:
            out['c_share_' + re.sub(r'\W+','_',str(c))[:22]] = cspend[c]/totc
    out['c_ncomm_84'] = m84.groupby('household_key').commodity_desc.nunique()
    out['c_comm_ent_84'] = cspend.apply(entropy, axis=1)

    # basket variability
    b84 = t84.groupby(['household_key','basket_id']).sales_value.sum()
    g84 = b84.groupby(level=0)
    out['b_std_84'] = g84.std(); out['b_max_84'] = g84.max()
    out['b_cv_84'] = g84.std()/g84.mean(); out['b_max_share_84'] = g84.max()/g84.sum()
    out['b_ntrips_7'] = t7.groupby('household_key').basket_id.nunique()
    out['b_active_days_28'] = t28.groupby('household_key').day.nunique()

    # weekly CV over 26 weeks
    wk = t26w.assign(w=(t26w.day+8)//7)
    ws = wk.groupby(['household_key','w']).sales_value.sum()
    gw = ws.groupby(level=0)
    out['w_cv_26'] = gw.std()/gw.mean()
    out['w_zero_share_26'] = 1 - gw.count()/26.0

    # discounts per trip
    disc28 = t28.groupby('household_key')[['coupon_disc','coupon_match_disc','retail_disc']].sum().sum(axis=1)
    trips28 = t28.groupby('household_key').basket_id.nunique()
    out['d_disc_per_trip_28'] = -disc28/trips28

    # time of day
    tt = pd.to_numeric(t84.trans_time, errors='coerce')
    t84m = t84.assign(morn=(tt<1200), aft=(tt>=1200)&(tt<1700))
    out['t_morning_share_84'] = t84m.groupby('household_key').morn.mean()
    out['t_afternoon_share_84'] = t84m.groupby('household_key').aft.mean()

    return out.reindex(hh)

v = A.snapshot(459)
res = fn(v, 459)
print(res.shape)
print(res.isna().mean().sort_values(ascending=False).head(8).round(3))
print(res.describe().T[['mean','std','min','max']].round(3).to_string())


# ---- cell ----
import agent_api as A
import numpy as np
v = A.snapshot(459)
h = v.households
print(type(h), getattr(h, 'shape', None))
print(repr(h)[:300])


# ---- cell ----
import agent_api as A
print([a for a in dir(A) if not a.startswith('_')])
print(A.KEYS, A.TARGET)
print(A.describe_tables())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    h = view.households
    print('type:', type(h))
    try:
        print('len:', len(h))
    except Exception as e:
        print('len err', e)
    print(repr(h)[:200])
    t = view.table('transactions')
    print('tx shape', t.shape, 'max day', t.day.max())
    print('view.day', view.day, 'view.week', view.week)
    return pd.DataFrame({'x': 1.0}, index=list(h)[:5] if not isinstance(h, pd.Index) else h[:5])

res = A.build_features(fn)
print(res.shape)
print(res.head())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, re

def entropy(s):
    s = s[s > 0]
    if len(s) == 0: return np.nan
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())

def fn(view, snapshot_day):
    day = view.day
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)
    t84 = tx[tx.day > day - 84]; t28 = tx[tx.day > day - 28]; t7 = tx[tx.day > day - 7]
    t26w = tx[tx.day > day - 182]

    # store behaviour
    out['s_nstores_84'] = t84.groupby('household_key').store_id.nunique()
    out['s_nstores_28'] = t28.groupby('household_key').store_id.nunique()
    out['s_nstores_364'] = tx[tx.day > day-364].groupby('household_key').store_id.nunique()
    se = t84.groupby(['household_key','store_id']).sales_value.sum()
    out['s_store_ent_84'] = se.groupby(level=0).apply(entropy)
    out['s_topstore_spend_share_84'] = se.groupby(level=0).max()/se.groupby(level=0).sum()
    se28 = t28.groupby(['household_key','store_id']).sales_value.sum()
    out['s_loyalty_shift'] = se28.groupby(level=0).max()/se28.groupby(level=0).sum() - out['s_topstore_spend_share_84']

    # commodity mix
    prod = view.table('products')[['product_id','commodity_desc']].copy()
    prod['commodity_desc'] = prod['commodity_desc'].astype(str)
    m84 = t84.merge(prod, on='product_id', how='left')
    m84['commodity_desc'] = m84['commodity_desc'].fillna('UNK')
    mall = tx.merge(prod, on='product_id', how='left')
    mall['commodity_desc'] = mall['commodity_desc'].fillna('UNK')
    topc = mall.groupby('commodity_desc').sales_value.sum().nlargest(15).index
    cspend = m84.groupby(['household_key','commodity_desc']).sales_value.sum().unstack(fill_value=0.0)
    totc = cspend.sum(axis=1)
    for c in topc:
        if c in cspend.columns:
            out['c_share_' + re.sub(r'\W+','_',str(c))[:22]] = cspend[c]/totc
    out['c_ncomm_84'] = m84.groupby('household_key').commodity_desc.nunique()
    out['c_comm_ent_84'] = cspend.apply(entropy, axis=1)

    # basket variability
    b84 = t84.groupby(['household_key','basket_id']).sales_value.sum()
    g84 = b84.groupby(level=0)
    out['b_std_84'] = g84.std(); out['b_max_84'] = g84.max()
    out['b_cv_84'] = (g84.std()/g84.mean()).replace([np.inf,-np.inf], np.nan)
    out['b_max_share_84'] = g84.max()/g84.sum()
    out['b_ntrips_7'] = t7.groupby('household_key').basket_id.nunique()
    out['b_active_days_28'] = t28.groupby('household_key').day.nunique()

    # weekly CV over 26 weeks
    wk = t26w.assign(w=(t26w.day+8)//7)
    ws = wk.groupby(['household_key','w']).sales_value.sum()
    gw = ws.groupby(level=0)
    out['w_cv_26'] = (gw.std()/gw.mean()).replace([np.inf,-np.inf], np.nan)
    out['w_zero_share_26'] = 1 - gw.count()/26.0

    # discounts per trip
    disc28 = t28.groupby('household_key')[['coupon_disc','coupon_match_disc','retail_disc']].sum().sum(axis=1)
    trips28 = t28.groupby('household_key').basket_id.nunique()
    out['d_disc_per_trip_28'] = -disc28/trips28

    # time of day
    tt = pd.to_numeric(t84.trans_time, errors='coerce')
    t84m = t84.assign(morn=(tt<1200), aft=(tt>=1200)&(tt<1700))
    out['t_morning_share_84'] = t84m.groupby('household_key').morn.mean()
    out['t_afternoon_share_84'] = t84m.groupby('household_key').aft.mean()

    return out.reindex(hh).replace([np.inf,-np.inf], np.nan)

res = A.build_features(fn)
print('built', res.shape)
e6 = A.load_saved('e006_zero_inflation.parquet')
print('e6', e6.shape)
merged = e6.merge(res, on=['household_key','snapshot_day'], how='left', validate='one_to_one')
print('merged', merged.shape, 'new cols:', merged.shape[1]-e6.shape[1])
print('NaN frac new cols (max):', merged[merged.columns[e6.shape[1]:]].isna().mean().max().round(3))
path = A.save_table(merged, 'e010_store_mix.parquet')
print(path)
