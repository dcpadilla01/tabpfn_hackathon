import numpy as np, pandas as pd, agent_api as A

def fn(view, snapshot_day):
    t = view.transactions
    hh = view.households
    if hh is None:
        first = t.groupby('household_key').day.min()
        hh = first[first <= snapshot_day - 84].index
    hh = pd.Index(hh, name='household_key')
    p = view.products[['product_id','department','brand']]
    t = t[t.household_key.isin(set(hh))].merge(p, on='product_id', how='left')
    t['department'] = t.department.fillna('UNKNOWN')
    last = snapshot_day
    day = t.day
    w = {k: day > last - k for k in (28,84,365)}
    out = {}
    t84 = t[w[84]]
    dep = t84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh, fill_value=0.0)
    tot84 = dep.sum(axis=1)
    for c in dep.columns:
        out[f'd84_{c}'] = dep[c]
        out[f'sh84_{c}'] = dep[c] / tot84.replace(0, np.nan)
    top6 = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD']
    t28 = t[w[28]]
    dep28 = t28[t28.department.isin(top6)].groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh, fill_value=0.0)
    for c in dep28.columns:
        out[f'd28_{c}'] = dep28[c]
    sp84 = t84.groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    priv84 = t84[t84.brand=='Private'].groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    out['priv_share_84'] = priv84 / sp84.replace(0, np.nan)
    t365 = t[w[365]]
    priv365 = t365[t365.brand=='Private'].groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    sp365 = t365.groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    out['priv_share_365'] = priv365 / sp365.replace(0, np.nan)
    for k in (28,84):
        tk = t[w[k]]
        out[f'retail_disc_{k}'] = tk.groupby('household_key').retail_disc.sum().reindex(hh, fill_value=0.0)
    tk = t[w[84]]
    out['coupon_disc_84'] = tk.groupby('household_key').coupon_disc.sum().reindex(hh, fill_value=0.0)
    out['match_disc_84'] = tk.groupby('household_key').coupon_match_disc.sum().reindex(hh, fill_value=0.0)
    disc = (tk.retail_disc.abs() + tk.coupon_disc.abs() + tk.coupon_match_disc.abs())
    g = disc.groupby(tk.household_key).sum().reindex(hh, fill_value=0.0)
    out['disc_share_84'] = g / (sp84.abs() + g).replace(0, np.nan)
    bt = t28.groupby(['household_key','basket_id']).trans_time.min().reset_index()
    out['avg_first_time_28'] = bt.groupby('household_key').trans_time.mean().reindex(hh)
    out['share_morning_28'] = bt.assign(m=bt.trans_time<1200).groupby('household_key').m.mean().reindex(hh)
    out['n_stores_84'] = t84.groupby('household_key').store_id.nunique().reindex(hh)
    gs = t84.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    idx = gs.groupby('household_key').sales_value.idxmax()
    out['top_store_share_84'] = gs.loc[idx].set_index('household_key').sales_value / sp84.replace(0, np.nan)
    out['nprod_84'] = t84.groupby('household_key').product_id.nunique().reindex(hh)
    a = t84[['household_key','product_id']].drop_duplicates()
    b = t[w[365] & ~w[84]][['household_key','product_id']].drop_duplicates()
    rep = a.merge(b, on=['household_key','product_id']).groupby('household_key').size().reindex(hh, fill_value=0)
    out['repeat_share_84'] = rep / out['nprod_84'].replace(0, np.nan)
    bd = t84[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    gd = bd.groupby('household_key').day.diff()
    gg = gd.groupby(bd.household_key).agg(['mean','std'])
    out['gap_mean_84'] = gg['mean'].reindex(hh)
    out['gap_std_84'] = gg['std'].reindex(hh)
    wk = t84.groupby(['household_key','week_no']).sales_value.sum().unstack(fill_value=0.0).reindex(hh, fill_value=0.0)
    out['spend_vol_84'] = wk.std(axis=1)
    out['zero_week_share_84'] = (wk == 0).mean(axis=1)
    pr = dep.div(tot84.replace(0, np.nan), axis=0)
    ent = (-(pr * np.log(pr))).sum(axis=1)
    out['dept_entropy_84'] = ent.fillna(0.0)
    out['avg_qty_84'] = t84.groupby('household_key').quantity.mean().reindex(hh)
    return pd.DataFrame(out).reindex(hh)

v = A.snapshot(459)
X = fn(v, 459)
print('smoke shape', X.shape)
print(X.isna().mean().sort_values(ascending=False).head(8))
print(X.head(3).iloc[:, :6])