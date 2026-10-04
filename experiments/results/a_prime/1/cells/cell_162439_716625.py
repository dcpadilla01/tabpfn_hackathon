import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    sd = snapshot_day
    tx = view.transactions
    tx84 = tx[tx['day'] > sd - 84]
    g = tx84.groupby('household_key')
    agg = g[['retail_disc','coupon_disc','coupon_match_disc']].sum()
    agg['disc_net_84'] = agg.sum(axis=1)          # net discount dollars (<=0)
    agg['coupon_disc_84'] = agg['coupon_disc']    # manufacturer coupon discounts
    agg['match_disc_84'] = agg['coupon_match_disc']
    agg = agg[['disc_net_84','coupon_disc_84','match_disc_84']]
    # coupon redemptions
    cr = view.coupon_redemptions
    cr84 = cr[cr['day'] > sd - 84]
    red = cr84.groupby('household_key').size().rename('coupon_redemptions_84')
    dsl = cr.groupby('household_key')['day'].max().rsub(sd).rename('dsl_coupon')
    out = agg.join(red, how='outer').join(dsl, how='outer')
    # cyclical seasonality from snapshot week
    wk = float(view.week)
    out['wk_sin'] = np.sin(2*np.pi*wk/52.0)
    out['wk_cos'] = np.cos(2*np.pi*wk/52.0)
    out['wk_sin2'] = np.sin(4*np.pi*wk/52.0)
    out['wk_cos2'] = np.cos(4*np.pi*wk/52.0)
    out = out.reindex(view.households)
    return out

X = agent_api.build_features(fn)
print(X.shape); print(X.dtypes); print(X.head())
base = agent_api.load_saved('e016_smoothed.parquet')
print('base', base.shape)
m = base.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'new cols:', [c for c in m.columns if c not in base.columns])
print(m[['disc_net_84','coupon_disc_84','match_disc_84','coupon_redemptions_84','dsl_coupon','wk_sin','wk_cos']].describe().round(3))
path = agent_api.save_table(m, 'e017_disc_seasonal.parquet')
print(path)