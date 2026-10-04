import agent_api as api, pandas as pd, numpy as np

def fn(view, s):
    hh = pd.Index(view.households, name='household_key')
    n = len(hh)
    code = pd.Series(np.arange(n), index=hh)
    tx = view.transactions
    d = tx['day'].to_numpy(); sv = tx['sales_value'].to_numpy()
    hc = code.reindex(tx['household_key'].to_numpy()).to_numpy()
    ok = hc >= 0
    d, sv, hc = d[ok], sv[ok], hc[ok].astype(np.int64)
    out = pd.DataFrame(index=hh)
    def wsum(lo, hi):
        m = (d >= max(lo, 1)) & (d <= hi)
        return np.bincount(hc[m], weights=sv[m], minlength=n)
    L = np.empty((n, 13))
    for j in range(1, 14):
        L[:, j-1] = wsum(s + 1 - 28*j, s + 28 - 28*j)
    for j in range(2, 14):
        out[f'tlag_{j}'] = L[:, j-1]
    L2 = L[:, 1:]
    out['tlag_mean'] = L2.mean(1); out['tlag_std'] = L2.std(1)
    out['tlag_max'] = L2.max(1);  out['tlag_zero_cnt'] = (L2 == 0).sum(1)
    out['tlag_ratio_yoy'] = L[:, 12] / (L[:, 0] + 1.0)
    out['tlag_ratio_prev'] = L[:, 1] / (L[:, 0] + 1.0)
    out['spend_7'] = wsum(s-6, s); out['spend_14'] = wsum(s-13, s)
    out['spend_7_over_28'] = out['spend_7'] / (L[:, 0] + 1.0)
    m84 = d >= s - 83
    idx = np.flatnonzero(m84)
    tx84 = tx.iloc[idx]
    sp84 = sv[idx]; h84 = hc[idx]
    tot = np.bincount(h84, weights=sp84, minlength=n)
    prod = view.products
    br = tx84['product_id'].map(prod['brand']).to_numpy()
    mp = br == 'Private'
    priv = np.bincount(h84[mp], weights=sp84[mp], minlength=n)
    out['private_share_84'] = np.where(tot > 0, priv / np.maximum(tot, 1e-9), np.nan)
    cds = -tx84['coupon_disc'].to_numpy()
    csh = np.bincount(h84, weights=cds, minlength=n)
    out['coupon_share_84'] = np.where(tot > 0, csh / np.maximum(tot, 1e-9), np.nan)
    com = tx84['product_id'].map(prod['commodity_desc']).to_numpy()
    dfc = pd.DataFrame({'h': h84, 'com': com, 'sv': sp84})
    gs = dfc.groupby(['h', 'com'])['sv'].sum()
    top = gs.groupby(level=0).max().reindex(hh).to_numpy()
    cnt = gs.groupby(level=0).size().reindex(hh).to_numpy()
    out['top_com_share_84'] = np.where(tot > 0, top / np.maximum(tot, 1e-9), np.nan)
    out['distinct_com_84'] = cnt
    cr = view.coupon_redemptions
    if cr is not None and len(cr):
        rc = cr[cr['day'] >= s - 83].groupby('household_key').size().reindex(hh)
        out['redemptions_84'] = rc.fillna(0).to_numpy()
    else:
        out['redemptions_84'] = 0.0
    return out

base = api.load_saved('basket_tenure.parquet')
feat = api.build_features(fn)
print('feat', feat.shape)
m = base.merge(feat, on=['household_key', 'snapshot_day'], how='inner')
print('merged', m.shape)
assert len(m) == len(base) == 36426
assert m[['household_key','snapshot_day']].duplicated().sum() == 0
path = api.save_table(m, 'churn_seasonality.parquet')
print(path)
print(m[['tlag_13','tlag_2','tlag_zero_cnt','spend_7','private_share_84','coupon_share_84','redemptions_84']].describe().T)