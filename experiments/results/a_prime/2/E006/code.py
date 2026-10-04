import agent_api as api, pandas as pd, numpy as np
for n in ['recency_agg.parquet','dept_mix_recency.parquet','temporal_structure.parquet','basket_tenure.parquet','marketing_exposure.parquet','behavioral_candidates.parquet']:
    df = api.load_saved(n)
    print(n, df.shape)
bt = api.load_saved('basket_tenure.parquet'); print('BT cols:', list(bt.columns))
bc = api.load_saved('behavioral_candidates.parquet'); print('BC cols:', list(bc.columns))
v = api.snapshot()
t = v.transactions
print('tx', t.shape)
print(t[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe().T)
p = v.products
print('prod', p.shape)
print(p['brand'].value_counts(dropna=False).head())
print('dept nunique', p['department'].nunique(), 'commod nunique', p['commodity_desc'].nunique())
tt = api.train_targets()
print('targets', tt.shape); print(tt['future_spend_4w'].describe())
print('households type', type(v.households), len(v.households))
print('cr', v.coupon_redemptions.shape)

# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def fn(view, s):
    hh = pd.Index(view.households, name='household_key')
    n = len(hh)
    code = pd.Series(np.arange(n), index=hh)
    tx = view.transactions
    d = tx['day'].to_numpy(); sv = tx['sales_value'].to_numpy()
    hc = code.reindex(tx['household_key'].to_numpy()).to_numpy()
    ok = hc >= 0
    d, sv, hc = d[ok], sv[ok], hc[ok]
    out = pd.DataFrame(index=hh)
    def wsum(lo, hi):
        m = (d >= max(lo, 1)) & (d <= hi)
        return np.bincount(hc[m], weights=sv[m], minlength=n)
    # target-aligned lagged 28d windows: window j covers [s+1-28j, s+28-28j]
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
    # mix features over last 84 days
    m84 = ok & (d >= s - 83)
    tx84 = tx[m84]
    sp84 = sv[m84]; h84 = hc[m84]
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

# ---- cell ----
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
    m84 = ok & (d >= s - 83)
    tx84 = tx[m84]
    sp84 = sv[m84]; h84 = hc[m84]
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

# ---- cell ----
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
    tx84 = tx[m84]
    sp84 = sv[m84]; h84 = hc[m84]
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

# ---- cell ----
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