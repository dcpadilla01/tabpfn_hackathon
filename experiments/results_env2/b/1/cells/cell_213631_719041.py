
import agent_api, pandas as pd, numpy as np

def cand(view, sd):
    tx = view.table('transactions')
    hh = view.households
    f = pd.DataFrame(index=hh)
    d = tx['day']
    # candidate: spend in specific recent windows
    for w in (7, 14, 21, 28, 42, 56, 84, 112, 168):
        f[f'c_sp{w}'] = tx[d > sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    # lagged same-window spend a year ago (364-392 days back)
    f['c_splag1y'] = tx[(d > sd-392) & (d <= sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag2y'] = tx[(d > sd-756) & (d <= sd-728)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    # weekly spend volatility last 12 weeks
    t2 = tx[d > sd-84].copy()
    t2['wk'] = t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum()
    g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    f['c_wkmean'] = g.mean().reindex(hh, fill_value=0.0)
    f['c_wkcv'] = (f['c_wkstd'] / (f['c_wkmean']+1e-9))
    # trip gap stats last 168d
    t3 = tx[d > sd-168].drop_duplicates(['household_key','day'])
    ds = t3.groupby('household_key')['day'].agg(['min','max','count'])
    f['c_span'] = (ds['max']-ds['min']).reindex(hh)
    f['c_gapmean'] = (f['c_span']/(ds['count']-1).clip(lower=1)).reindex(hh)
    # share of spend at top store (loyalty) last 168d
    ss = t3.groupby(['household_key','store_id'])['sales_value'].sum()
    f['c_topstore'] = ss.groupby('household_key').max().reindex(hh, fill_value=0.0)/ (f['c_sp168']+1e-9)
    # distinct products last 168
    f['c_nprod168'] = t3.groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    # avg basket value last 84
    bs = tx[d > sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    # coupon redemption count last 84/168
    cr = view.table('coupon_redemptions')
    for w in (84,168):
        f[f'c_nred{w}'] = cr[cr['day']>sd-w].groupby('household_key').size().reindex(hh, fill_value=0)
    # active days ratio last 28
    f['c_act28'] = tx[d > sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    return f

X = agent_api.build_features(cand)
t = agent_api.train_targets()
m = t.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
num = [c for c in X.columns if pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m['future_spend_4w']).sort_values()
print(corr)
