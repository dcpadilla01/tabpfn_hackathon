import agent_api, pandas as pd, numpy as np

def stock_block(view, sd):
    hh = pd.Index(view.households)
    tx = view.transactions
    d = tx[(tx.day > sd-84) & (tx.day <= sd)]
    b = d.groupby(['household_key','basket_id']).agg(spend=('sales_value','sum'), day=('day','max')).reset_index()
    b = b[b.household_key.isin(hh)]
    g84 = b.groupby('household_key')
    tot84 = g84.spend.sum(); cnt84 = g84.size(); max84 = g84.spend.max()
    # most recent max-spend basket (ties -> latest)
    bmax = b.sort_values(['spend','day']).groupby('household_key').tail(1).set_index('household_key')
    last_day = g84.day.max()
    b28 = b[b.day > sd-28]
    g28 = b28.groupby('household_key')
    tot28 = g28.spend.sum(); max28 = g28.spend.max(); cnt28 = g28.size()
    top2 = b28.sort_values('spend', ascending=False).groupby('household_key').head(2).groupby('household_key').spend.sum()
    # weekly spend, last 12 weeks
    dd = d.copy(); dd['wk'] = (dd.day+8)//7
    cur = (sd+8)//7
    s = dd.groupby(['household_key','wk']).sales_value.sum().reset_index()
    p = s.pivot(index='household_key', columns='wk', values='sales_value')
    W = {}
    for k in range(1,13):
        col = cur-k
        W[f'w{k}'] = p[col].reindex(hh).fillna(0.0) if col in p.columns else pd.Series(0.0, index=hh)
    W = pd.DataFrame(W)
    tot12 = W.sum(axis=1)
    k = np.arange(1,13)
    L = np.log1p(W.values)
    Lbar = L.mean(axis=1, keepdims=True)
    slope = ((L - Lbar) * (k - k.mean())).sum(axis=1) / ((k - k.mean())**2).sum()
    f = pd.DataFrame(index=hh)
    f['stk_max_share28'] = (max28/tot28).reindex(hh).fillna(0.0)
    f['stk_max_share84'] = (max84/tot84).reindex(hh).fillna(0.0)
    f['stk_top2_share28'] = (top2/tot28).reindex(hh).fillna(0.0)
    f['stk_maxb28'] = np.log1p(max28.reindex(hh).fillna(0.0))
    f['stk_maxb84'] = np.log1p(max84.reindex(hh).fillna(0.0))
    f['stk_gap_max'] = (sd - bmax.day).reindex(hh).fillna(84).astype(float)
    f['stk_gap_max_r'] = f.stk_gap_max/28.0
    f['stk_days_last'] = (sd - last_day).reindex(hh).fillna(84).astype(float)
    f['stk_cnt7'] = b28[b28.day > sd-7].groupby('household_key').size().reindex(hh).fillna(0.0)
    f['stk_s7_28'] = W.w1/(tot28.reindex(hh).fillna(0.0)+1)
    f['stk_s14_28'] = (W.w1+W.w2)/(tot28.reindex(hh).fillna(0.0)+1)
    f['stk_w1_share'] = W.w1/(tot12+1)
    f['stk_w3_share'] = (W.w1+W.w2+W.w3)/(tot12+1)
    f['stk_w4_12_share'] = W[[f'w{k2}' for k2 in range(4,13)]].sum(axis=1)/(tot12+1)
    f['stk_slope12'] = slope
    f['stk_no7'] = (W.w1 <= 0).astype(float)
    f['stk_stock_flag'] = ((f.stk_max_share28 > 0.5) & (cnt28.reindex(hh).fillna(0) <= 3)).astype(float)
    f['stk_max_over_mean'] = (max28/(tot28/np.maximum(cnt28,1))).reindex(hh).fillna(0.0)
    return f

tbl = agent_api.build_features(stock_block)
print('built', tbl.shape, 'snaps', sorted(tbl.snapshot_day.unique()))
agent_api.save_table(tbl, 'stock_v1.parquet')
base = agent_api.load_saved('e011_rank.parquet')
m = base.merge(tbl.drop(columns=['snapshot_day']), on=['household_key','snapshot_day'], how='left', suffixes=('','_stk'))
# build_features returns snapshot_day col? check
print('merged', m.shape)
assert len(m) == len(base), (len(m), len(base))
agent_api.save_table(m, 'e013_stock.parquet')
print('saved e013_stock', m.shape)
print([c for c in m.columns if c.startswith('stk_')])
