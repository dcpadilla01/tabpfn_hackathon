import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
print('hh type:', type(v.households))
try:
    print(v.households.head())
except Exception as e:
    print('hh head fail', repr(e))
print('v.day', v.day, 'v.week', v.week)
tx = v.transactions
print('tx', tx.shape)
print(tx[['sales_value','quantity']].describe().to_string())
print('neg share', float((tx.sales_value<0).mean()), 'zero share', float((tx.sales_value==0).mean()))
d = tx[tx.day>459-56]
bs = d.groupby(['household_key','basket_id'])['sales_value'].sum()
print('baskets/hh 56d:'); print(bs.groupby('household_key').size().describe().to_string())
t = agent_api.train_targets()
print('targets', t.shape)
print(t.future_spend_4w.describe().to_string())
print('zero share', float((t.future_spend_4w==0).mean()))
print(t.future_spend_4w.quantile([.5,.75,.9,.95,.99]).to_string())
for p in ['e011_rank.parquet']:
    try:
        e = agent_api.load_saved(p); print('loaded', p, e.shape); print(list(e.columns)[:40])
    except Exception as ex: print('fail', p, repr(ex))
print(agent_api.snapshot_days())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
e = agent_api.load_saved('e011_rank.parquet')
t = agent_api.train_targets()
m = e.merge(t, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'nan target', m.future_spend_4w.isna().sum())

# per snapshot: last-28d basket totals; stockpile = max basket / total; share of spend in last 7 days
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    hh = m.loc[m.snapshot_day==sd,'household_key'].unique()
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    b = d.groupby(['household_key','basket_id'])['sales_value'].sum().reset_index()
    g = b.groupby('household_key')['sales_value']
    f = pd.DataFrame({'tot':g.sum(),'maxb':g.max(),'cnt':g.size()})
    f['max_share'] = f.maxb/f.tot
    f['sd']=sd
    f=f.reset_index()
    rows.append(f)
F = pd.concat(rows)
m2 = m.merge(F, on=['household_key','snapshot_day'], how='left')
m2['max_share'] = m2.max_share.fillna(0)
m2['log_t'] = np.log1p(m2.future_spend_4w)
m2['log_tot'] = np.log1p(m2.tot)
for c in ['max_share','tot','maxb','log_tot']:
    print(c, 'corr log-target:', round(np.corrcoef(m2[c].fillna(0), m2.log_t)[0,1],4))
# partial: residualize log target on log_tot, then corr with max_share
import numpy.linalg as la
X = np.c_[np.ones(len(m2)), m2.log_tot.fillna(0)]
y = m2.log_t.fillna(0).values
beta = la.lstsq(X,y,rcond=None)[0]
r = y - X@beta
print('partial corr max_share|log_tot:', round(np.corrcoef(m2.max_share.fillna(0), r)[0,1],4))
# spend in final 7 days share
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    hh = m.loc[m.snapshot_day==sd,'household_key'].unique()
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    s = d.groupby('household_key')['sales_value'].sum()
    s7 = d[d.day>sd-7].groupby('household_key')['sales_value'].sum()
    f = pd.DataFrame({'s28':s,'s7':s7}); f['s7_share']=(f.s7/f.s28).fillna(0); f['sd']=sd
    rows.append(f.reset_index())
G = pd.concat(rows)
m3 = m.merge(G, on=['household_key','snapshot_day'], how='left')
m3['s7_share']=m3.s7_share.fillna(0)
r2 = y - np.c_[np.ones(len(m3)), np.log1p(m3.s28.fillna(0))]@la.lstsq(np.c_[np.ones(len(m3)), np.log1p(m3.s28.fillna(0))], y, rcond=None)[0]
print('partial corr s7_share|log_s28:', round(np.corrcoef(m3.s7_share, r2)[0,1],4))
print(m2[['max_share']].describe().to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
e = agent_api.load_saved('e011_rank.parquet')
t = agent_api.train_targets()
m = e.merge(t, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'nan target', m.future_spend_4w.isna().sum())
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    b = d.groupby(['household_key','basket_id'])['sales_value'].sum().reset_index()
    g = b.groupby('household_key')['sales_value']
    f = pd.DataFrame({'tot':g.sum(),'maxb':g.max(),'cnt':g.size()})
    f['max_share'] = f.maxb/f.tot
    f['snapshot_day']=sd
    rows.append(f.reset_index())
F = pd.concat(rows)
m2 = m.merge(F, on=['household_key','snapshot_day'], how='left')
m2['max_share'] = m2.max_share.fillna(0)
m2['log_t'] = np.log1p(m2.future_spend_4w)
m2['log_tot'] = np.log1p(m2.tot)
for c in ['max_share','tot','maxb','log_tot']:
    print(c, 'corr log-target:', round(np.corrcoef(m2[c].fillna(0), m2.log_t)[0,1],4))
import numpy.linalg as la
X = np.c_[np.ones(len(m2)), m2.log_tot.fillna(0)]
y = m2.log_t.fillna(0).values
beta = la.lstsq(X,y,rcond=None)[0]
r = y - X@beta
print('partial corr max_share|log_tot:', round(np.corrcoef(m2.max_share.fillna(0), r)[0,1],4))
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    s = d.groupby('household_key')['sales_value'].sum()
    s7 = d[d.day>sd-7].groupby('household_key')['sales_value'].sum()
    f = pd.DataFrame({'s28':s,'s7':s7}); f['s7_share']=(f.s7/f.s28).fillna(0); f['snapshot_day']=sd
    rows.append(f.reset_index())
G = pd.concat(rows)
m3 = m.merge(G, on=['household_key','snapshot_day'], how='left')
m3['s7_share']=m3.s7_share.fillna(0)
X3 = np.c_[np.ones(len(m3)), np.log1p(m3.s28.fillna(0))]
beta3 = la.lstsq(X3,y,rcond=None)[0]
r2 = y - X3@beta3
print('partial corr s7_share|log_s28:', round(np.corrcoef(m3.s7_share, r2)[0,1],4))
print(m2[['max_share']].describe().to_string())
# also: does max_share relate to next-block drop? compare target vs p1 spend
m2['drop_next'] = np.where(m2.future_spend_4w.notna(), m2.future_spend_4w/(m2.tot+1), np.nan)
print('mean ratio target/tot by max_share quartile:')
print(m2.dropna(subset=['drop_next']).groupby(pd.qcut(m2.max_share.dropna(),4,duplicates='drop')).drop_next.mean().to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np, numpy.linalg as la
v = agent_api.snapshot(459); tx = v.transactions
e = agent_api.load_saved('e011_rank.parquet'); t = agent_api.train_targets()
m = e.merge(t, on=['household_key','snapshot_day']).dropna(subset=['future_spend_4w'])
print('train rows', len(m))
def part_corr(df, feat, base):
    X = np.c_[np.ones(len(df)), base]
    y = np.log1p(df.future_spend_4w.values)
    b = la.lstsq(X,y,rcond=None)[0]; r = y - X@b
    f = df[feat].fillna(0).values
    return round(np.corrcoef(f, r)[0,1],4)
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    b = d.groupby(['household_key','basket_id'])['sales_value'].sum().reset_index()
    g = b.groupby('household_key')['sales_value']
    f = pd.DataFrame({'tot':g.sum(),'maxb':g.max(),'cnt':g.size()})
    f['max_share']=f.maxb/f.tot
    f['snapshot_day']=sd
    rows.append(f.reset_index())
F = pd.concat(rows); m = m.merge(F, on=['household_key','snapshot_day'], how='left')
m['max_share']=m.max_share.fillna(0)
lt = np.log1p(m.tot.fillna(0).values)
print('corr max_share w log-target:', round(np.corrcoef(m.max_share, np.log1p(m.future_spend_4w))[0,1],4))
print('partial corr max_share | log tot:', part_corr(m,'max_share', lt))
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-84)&(tx.day<=sd)].copy()
    d['wk'] = (d.day+8)//7
    cur = (sd+8)//7
    s = d.groupby(['household_key','wk'])['sales_value'].sum().reset_index()
    p = s.pivot(index='household_key', columns='wk', values='sales_value')
    out = {}
    for k in range(1,13):
        col = cur-k
        out[f'w{k}'] = p[col] if col in p.columns else pd.Series(0.0, index=p.index)
    o = pd.DataFrame(out); o['snapshot_day']=sd
    rows.append(o.reset_index())
W = pd.concat(rows)
m = m.merge(W, on=['household_key','snapshot_day'], how='left')
wkcols=[f'w{k}' for k in range(1,13)]
m[wkcols]=m[wkcols].fillna(0)
m['w1_share'] = m.w1/(m[wkcols].sum(axis=1)+1)
m['w1_3_share'] = m[['w1','w2','w3']].sum(axis=1)/(m[wkcols].sum(axis=1)+1)
m['w4_12_share'] = m[[f'w{k}' for k in range(4,13)]].sum(axis=1)/(m[wkcols].sum(axis=1)+1)
lt84 = np.log1p(m[wkcols].sum(axis=1).values)
for c in ['w1_share','w1_3_share','w4_12_share']:
    print(c, 'partial | log84:', part_corr(m, c, lt84))
print(m[['w1_share','w1_3_share','w4_12_share']].describe().to_string())
# check e011 already has x_spend_p1 (prior aligned 4w block) — correlation of target with p1
print('corr log-target vs log p1:', part_corr(m,'x_spend_p1', np.zeros(len(m))))


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd
tbl = agent_api.load_saved('stock_v1.parquet')
print(list(tbl.columns)); print(tbl.head(3).to_string())
base = agent_api.load_saved('e011_rank.parquet')
print('base cols sample:', list(base.columns)[:5])


# ---- cell ----
import agent_api, pandas as pd
tbl = agent_api.load_saved('stock_v1.parquet')
base = agent_api.load_saved('e011_rank.parquet')
m = base.merge(tbl, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'base', base.shape)
assert len(m)==len(base)
assert m[[c for c in tbl.columns if c.startswith('stk_')]].isna().sum().sum()==0
agent_api.save_table(m, 'e013_stock.parquet')
print('saved e013_stock.parquet')
