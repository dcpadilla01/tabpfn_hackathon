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
