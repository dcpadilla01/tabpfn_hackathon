import agent_api, pandas as pd, numpy as np

s = 431
v = agent_api.snapshot(s)
tx = v.transactions
p = v.products

t = tx[(tx['day'] > s-168) & (tx['day'] <= s)][['household_key','day','product_id','sales_value']]
t = t.merge(p[['product_id','commodity_desc']], on='product_id', how='left')
t['commodity_desc'] = t['commodity_desc'].astype('object').fillna('UNK')

daily = t.groupby(['household_key','commodity_desc','day'], as_index=False)['sales_value'].sum()
daily = daily.sort_values(['household_key','commodity_desc','day'])

g = daily.groupby(['household_key','commodity_desc'])
occ = g.agg(first_d=('day','min'), last_d=('day','max'), n=('day','size'), tot=('sales_value','sum')).reset_index()
occ['mean_spend'] = occ['tot']/occ['n']
occ['gap'] = (occ['last_d']-occ['first_d'])/(occ['n']-1)
occ = occ[occ['n']>=2]
occ['gap_c'] = occ['gap'].clip(1, 60)
occ['next'] = occ['last_d'] + occ['gap_c']
occ['due'] = (occ['next'] <= s+28).astype(float)
occ['contrib'] = occ['due']*occ['mean_spend']

hh_feat = occ.groupby('household_key').agg(
    cyc_pred=('contrib','sum'),
    cyc_due=('due','sum'),
    cyc_active=('commodity_desc','size'),
).reset_index()
# pressure: sum over commodities of min(1, days_since_last/gap)
occ['pressure'] = np.minimum(1.0, (s - occ['last_d'])/occ['gap_c'])
press = occ.groupby('household_key')['pressure'].sum().rename('cyc_pressure').reset_index()
hh_feat = hh_feat.merge(press, on='household_key', how='left')
print(hh_feat.describe())

tt = agent_api.train_targets()
tt = tt[tt['snapshot_day']==s]
m = tt.merge(hh_feat, on='household_key', how='left').fillna({'cyc_pred':0,'cyc_due':0,'cyc_active':0,'cyc_pressure':0})
base = agent_api.load_saved('e009_ewma_longlags.parquet')
b = base[base['snapshot_day']==s][['household_key','spend_28','tlag_mean','ewma_4']]
m = m.merge(b, on='household_key', how='left')
print("rows", len(m))
for c in ['cyc_pred','cyc_due','cyc_active','cyc_pressure','spend_28','tlag_mean','ewma_4']:
    print(c, "corr:", round(m[c].corr(m['future_spend_4w']), 4))
# partial: corr of cyc_pred with target after removing spend_28 effect (simple residual)
import numpy.polynomial as _  # noop
def resid_corr(x, y, z):
    # residualize x and y on z (linear)
    A = np.vstack([z, np.ones_like(z)]).T
    bx = np.linalg.lstsq(A, x, rcond=None)[0]
    by = np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A@bx; ry = y - A@by
    return round(np.corrcoef(rx, ry)[0,1], 4)
print("partial cyc_pred | spend_28:", resid_corr(m['cyc_pred'].values, m['future_spend_4w'].values, m['spend_28'].values))
print("partial tlag_mean | spend_28:", resid_corr(m['tlag_mean'].values, m['future_spend_4w'].values, m['spend_28'].values))
print("partial cyc_pressure | spend_28:", resid_corr(m['cyc_pressure'].values, m['future_spend_4w'].values, m['spend_28'].values))
