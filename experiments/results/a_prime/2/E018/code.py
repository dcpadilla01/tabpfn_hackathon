import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot(459)
print("tx shape", v.transactions.shape)
print("tx cols", v.transactions.columns.tolist())
print("has products:", hasattr(v, "products"))
p = v.products
print("prod shape", p.shape, p.columns.tolist())
hh = v.households
print("households type:", type(hh))
print("households head:\n", hh.head(3))

base = agent_api.load_saved('e009_ewma_longlags.parquet')
print("base shape", base.shape)
cols = base.columns.tolist()
print("first cols", cols[:8])
print("has spend_28:", 'spend_28' in cols, "| index col:", 'index' in cols)
print("ewma cols:", [c for c in cols if 'ewma' in c.lower()][:8])
print("tlag cols:", [c for c in cols if c.startswith('tlag')][:15])
print("KEYS", agent_api.KEYS, "TARGET", agent_api.TARGET)
print("snapdays", agent_api.snapshot_days())

# timing check for commodity-level cycle pipeline at s=459
s = 459
tx = v.transactions
t = tx[(tx['day'] > s-168) & (tx['day'] <= s)][['household_key','basket_id','day','product_id','sales_value']]
print("rows 168d:", len(t))
t = t.merge(p[['product_id','commodity_desc']], on='product_id', how='left')
t['commodity_desc'] = t['commodity_desc'].fillna('UNK')
daily = t.groupby(['household_key','commodity_desc','day'], as_index=False)['sales_value'].sum()
daily = daily.sort_values(['household_key','commodity_desc','day'])
print("daily rows:", len(daily))
occ = daily.groupby(['household_key','commodity_desc']).agg(last_d=('day','max'), n=('day','size'), tot=('sales_value','sum')).reset_index()
print("occ rows:", len(occ), "occ>=2:", int((occ['n']>=2).sum()))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot(459)
print("tx shape", v.transactions.shape)
print("prod shape", v.products.shape, v.products.columns.tolist())
print("households is None at snapshot view:", v.households)

base = agent_api.load_saved('e009_ewma_longlags.parquet')
print("base shape", base.shape)
cols = base.columns.tolist()
print("first cols", cols[:8])
print("has spend_28:", 'spend_28' in cols, "| index col:", 'index' in cols)
print("ewma cols:", [c for c in cols if 'ewma' in c.lower()][:10])
print("tlag cols:", [c for c in cols if c.startswith('tlag')][:15])
print("KEYS", agent_api.KEYS, "TARGET", agent_api.TARGET)
print("snapdays", agent_api.snapshot_days())

# timing check for commodity-level cycle pipeline at s=459
s = 459
tx = v.transactions
t = tx[(tx['day'] > s-168) & (tx['day'] <= s)][['household_key','basket_id','day','product_id','sales_value']]
print("rows 168d:", len(t))
p = v.products
t = t.merge(p[['product_id','commodity_desc']], on='product_id', how='left')
t['commodity_desc'] = t['commodity_desc'].fillna('UNK')
daily = t.groupby(['household_key','commodity_desc','day'], as_index=False)['sales_value'].sum()
daily = daily.sort_values(['household_key','commodity_desc','day'])
print("daily rows:", len(daily))
occ = daily.groupby(['household_key','commodity_desc']).agg(last_d=('day','max'), n=('day','size'), tot=('sales_value','sum')).reset_index()
print("occ rows:", len(occ), "occ>=2:", int((occ['n']>=2).sum()))


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np

BASE = agent_api.load_saved('e009_ewma_longlags.parquet')

def fn(view, s):
    tx = view.transactions
    p = view.products
    t = tx[(tx['day'] > s-182) & (tx['day'] <= s)][['household_key','day','product_id','sales_value']].copy()
    hh_all = BASE.loc[BASE['snapshot_day']==s, 'household_key']
    if len(t) == 0:
        return pd.DataFrame(index=hh_all)
    t = t.merge(p[['product_id','commodity_desc']], on='product_id', how='left')
    t['commodity_desc'] = t['commodity_desc'].astype('object').fillna('UNK')
    # recency weights for EWMA variant
    t['w'] = 0.5 ** ((s - t['day'])/56.0)
    t['sw'] = t['sales_value']*t['w']

    daily = t.groupby(['household_key','commodity_desc','day'], as_index=False)['sales_value'].sum()
    daily = daily.sort_values(['household_key','commodity_desc','day'])
    g = daily.groupby(['household_key','commodity_desc'])
    occ = g.agg(first_d=('day','min'), last_d=('day','max'), n=('day','size'), tot=('sales_value','sum')).reset_index()
    occ = occ[occ['n']>=2].copy()
    occ['gap'] = (occ['last_d']-occ['first_d'])/(occ['n']-1)
    occ['mean_spend'] = occ['tot']/occ['n']

    # EWMA-weighted spend per commodity
    ew = t.groupby(['household_key','commodity_desc']).agg(sw=('sw','sum'), w=('w','sum')).reset_index()
    ew['ew_spend'] = ew['sw']/ew['w'].replace(0, np.nan)
    ew = ew[['household_key','commodity_desc','ew_spend']]

    res = None
    for clip, tag in [(30,'c30'),(60,'c60'),(90,'c90')]:
        o = occ.copy()
        o['gap_c'] = o['gap'].clip(1, clip)
        o['next'] = o['last_d'] + o['gap_c']
        o['due'] = (o['next'] <= s+28).astype(float)
        o['contrib'] = o['due']*o['mean_spend']
        o['pressure'] = np.minimum(1.0, (s - o['last_d'])/o['gap_c'])
        agg = o.groupby('household_key').agg(**{f'cyc_pred_{tag}':('contrib','sum'), f'cyc_due_{tag}':('due','sum'), f'cyc_press_{tag}':('pressure','sum')}).reset_index()
        if tag == 'c60':
            act = o.groupby('household_key')['commodity_desc'].size().rename('cyc_active').reset_index()
            agg = agg.merge(act, on='household_key', how='outer')
            # EWMA-spend projection using c60 due flags
            o2 = o.merge(ew, on=['household_key','commodity_desc'], how='left')
            o2['contrib_ew'] = o2['due']*o2['ew_spend'].fillna(o2['mean_spend'])
            ewp = o2.groupby('household_key')['contrib_ew'].sum().rename('cyc_pred_ew').reset_index()
            agg = agg.merge(ewp, on='household_key', how='outer')
        res = agg if res is None else res.merge(agg, on='household_key', how='outer')
    res = res.set_index('household_key').reindex(hh_all)
    res['cyc_press_ratio'] = res['cyc_press_c60']/(res['cyc_active']+1.0)
    res['cyc_pred_per_active'] = res['cyc_pred_c60']/(res['cyc_active']+1.0)
    return res.fillna(0.0)

feats = agent_api.build_features(fn)
print("feats shape", feats.shape)
newcols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
print("new cols", newcols)

full = BASE.merge(feats, on=['household_key','snapshot_day'], how='left')
print("full shape", full.shape, "| NaN new feats:", int(full[newcols].isna().sum().sum()))
full[newcols] = full[newcols].fillna(0.0)
path = agent_api.save_table(full, 'e018_cycle_projection.parquet')
print("saved", path)

# sanity: correlations at s=431
tt = agent_api.train_targets()
m = full[full['snapshot_day']==431].merge(tt[tt['snapshot_day']==431], on=['household_key','snapshot_day'])
for c in newcols:
    print(c, round(m[c].corr(m['future_spend_4w']), 4))
