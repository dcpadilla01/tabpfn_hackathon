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
