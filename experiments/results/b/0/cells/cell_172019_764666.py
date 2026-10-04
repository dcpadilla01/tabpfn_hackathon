import agent_api as api, pandas as pd, numpy as np, re
v = api.snapshot()
p = v.products
print(p.shape, list(p.columns))
print(p['brand'].value_counts(dropna=False).head())
print(p['curr_size_of_product'].dropna().sample(15, random_state=0).tolist())
tx = v.transactions
mg = tx.merge(p[['product_id','department','brand','curr_size_of_product']], on='product_id', how='left')
dept_spend = mg.groupby('department')['sales_value'].sum().sort_values(ascending=False)
print('top12 depts:', list(dept_spend.head(12).index))
print('brand na in tx share:', mg['brand'].isna().mean())
def sz(s):
    if not isinstance(s,str): return np.nan
    m = re.search(r'(\d+(\.\d+)?)', s)
    return float(m.group(1)) if m else np.nan
sizes = p['curr_size_of_product'].map(sz)
print('size parsed frac:', sizes.notna().mean(), 'p75:', sizes.quantile(0.75), 'median:', sizes.median())
print('sample sizes:', sizes.dropna().sample(10, random_state=1).tolist())