
import pandas as pd, numpy as np
v = agent_api.snapshot()
p = v.products
print(p.shape); print(p.columns.tolist())
t = v.transactions
print(t.shape)
# top commodities by spend
m = t.merge(p[['product_id','commodity_desc','department','brand']], on='product_id', how='left')
g = m.groupby('commodity_desc')['sales_value'].sum().sort_values(ascending=False)
print(g.head(20))
print('n commodities', g.shape[0])
print(m['brand'].value_counts(dropna=False).head())
# unit price distribution
m['unit_price'] = m['sales_value']/m['quantity'].replace(0,np.nan)
print(m['unit_price'].describe())
