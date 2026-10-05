import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
prod = agent_api.snapshot(459).products
print(prod.commodity_desc.value_counts().head(40))
print('---curr size sample---')
print(prod.curr_size_of_product.head(10).tolist())
t = agent_api.snapshot(459).transactions
print('trans', t.shape)
m = t.merge(prod[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
print('top commodities by sales:')
print(m.groupby('commodity_desc').sales_value.sum().sort_values(ascending=False).head(20))
