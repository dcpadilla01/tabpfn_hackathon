
import pandas as pd, numpy as np
v = agent_api.snapshot(459)
p = v.products
print('products rows', len(p), 'unique pid', p['product_id'].nunique())
print(p['product_id'].value_counts().head())
t = v.transactions
print('tx unique pid', t['product_id'].nunique())
print('tx unique baskets', t['basket_id'].nunique())
# check a sample household commodity count without merge
h = t['household_key'].value_counts().index[0]
th = t[t.household_key==h]
print('hh', h, 'rows', len(th), 'commodities via merge dup?', th['product_id'].nunique())
