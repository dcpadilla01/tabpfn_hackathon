
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
v = agent_api.snapshot(459)
prods = v.products
print(prods.department.value_counts().head(20))
print(prods.brand.value_counts())
tx = v.transactions
print(tx.shape)
print(tx.head(3))
print('neg sales_value share %.4f' % (tx.sales_value<0).mean())
print('zero sales share %.4f' % (tx.sales_value==0).mean())
h = agent_api.history(tx.household_key.iloc[0], 459)
print(h.head(3))
