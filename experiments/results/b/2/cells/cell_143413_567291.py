import numpy as np, pandas as pd
from agent_api import snapshot
v = snapshot()
tx = v.transactions
print('tx days', tx.day.min(), tx.day.max())
# spend by department overall
p = v.products
t = tx.merge(p[['product_id','department','brand']], on='product_id', how='left')
print(t.department.value_counts().head(15))
print('null dept frac', t.department.isna().mean())
# discount cols
print(tx[['coupon_match_disc','coupon_disc','retail_disc','quantity','sales_value']].describe())
# per-household dept shares
g = t.groupby(['household_key','department']).sales_value.sum()
hh = t.groupby('household_key').sales_value.sum()
sh = (g/hh).unstack()
print(sh.shape)
print(sh.head())
