import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
tx = s.table('transactions')
prod = s.table('products')
m = tx.merge(prod[['product_id','department']], on='product_id', how='left')
print(m.department.value_counts().head(25).to_string())
# discount columns sign
print(tx[['sales_value','coupon_disc','coupon_match_disc','retail_disc']].describe().loc[['mean','min','max']].to_string())
