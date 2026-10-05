import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(200)
tx = v.table('transactions')
print(tx[['sales_value','retail_disc','coupon_disc','coupon_match_disc','quantity']].describe().round(2))
print(tx.shape)
