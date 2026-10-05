
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot()
tx = v.table('transactions')
print(tx.shape)
print(tx[['sales_value','coupon_match_disc','coupon_disc','retail_disc','quantity']].describe().round(3).to_string())
print('neg frac:', {c: float((tx[c]<0).mean()) for c in ['sales_value','coupon_match_disc','coupon_disc','retail_disc']})
print('zero sales frac', float((tx.sales_value==0).mean()))
print('households in tx:', tx.household_key.nunique())
print('day range', tx.day.min(), tx.day.max())
