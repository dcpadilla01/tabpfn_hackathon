
import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot()  # capped at day 459
tx = v.table('transactions')
print("tx shape:", tx.shape, "day range:", tx.day.min(), tx.day.max())
tx['week_no'] = ((tx.day+8)//7).astype(int)
wk = tx.groupby('week_no').agg(spend=('sales_value','sum'), hh=('household_key','nunique'))
wk['spend_per_hh'] = wk.spend/wk.hh
print("\nweekly spend per household, weeks 1..66:")
print(wk.spend_per_hh.round(1).to_string())
