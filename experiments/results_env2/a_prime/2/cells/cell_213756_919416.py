import pandas as pd, numpy as np
v = agent_api.snapshot()
tx = v.table('transactions')
print("tx rows@459:", len(tx), "max day:", tx.day.max(), "n hh:", tx.household_key.nunique())
# seasonality: total spend by 4-week block
tx['blk'] = ((tx.day-1)//28)
s = tx.groupby('blk').sales_value.sum()
print(s.round(0).to_string())
