
import agent_api as A
import pandas as pd, numpy as np

v = A.snapshot()  # capped at day 459
tx = v.transactions
# population weekly per-household spend
wk = tx.groupby('week_no').agg(total=('sales_value','sum'), hh=('household_key','nunique'))
wk['per_hh'] = wk['total']/wk['hh']
print("weeks:", wk.index.min(), wk.index.max())
print(wk['per_hh'].describe())
# YoY overlap: year1 weeks 1-14 vs year2 weeks 53-66
y1 = wk.loc[1:14,'per_hh'].values; y2 = wk.loc[53:66,'per_hh'].values
print("year1 w1-14:", np.round(y1,1))
print("year2 w53-66:", np.round(y2,1))
print("corr:", np.corrcoef(y1,y2)[0,1], " ratio y2/y1 mean:", (y2/y1).mean())
# full series rounded
print(np.round(wk['per_hh'].values,1))
