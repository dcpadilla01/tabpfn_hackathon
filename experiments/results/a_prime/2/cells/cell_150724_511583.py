import agent_api as A, pandas as pd, numpy as np
v = A.snapshot(95)
print("day", v.day, "week", v.week)
print(type(v.households))
print(v.households.head(3))
print(v.households.shape, v.households.index.name)
t = v.transactions
print(t.shape, t['day'].max(), t['week_no'].max())
print(t.columns.tolist())
# quick check: weekly spend grouping
cur_week = (95+8)//7
print("cur_week", cur_week)
s = t[t.week_no==cur_week-1].groupby('household_key')['sales_value'].sum()
print(s.head(3), s.shape)