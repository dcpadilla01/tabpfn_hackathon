import agent_api as A
tr = A.snapshot().transactions
g = tr.groupby('household_key')['sales_value'].agg(['sum','count','nunique'])
print(g.describe())
# spend per 28d window per household
tr['w'] = (tr['day']-1)//28
w = tr.groupby(['household_key','w'])['sales_value'].sum()
print(w.describe())
print("share of windows with zero spend:", (w==0).mean())
# how many households have a row at a snapshot? rule: first purchase >= 84 days earlier
first = tr.groupby('household_key')['day'].min()
print(first.describe())
