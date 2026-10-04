tt = agent_api.train_targets()
print(tt.shape)
print(tt['future_spend_4w'].describe())
# correlation of target with recent spend
import numpy as np, pandas as pd
v = agent_api.snapshot()
tr = v.transactions
# recent 28d spend before day 459
recent = tr[(tr.day>=459-28)&(tr.day<=459)].groupby('household_key').sales_value.sum()
hh = agent_api.snapshot().households
print(len(hh))
df = pd.DataFrame({'hh': recent.index}).set_index('household_key')
m = tt[tt.snapshot_day==459].merge(recent.rename('recent'), left_on='household_key', right_index=True, how='left').fillna(0)
print(m[['future_spend_4w','recent']].corr())
print(m[['future_spend_4w','recent']].describe())
