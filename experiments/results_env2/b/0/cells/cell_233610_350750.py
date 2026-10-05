import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
print('households repr:', repr(s.households)[:200])
print('day', s.day, 'week', s.week)
tx = s.table('transactions')
print('tx shape', tx.shape)
print(tx.head(3))
print('n hh', tx.household_key.nunique())
# check a household history
h = agent_api.history(tx.household_key.iloc[0])
print('hist shape', h.shape)
print(h.head(3))
