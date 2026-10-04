import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(431)
print('day', v.day, 'week', v.week)
hh = v.households
print('households type', type(hh), hh.shape if hasattr(hh,'shape') else len(hh))
print(hh.head())
tx = v.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print('demographics', v.demographics.shape)
print('n households with tx', tx.household_key.nunique())
sd = agent_api.snapshot_days(); print(sd)
print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)