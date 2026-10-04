import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(431)
hh = v.households
print(type(hh), hh)
tx = v.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print('demographics', v.demographics.shape)
print('n hh with tx', tx.household_key.nunique())
print('KEYS', agent_api.KEYS, 'TARGET', agent_api.TARGET)
print('snapshot_days', agent_api.snapshot_days())