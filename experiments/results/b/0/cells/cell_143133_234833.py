import agent_api, numpy as np, pandas as pd
snap = agent_api.snapshot()
tr = snap.transactions
print(tr.dtypes)
print(tr.head(3))
# check one household history
h = agent_api.history(tr.household_key.iloc[0])
print(type(h), h.shape)
print(h.head(5))
print("basket sizes:")
bs = tr.groupby('basket_id').sales_value.sum()
print(bs.describe())
