import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
tr = v.transactions
print(tr.shape)
print(tr.groupby('household_key').day.agg(['min','max','count']).describe())
# gap structure: days between baskets per household
g = tr.sort_values(['household_key','day']).groupby('household_key').day.apply(lambda s: s.drop_duplicates().diff().dropna())
print(g.describe())
