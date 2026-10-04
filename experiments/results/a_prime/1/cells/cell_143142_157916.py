import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
print(type(v.households))
print(v.households if not hasattr(v.households,'shape') else (v.households.shape, v.households.columns.tolist() if hasattr(v.households,'columns') else ''))
b = agent_api.baseline_features()
print(b.shape, b.columns.tolist())
print(b.head(5).T)
print(v.demographics.head(3))
print(v.transactions.columns.tolist(), v.transactions.shape)
