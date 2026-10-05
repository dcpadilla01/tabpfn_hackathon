import agent_api, pandas as pd, numpy as np
f4 = agent_api.load_saved('feats_v4.parquet')
print(f4.shape)
print(f4.columns.tolist())
print(f4.head(3).T)
