import agent_api, pandas as pd, numpy as np
f3 = agent_api.load_saved('feats_v3.parquet')
print(f3.shape)
print(list(f3.columns))
tt = agent_api.train_targets()
print(tt.shape, tt.head())
print(tt.future_spend_4w.describe())
