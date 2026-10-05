import pandas as pd, numpy as np, agent_api
f = agent_api.load_saved("feats_v1.parquet")
print(f.shape)
print(f.columns.tolist())
print(f.head(3))
t = agent_api.train_targets()
print(t.shape, t.future_spend_4w.describe())
