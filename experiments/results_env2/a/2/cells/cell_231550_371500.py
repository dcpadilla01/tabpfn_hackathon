
import pandas as pd, numpy as np, agent_api

f = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", f.shape)
print("cols:", list(f.columns))
print(f.head(3))

t = agent_api.train_targets()
print("\ntrain_targets:", t.shape)
print(t['future_spend_4w'].describe())
print("zero share:", (t['future_spend_4w']==0).mean())
print(t['snapshot_day'].value_counts().sort_index())
print("\nsnapshot_days:", agent_api.snapshot_days())
