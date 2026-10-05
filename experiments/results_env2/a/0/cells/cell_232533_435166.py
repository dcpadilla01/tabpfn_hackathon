
import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", f3.shape)
print("cols:", list(f3.columns))
print(f3.head(3))

tt = agent_api.train_targets()
print("\ntrain_targets shape:", tt.shape)
print(tt.head(3))
print(tt.future_spend_4w.describe())

p7 = agent_api.load_saved('pred_e007.parquet')
print("\npred_e007 shape:", p7.shape, "cols:", list(p7.columns))
print(p7.head(3))
