
import agent_api, pandas as pd, numpy as np
f4 = agent_api.load_saved('feats_v4.parquet')
print("feats_v4 shape:", f4.shape)
print([c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns])
