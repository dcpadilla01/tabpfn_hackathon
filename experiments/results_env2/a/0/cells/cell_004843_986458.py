
import pandas as pd, numpy as np, agent_api
f4 = agent_api.load_saved("feats_v4.parquet")
print(f4.shape)
print(list(f4.columns))
print(f4.head(3).T.head(60))
