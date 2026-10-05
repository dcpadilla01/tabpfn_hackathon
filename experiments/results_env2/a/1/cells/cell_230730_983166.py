
import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e005_newfeats.parquet")
print(list(feats.columns))
print(feats.head(3).T)
