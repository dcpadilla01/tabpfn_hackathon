
import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e004_features.parquet")
print(list(feats.columns))
print(feats.snapshot_day.value_counts().sort_index())
