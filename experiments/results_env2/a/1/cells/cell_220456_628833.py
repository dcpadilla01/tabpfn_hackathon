import numpy as np, pandas as pd, xgboost as xgb

fe = agent_api.load_saved("e002_features.parquet").drop(columns=["index"])
new = agent_api.load_saved("e004_features.parquet")
# new table was saved AFTER the merge attempt? check columns
print("e004 cols sample:", [c for c in new.columns if c.startswith("spend")][:10])
print(new.shape)