import agent_api as api, pandas as pd, numpy as np
tt = api.train_targets()
print("targets:", tt.shape, tt.columns.tolist())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1))
allF = api.load_saved('allF.parquet')
print("allF:", allF.shape)
print("cols:", list(allF.columns))
print(allF.groupby('snapshot_day').size())
