
import pandas as pd, numpy as np

t15 = agent_api.load_saved('e015_stack.parquet')
print("E015 shape:", t15.shape)
print("E015 cols:", t15.columns.tolist())
print("\nrows with any NaN per snapshot_day (E015):")
print(t15.groupby('snapshot_day').apply(lambda g: int(g.isna().any(axis=1).sum())))

t12 = agent_api.load_saved('e012_style.parquet')
print("\nE012 shape:", t12.shape)
print("rows ALL-NaN per snapshot_day (E012):")
print(t12.groupby('snapshot_day').apply(lambda g: int(g.isna().all(axis=1).sum())))

tt = agent_api.train_targets()
print("\ntrain_targets:", tt.shape, tt.columns.tolist())
print(tt.head(3))
print("target describe:\n", tt[agent_api.TARGET].describe())
