import agent_api as A
import pandas as pd, numpy as np

v3 = A.load_saved("feats_v3.parquet")
se = A.load_saved("feats_seasonal.parquet")
print("v3", v3.shape, v3.columns.tolist()[:12], "...")
print("se", se.shape, se.columns.tolist())
tt = A.train_targets()
print("train rows", tt.shape)
# check val/train split of feature tables
print(v3.snapshot_day.value_counts().sort_index())
print(se.snapshot_day.value_counts().sort_index())
# overlap of keys
k_v3 = set(map(tuple, v3[["household_key","snapshot_day"]].values))
k_se = set(map(tuple, se[["household_key","snapshot_day"]].values))
print("v3==se keys:", k_v3==k_se)
print("NaN share in seasonal cols:")
print(se.drop(columns=["household_key","snapshot_day"]).isna().mean().round(3))
