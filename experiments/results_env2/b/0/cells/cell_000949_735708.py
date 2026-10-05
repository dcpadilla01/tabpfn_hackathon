import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved("e013_stationary.parquet")
t = agent_api.load_saved("e017_reversion.parquet")
print(base.dtypes.head(3)); print(t.dtypes.head(3))
print("base hk dtype:", base.household_key.dtype, "t hk dtype:", t.household_key.dtype)
print("base sd dtype:", base.snapshot_day.dtype, "t sd dtype:", t.snapshot_day.dtype)
nan_by_col = t.isna().mean().sort_values(ascending=False)
print(nan_by_col.head(10))
m = t[t.isna().any(axis=1)]
print("rows with NaN:", len(m))
print(m.head(3))
# check overlap
k1 = set(map(tuple, base[["household_key","snapshot_day"]].drop_duplicates().values))
k2 = set(map(tuple, t[["household_key","snapshot_day"]].values))
print("in t not in base:", len(k2-k1), "in base not in t:", len(k1-k2))
