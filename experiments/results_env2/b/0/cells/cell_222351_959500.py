import agent_api as api, pandas as pd, numpy as np
for name in ["e006_zero_inflation.parquet","e006_newblock.parquet","e007_ar_lags.parquet","e008_log_transform.parquet"]:
    t = api.load_saved(name)
    print(name, t.shape, "dup:", t.duplicated(["household_key","snapshot_day"]).sum())
b = api.load_saved("e006_zero_inflation.parquet")
print(b.columns.tolist())
