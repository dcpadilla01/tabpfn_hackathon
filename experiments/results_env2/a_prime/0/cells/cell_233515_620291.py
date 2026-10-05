import agent_api, pandas as pd, numpy as np
e13 = agent_api.load_saved("e013_storeprod.parquet")
b2 = agent_api.load_saved("e016_batch2.parquet")
merged = e13.merge(b2, on=["household_key","snapshot_day"], how="inner")
print("merged:", merged.shape)
agent_api.save_table(merged, "e016_merged.parquet")
