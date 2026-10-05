
import agent_api, numpy as np, pandas as pd
tt = agent_api.train_targets()
df = agent_api.load_saved("e005_preds.parquet")
print(df.shape, list(df.columns))
print(df.head())
m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
print(m.shape, list(m.columns))
print(m.head())
