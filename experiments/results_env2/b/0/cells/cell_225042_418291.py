import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
print(feats.dtypes.value_counts())
print(feats.snapshot_day.unique()[:20])
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape, df.snapshot_day.dtype)
print(df.snapshot_day.unique())
