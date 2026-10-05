
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
v = allF[allF.snapshot_day>=459]
print("val rows in allF:", v.shape)
print(v[['household_key','snapshot_day','future_spend_4w']].head(10))
print("nan count val:", v['future_spend_4w'].isna().sum())
tr = allF[allF.snapshot_day<459]
print("nan count train:", tr['future_spend_4w'].isna().sum())
