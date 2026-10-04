import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
demo = agent_api.snapshot().demographics
print("mkt hh dtype:", df.household_key.dtype, "demo hh dtype:", demo.household_key.dtype, flush=True)
print("demo sample keys:", demo.household_key.head(3).tolist(), " df keys:", df.household_key.head(3).tolist(), flush=True)
m = df.merge(demo, on='household_key', how='left')
print("merged rows:", len(m), "non-null class1:", m.classification_1.notna().sum(), flush=True)
# how many df households in demo
overlap = df.household_key.isin(demo.household_key).mean()
print("frac of rows with demo:", overlap, flush=True)
