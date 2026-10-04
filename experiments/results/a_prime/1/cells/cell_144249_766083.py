import agent_api as api
import pandas as pd, numpy as np
base = api.load_saved('e001_txhist.parquet')
mkt = api.load_saved('e004_mkt.parquet')
print(base.index.name, base.columns.tolist()[:3])
print(mkt.index.name, mkt.columns.tolist()[:3])
if 'household_key' not in base.columns:
    base = base.reset_index()
if 'household_key' not in mkt.columns:
    mkt = mkt.reset_index()
merged = base.merge(mkt, on=['household_key','snapshot_day'], how='left', suffixes=('','_m'))
print(merged.shape)
print(merged.isna().mean().max())
path = api.save_table(merged, 'e004_mkt_full.parquet')
print(path)