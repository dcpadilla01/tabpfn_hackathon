import agent_api as api
import pandas as pd, numpy as np
base = api.load_saved('e001_txhist.parquet')
mkt = api.load_saved('e004_mkt.parquet')
drop = ['household_key','snapshot_day']
merged = base.merge(mkt.drop(columns=drop), on=['household_key','snapshot_day'], how='left')
print(merged.shape, merged.columns.shape)
print(merged.isna().mean().max())
path = api.save_table(merged, 'e004_mkt_full.parquet')
print(path)