
import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved('feats_v3.parquet')
print(f3.dtypes.to_string())
tt = agent_api.train_targets()
m = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("\nmerged:", m.shape)
# local split: train snaps 151..403, local val 431
print(sorted(m.snapshot_day.unique()))
print("rows per snap:\n", m.groupby('snapshot_day').size())
print("\nmean target by snap:\n", m.groupby('snapshot_day').future_spend_4w.agg(['mean','median']))
