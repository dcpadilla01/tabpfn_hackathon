
import agent_api, pandas as pd, numpy as np
f4 = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print([c for c in f4.columns if 'future' in c or 'spend_4w' in c])
m = f4.merge(tt, on=['household_key','snapshot_day'], how='left')
print(m.shape)
print([c for c in m.columns if 'future' in c])
print(m.future_spend_4w.isna().sum())
