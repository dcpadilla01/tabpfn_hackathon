import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
print(base.index.dtype, base.columns[:3].tolist(), base.shape)
print(tt.head(3))
m = base.merge(tt, on=['household_key','snapshot_day'], how='left')
print(m.shape, m.future_spend_4w.isna().sum())
print(m[m.snapshot_day==431].future_spend_4w.describe())
