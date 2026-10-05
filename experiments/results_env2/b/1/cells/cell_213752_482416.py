
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
print(e2.dtypes.head(5))
print(e2['snapshot_day'].dtype, e2['snapshot_day'].unique()[:20])
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
print(m['snapshot_day'].dtype, m['snapshot_day'].unique())
print(agent_api.snapshot_days())
print("vd count:", m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).sum())
print("rows:", len(m), "targets:", len(t))
