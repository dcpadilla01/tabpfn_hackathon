import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e012_robust.parquet')
print(t.shape)
print(t.dtypes.value_counts())
print(t.columns.tolist()[:50])
print('has keys:', 'household_key' in t.columns, 'snapshot_day' in t.columns)
print(t['snapshot_day'].value_counts().sort_index())
print('dup rows:', t.duplicated(['household_key','snapshot_day']).sum())
