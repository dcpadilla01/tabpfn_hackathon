
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
print('t dtypes sample:', t.dtypes.value_counts().to_dict())
print('t index:', t.index.name, t.index[:3].tolist())
print('tt dtypes:', tt.dtypes.to_dict())
print('tt head:\n', tt.head(3))
print('dup keys in t:', t.duplicated(['household_key','snapshot_day']).sum())
print('dup keys in tt:', tt.duplicated(['household_key','snapshot_day']).sum())
m = t.merge(tt, on=['household_key','snapshot_day'])
print('len t', len(t), 'len m', len(m))
# per-snapshot counts
print(t.groupby('snapshot_day').size().to_string())
print(m.groupby('snapshot_day')['future_spend_4w'].count().to_string())
print('y nan in m:', m['future_spend_4w'].isna().sum())
