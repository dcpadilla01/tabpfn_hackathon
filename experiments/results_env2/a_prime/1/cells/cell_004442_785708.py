
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
feat = [c for c in e16.columns if c not in ('household_key','snapshot_day')]
allnan = e16[feat].isna().all(axis=1)
print('e16 rows with ALL features NaN:', allnan.sum())
print(e16.loc[allnan].groupby('snapshot_day').size())
# count fully-nan per snapshot
cnt = e16.groupby('snapshot_day').apply(lambda d: d[feat].isna().all(axis=1).sum())
print(cnt)
