
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
print('e16 dup (hh,day) pairs:', e16.duplicated(['household_key','snapshot_day']).sum())
rr = agent_api.load_saved('rawrec.parquet')
print('rawrec dup (hh,g):', rr.duplicated(['household_key','g']).sum())
print('e16 shape', e16.shape, 'unique pairs', len(e16.drop_duplicates(['household_key','snapshot_day'])))
# NaN fraction per feature group
feat = [c for c in e16.columns if c not in ('household_key','snapshot_day')]
nanfrac = e16[feat].isna().mean().sort_values(ascending=False)
print(nanfrac.head(15))
print('features with >50% NaN:', (nanfrac>0.5).sum())
