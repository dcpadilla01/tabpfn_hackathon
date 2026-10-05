
import agent_api, pandas as pd, numpy as np
e12 = agent_api.load_saved('e012_style.parquet')
e16 = agent_api.load_saved('e016_merged.parquet')
feat12 = [c for c in e12.columns if c not in ('household_key','snapshot_day')]
allnan = e12[feat12].isna().all(axis=1)
print('e12 rows with ALL features NaN:', allnan.sum())
print(e12.loc[allnan].groupby('snapshot_day').size().head(20))
# compare e16 rawrec cols vs e12 on overlapping non-nan rows
rr = agent_api.load_saved('rawrec.parquet')
common = [c for c in rr.columns if c in e16.columns and c not in ('household_key','g')]
print('n common rawrec cols in e16:', len(common))
sub = e16.merge(rr.rename(columns={'g':'snapshot_day'}), on=['household_key','snapshot_day'], suffixes=('_x','_r'))
print('merged rows', len(sub), 'of', len(e16))
for c in ['sp28','sp56','ew'][:3]:
    if c+'_x' in sub.columns and c+'_r' in sub.columns:
        d = (sub[c+'_x']-sub[c+'_r']).abs()
        print(c, 'median abs diff e12vsrawrec:', d.median(), 'max', d.max())
