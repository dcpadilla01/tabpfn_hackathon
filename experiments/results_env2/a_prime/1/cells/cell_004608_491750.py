
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
print('e16 n feat:', len(feat), 'max allowed 500 -> ok')
# NaN fraction of the 58 rawrec-only cols
rr = agent_api.load_saved('rawrec.parquet')
rronly = [c for c in rr.columns if c not in k+['g']]
nanfrac = e16[rronly].isna().mean().sort_values(ascending=False)
print('rawrec-only cols:', len(rronly))
print(nanfrac.head(10))
print('rawrec cols with any NaN in e16:', (nanfrac>0).sum())
# where do NaNs come from? rows in e16 not in rawrec merge
m = e16.merge(rr.rename(columns={'g':'snapshot_day'}), on=k, how='left', indicator=True)
print('e16 rows missing from rawrec:', (m['_merge']=='left_only').sum())
print(m.loc[m['_merge']=='left_only'].groupby('snapshot_day').size())
