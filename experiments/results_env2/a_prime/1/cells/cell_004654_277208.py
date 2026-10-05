
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
isval = e16['snapshot_day'].isin([459,487,515,543])
# how many validation rows have ANY NaN among the rawrec cols?
rrcols = [c for c in e16.columns if c in agent_api.load_saved('rawrec.parquet').columns and c not in k]
any_nan = e16.loc[isval, rrcols].isna().any(axis=1)
print('validation rows with any NaN in rawrec cols:', any_nan.sum(), 'of', isval.sum())
print(e16.loc[isval].groupby('snapshot_day').size())
print('frac', any_nan.mean())
# train rows with any NaN
antr = e16.loc[~isval, rrcols].isna().any(axis=1)
print('train rows with any NaN in rawrec cols:', antr.sum(), 'of', (~isval).sum())
