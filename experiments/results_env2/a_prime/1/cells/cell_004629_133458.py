
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
feat = [c for c in e16.columns if c not in k]
print('e16 n feat:', len(feat))
# NaN fraction per feature for validation-only rows vs train rows
isval = e16['snapshot_day'].isin([459,487,515,543])
nanval = e16.loc[isval, feat].isna().mean()
nantr  = e16.loc[~isval, feat].isna().mean()
d = (nanval-nantr).sort_values(ascending=False)
print('cols with much higher NaN on validation:')
print(d.head(12))
# any feature 100% NaN on validation?
print('100% NaN on validation:', (nanval==1).sum())
print(list(nanval[nanval==1].index)[:20])
