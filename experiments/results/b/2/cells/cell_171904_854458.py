import agent_api as A
import pandas as pd, numpy as np

s = A.load_saved('e014_stack.parquet')
print(s.columns.tolist()[-8:])
print(s[['household_key','snapshot_day', s.columns[-1]]].head())
print(s[s.columns[-1]].describe())

g = A.load_saved('e014_gbm_oob.parquet')
print(g.columns.tolist()[-3:])
print(g['gbm_corr'].describe())
