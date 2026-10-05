import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
print('e15 shape:', e15.shape)
scols = [c for c in e15.columns if 'stack' in c.lower()]
print('stack cols:', scols)
print(e15[scols].describe())

tt = api.train_targets()
print('\ntrain_targets:', tt.shape)
print(tt.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share:', (tt.future_spend_4w==0).mean())

base = api.baseline_features()
print('\nbaseline rows:', base.shape)
k15 = set(map(tuple, e15[['household_key','snapshot_day']].values))
kb = set(map(tuple, base[['household_key','snapshot_day']].values))
print('e15 missing rows vs baseline:', len(kb-k15), 'extra:', len(k15-kb))

# candidate columns for peer median base + recency
cands = [c for c in e15.columns if ('ew' in c.lower()) or ('spend_28' in c) or ('sp28' in c) or ('recen' in c.lower())]
print('\ncandidate cols:', cands[:40])
print('\ndtypes:', e15.dtypes.value_counts().to_dict())
print('snapshot days present:', sorted(e15.snapshot_day.unique()))
