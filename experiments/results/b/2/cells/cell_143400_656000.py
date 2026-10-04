import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, KEYS, TARGET, snapshot_days

e = load_saved('e001_recent_spend.parquet')
print(e.shape)
print(e.columns.tolist())
tt = train_targets()
print(tt[TARGET].describe())
print('zero frac:', (tt[TARGET]==0).mean())
m = tt.merge(e, on=KEYS, how='left')
print(m.shape, 'missing:', m[e.columns.tolist()[2:]].isna().sum().sum())
num = m.select_dtypes(include=[np.number])
corr = num.corr()[TARGET].drop(TARGET).sort_values()
print(corr)
print(snapshot_days())
