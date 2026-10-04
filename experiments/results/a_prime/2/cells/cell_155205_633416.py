
import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()
b = agent_api.baseline_features()
print('baseline cols:', b.columns.tolist())
print('baseline dtypes:', b.dtypes.to_dict())
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
for c in b.columns:
    if c in ('household_key','snapshot_day'): continue
    print(c, 'train_nan%', round(tr[c].isna().mean(),3), 'val_nan%', round(va[c].isna().mean(),3), 'dtype', b[c].dtype)
