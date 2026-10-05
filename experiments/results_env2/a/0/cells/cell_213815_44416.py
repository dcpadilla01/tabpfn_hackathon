import numpy as np, pandas as pd
print(api.snapshot_days())
tt = api.train_targets()
print('targets', tt.shape, tt.columns.tolist())
print(tt.future_spend_4w.describe())
print('zero frac', (tt.future_spend_4w==0).mean())
bf = api.baseline_features()
print('baseline', bf.shape, bf.columns.tolist())
v = api.snapshot()
print('day', v.day, 'week', v.week)
tr = v.transactions
print('txn', tr.shape, 'hh', tr.household_key.nunique())
print(tr.head(3).to_string())
hh = v.households
print('households type', type(hh), len(hh))
h0 = list(hh)[0]
h = api.history(h0)
print('hist', h.shape)
print(h.head(3).to_string())
d = v.table('demographics')
print('demo', d.shape)
print(api.describe_tables())
