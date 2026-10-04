import pandas as pd, numpy as np
api = agent_api
print(api.snapshot_days())
tx = api.snapshot().transactions
print('tx shape', tx.shape, 'day range', tx.day.min(), tx.day.max())
print(tx[['sales_value','quantity']].describe())
print('n households', tx.household_key.nunique())
bf = api.baseline_features()
print('baseline', bf.shape, bf.columns.tolist())
print(bf.head(3))
tt = api.train_targets()
print('targets', tt.shape); print(tt.head())
v = api.snapshot(95)
print('households type', type(v.households), 'n', len(v.households))
print('view day/week', v.day, v.week)
print('has demographics?', hasattr(v, 'demographics'))
