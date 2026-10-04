import agent_api as A, pandas as pd, numpy as np

dm = A.load_saved('dm_exp.parquet')
print('dm_exp:', dm.shape)
print(dm.columns.tolist())
print(dm.head(3).to_string())
if 'snapshot_day' in dm.columns:
    print(dm['snapshot_day'].value_counts().sort_index())

e11 = A.load_saved('e011_price.parquet')
print('\ne011:', e11.shape, 'snapdays:', sorted(e11.snapshot_day.unique()))
print(e11.columns.tolist())

# retailer-wide weekly spend trend (view capped at 459 - fine for exploration)
v = A.snapshot()
t = v.transactions
w = t.groupby('week_no')['sales_value'].sum()
print('\nweekly retailer spend, last 12 weeks:')
print(w.tail(12).to_string())
y = w.values
print('mean weekly spend:', y.mean(), 'first13 mean:', y[:13].mean(), 'last13 mean:', y[-13:].mean())

# train target level by snapshot (train only)
tt = A.train_targets()
print('\ntrain target by snapshot:')
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).to_string())
