import pandas as pd, numpy as np, agent_api

e7 = agent_api.load_saved('e007_lagseq.parquet')
print('e007 shape:', e7.shape)
cols = list(e7.columns)
print('n cols:', len(cols))
for i in range(0, len(cols), 6):
    print('  ' + ' | '.join(cols[i:i+6]))
print('snapshots:', sorted(e7.snapshot_day.unique()))
print('rows per snapshot:', e7.groupby('snapshot_day').size().to_dict())

tt = agent_api.train_targets()
print('\ntrain_targets shape:', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac:', (tt.future_spend_4w == 0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean', 'median', 'max']))

md = agent_api.load_saved('mkt_demo.parquet')
print('\nmkt_demo shape:', md.shape)
print(list(md.columns)[:40])
