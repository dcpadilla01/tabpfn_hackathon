import numpy as np, pandas as pd
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
print('t16', t16.shape, 't8', t8.shape, 'tt', tt.shape)
gcols = [c for c in t16.columns if c.startswith('g_')]
print('grid cols:', gcols)
print('t8 cols subset of t16:', set(t8.columns) <= set(t16.columns))
df = t16.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
y = df['future_spend_4w'].values
print('target mean %.2f std %.2f zero-share %.3f median %.2f' % (y.mean(), y.std(), (y==0).mean(), np.median(y)))
tr_days = agent_api.snapshot_days()['train']
print('train days', tr_days)
