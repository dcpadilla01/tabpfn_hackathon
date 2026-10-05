import numpy as np, pandas as pd, agent_api
t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
print('shapes', t8.shape, t16.shape, tt.shape)
c8 = [c for c in t8.columns if c not in ('household_key','snapshot_day')]
c16 = [c for c in t16.columns if c not in ('household_key','snapshot_day')]
print('E008(%d):' % len(c8)); print(c8)
print('E016(%d):' % len(c16)); print(c16)
y = tt.future_spend_4w
print('tgt mean %.2f med %.2f zero%% %.3f p90 %.1f p99 %.1f max %.1f' % (y.mean(), y.median(), (y==0).mean(), y.quantile(.9), y.quantile(.99), y.max()))
print(t8.dtypes.value_counts().to_dict())
print('rows/snap:'); print(tt.groupby('snapshot_day').size().to_dict())
