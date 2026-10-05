import agent_api as api, pandas as pd, numpy as np
v = api.snapshot()
print(type(v.households), v.households if v.households is None else len(v.households))
print([a for a in dir(v) if not a.startswith('_')])
print('day', v.day, 'week', v.week)
tt = api.train_targets()
print('n hh per day:'); print(tt.groupby('snapshot_day').household_key.nunique())
# how many val rows expected?
OLD = api.load_saved('feats_v2.parquet')
print(OLD.groupby('snapshot_day').size())
print(OLD[OLD.snapshot_day>=459].shape)
