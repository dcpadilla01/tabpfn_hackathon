
import agent_api, pandas as pd, numpy as np
e12 = agent_api.load_saved('e012_style.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
rr  = agent_api.load_saved('rawrec.parquet')
print('e12', e12.shape, 'e15', e15.shape, 'rr', rr.shape)
print('e12 per snapshot:\n', e12.groupby('snapshot_day').size())
print('e15 per snapshot:\n', e15.groupby('snapshot_day').size())
print('rr per snapshot:\n', rr.groupby('snapshot_day').size())
print('e12 cols sample:', list(e12.columns)[:12])
print('e15 extra cols:', [c for c in e15.columns if c not in e12.columns])
print('rr cols:', list(rr.columns)[:40])
