import agent_api, pandas as pd, numpy as np
a = agent_api.load_saved('e012_dorm.parquet'); b = agent_api.load_saved('e016_smoothed.parquet')
ca, cb = set(a.columns)-{'household_key','snapshot_day'}, set(b.columns)-{'household_key','snapshot_day'}
print('e012 feats', len(ca), 'e016 feats', len(cb))
print('shared', len(ca&cb), 'only e012', sorted(ca-cb), 'only e016', len(cb-ca))
m = a.merge(b, on=['household_key','snapshot_day'], suffixes=('_a','_b'))
for c in sorted(ca&cb):
    x, y = m[c+'_a'], m[c+'_b']
    if x.dtype.kind in 'ifb' and y.dtype.kind in 'ifb':
        d = (x.fillna(-9e9)-y.fillna(-9e9)).abs().max()
        if d > 1e-9: print('DIFF', c, d)
print('merge shape', m.shape, '-> union feats', len(ca|cb))