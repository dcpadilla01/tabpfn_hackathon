
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
print([c for c in e15.columns if 'stack' in c])
print([c for c in e16.columns if 'stack' in c])
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in k+['stack_ridge_log','stack_ridge_lin']]
m = e16.merge(e15[k+e12c+['stack_ridge_log']], on=k, suffixes=('_x','_y'))
print('merged', m.shape)
bad = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if np.issubdtype(a.dtype, np.number):
        d = (a-b).abs().max()
        if d > 1e-6: bad[c]=d
print('cols differing:', len(bad), list(bad.items())[:20])
# check stack col present in e16
print('stack in e16:', [c for c in e16.columns if 'stack' in c])
