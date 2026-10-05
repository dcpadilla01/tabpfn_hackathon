
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in ('household_key','snapshot_day','stack_ridge_log','stack_ridge_lin')]
m = e16.merge(e15[k+e12c+['stack_ridge_log','stack_ridge_lin']], on=k, suffixes=('_x','_y'))
print('merged', m.shape)
diffs = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if np.issubdtype(a.dtype, np.number):
        d = (a-b).abs()
        diffs[c] = d.max()
bad = {c:v for c,v in diffs.items() if v > 1e-6}
print('cols differing:', len(bad))
print(list(bad.items())[:20])
