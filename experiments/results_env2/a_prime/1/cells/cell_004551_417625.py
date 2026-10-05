
import agent_api, pandas as pd, numpy as np
e16 = agent_api.load_saved('e016_merged.parquet')
e15 = agent_api.load_saved('e015_stack.parquet')
k = ['household_key','snapshot_day']
e12c = [c for c in e15.columns if c not in k+['stack_ridge_log','stack_ridge']]
m = e16.merge(e15[k+e12c+['stack_ridge_log']], on=k, suffixes=('_x','_y'))
bad = {}
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if pd.api.types.is_numeric_dtype(a):
        d = (a-b).abs().max()
        if d > 1e-6: bad[c]=d
print('numeric cols differing:', len(bad), list(bad.items())[:20])
# categorical compare
for c in e12c:
    a, b = m[c+'_x'], m[c+'_y']
    if not pd.api.types.is_numeric_dtype(a):
        neq = (a.astype(str)!=b.astype(str)).sum()
        if neq>0: print('cat diff', c, neq)
print('done')
