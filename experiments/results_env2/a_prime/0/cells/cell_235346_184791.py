
import pandas as pd, numpy as np
key = ['household_key','snapshot_day']
base = load_saved('e013_storeprod.parquet')
b16  = load_saved('e016_batch2.parquet')
b17  = load_saved('e017_phase.parquet')
b18  = load_saved('e018_dyn.parquet')

m = base.merge(b16, on=key, how='inner', suffixes=('','_x'))
m = m.merge(b17, on=key, how='inner')
m = m.merge(b18, on=key, how='inner')
print('union shape', m.shape)
print('dup cols check:', [c for c in m.columns if c.endswith('_x')][:5])
assert len(m) == 36426
m = m.sort_values(key).reset_index(drop=True)
path = save_table(m, 'e019_union')
print(path)
print('n features:', m.shape[1]-2)
print(m[key].head(3))
