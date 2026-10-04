import agent_api as A, pandas as pd
r = A.load_saved('rhythm_v1.parquet')
base = A.load_saved('e009_demo.parquet')
m = base.merge(r, on=['household_key','snapshot_day'], how='inner', suffixes=('','_dup'))
print('merged:', m.shape)
print('dup cols:', [c for c in m.columns if c.endswith('_dup')])
A.save_table(m, 'e010_rhythm.parquet')
print('saved ok')