
import agent_api as A
import pandas as pd
e13 = A.load_saved('e013_storeprod.parquet')
b1 = A.load_saved('cand_batch1.parquet')
print('b1 dup keys:', b1.duplicated(['household_key','snapshot_day']).sum())
m = e13.merge(b1.drop(columns=['snapshot_day']), on=['household_key','snapshot_day'], how='left')
print(m.shape)
assert m[['household_key','snapshot_day']].equals(e13[['household_key','snapshot_day']])
print('added:', [c for c in m.columns if c not in e13.columns])
print(m[[c for c in m.columns if c not in e13.columns]].isna().mean().round(3))
p = A.save_table(m, 'e014_batch1.parquet')
print(p)
