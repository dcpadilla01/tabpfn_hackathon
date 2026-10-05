
import agent_api as A
import pandas as pd
e13 = A.load_saved('e013_storeprod.parquet')
b1 = A.load_saved('cand_batch1.parquet')
m = e13.merge(b1, on=['household_key','snapshot_day'], how='left')
print(m.shape)
assert m[['household_key','snapshot_day']].equals(e13[['household_key','snapshot_day']])
newc = [c for c in m.columns if c not in e13.columns]
print('added:', newc)
print(m[newc].isna().mean().round(3))
p = A.save_table(m, 'e014_batch1.parquet')
print(p)
