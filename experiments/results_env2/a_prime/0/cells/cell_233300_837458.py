
import agent_api as A
import pandas as pd
e13 = A.load_saved('e013_storeprod.parquet')
b1 = A.load_saved('cand_batch1.parquet')
b1f = b1.drop(columns=['snapshot_day'])
m = e13.merge(b1f, on='household_key', how='left')
print(m.shape)
# sanity: rows match keys
canon = e13[['household_key','snapshot_day']]
assert m[['household_key','snapshot_day']].equals(canon)
print('new cols ok:', [c for c in m.columns if c not in e13.columns])
p = A.save_table(m, 'e014_batch1.parquet')
print(p)
