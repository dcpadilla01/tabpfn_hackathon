
import agent_api as A
e13 = A.load_saved('e013_storeprod.parquet')
b1 = A.load_saved('cand_batch1.parquet')
print('e13', e13.shape)
print('b1', b1.shape)
print('e13 cols:', list(e13.columns))
print('b1 cols:', list(b1.columns))
print(b1.head(3).T)
t = A.train_targets()
print(t['future_spend_4w'].describe())
print(A.snapshot_days())


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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
