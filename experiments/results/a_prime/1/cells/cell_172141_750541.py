
import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')          # E018's 186-col table (best, MAE 62.037)
v2 = A.load_saved('e017_v2.parquet')                 # E017's 136-col table incl. discount decomposition
new = ['coupon_disc_84', 'match_disc_84']            # the only validated-family cols missing from E018
assert not any(c in u.columns for c in new)
add = v2[['household_key','snapshot_day'] + new]
uni = u.merge(add, on=['household_key','snapshot_day'], how='left')
print('union shape:', uni.shape, '| dup keys:', uni.duplicated(['household_key','snapshot_day']).sum())
print('NaN frac new cols:', uni[new].isna().mean().round(3).to_dict())
print(uni[new].describe().round(2))
path = A.save_table(uni, 'e019_true_union.parquet')
print(path)
