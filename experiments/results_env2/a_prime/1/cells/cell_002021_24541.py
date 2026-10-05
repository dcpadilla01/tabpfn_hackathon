
import pandas as pd, numpy as np
t15 = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
print("t15:", t15.shape, "tt:", tt.shape)
print("t15 dup keys:", t15.duplicated(['household_key','snapshot_day']).sum())
print("tt dup keys:", tt.duplicated(['household_key','snapshot_day']).sum())
print("t15 dtypes hh/sd:", t15['household_key'].dtype, t15['snapshot_day'].dtype)
print("tt dtypes hh/sd:", tt['household_key'].dtype, tt['snapshot_day'].dtype)
df = t15.merge(tt, on=['household_key','snapshot_day'], how='left')
print("merged:", df.shape)
print("t15 index type:", t15.index.dtype, "tt index:", tt.index.dtype)
