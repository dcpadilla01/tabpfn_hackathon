import agent_api as api
import pandas as pd

e16 = api.load_saved("e016_smoothed.parquet")   # 127 cols incl keys
e14 = api.load_saved("e014_demo_l13fix.parquet")  # has demo cols
demo_cols = ['classification_1','classification_2','classification_3','classification_4',
             'classification_5','homeowner_desc','kid_category_desc','has_demographics']
extra14 = ['has_real_l13','l13_over_recent']

merged = e16.merge(e14[['household_key','snapshot_day']+demo_cols+extra14],
                   on=['household_key','snapshot_day'], how='left')
print(merged.shape)
print("dup:", merged.duplicated(['household_key','snapshot_day']).sum())
print("demo coverage:", merged['has_demographics'].mean())
print(merged['classification_1'].value_counts(dropna=False).head())
path = api.save_table(merged, "e019_full_merged")
print(path)