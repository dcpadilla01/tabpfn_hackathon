import agent_api as api
import pandas as pd

e16  = api.load_saved("e016_smoothed.parquet")
e14  = api.load_saved("e014_demo_l13fix.parquet")
e13  = api.load_saved("e013_peers.parquet")
v2   = api.load_saved("e017_v2.parquet")

demo_cols = ['classification_1','classification_2','classification_3','classification_4',
             'classification_5','homeowner_desc','kid_category_desc','has_demographics']
l13_cols = ['has_real_l13','l13_over_recent']
peer_cols = ['peer_recent28','peer_recent28_med','peer_ratio','peer_ratio2','peer_p90','peer_p10',
             'cohort_prior4w','anchor_ly_pop','spend_ly4w','has_ly4w','pop_ratio']
disc_cols = ['disc_net_84','coupon_redemptions_84','dsl_coupon']
cal_cols  = ['wk_sin','wk_cos','wk_sin2','wk_cos2']

add = e14[['household_key','snapshot_day']+demo_cols+l13_cols]
add = add.merge(e13[['household_key','snapshot_day']+peer_cols], on=['household_key','snapshot_day'], how='left')
add = add.merge(v2[['household_key','snapshot_day']+disc_cols+cal_cols], on=['household_key','snapshot_day'], how='left')

full = e16.merge(add, on=['household_key','snapshot_day'], how='left')
print(full.shape, "dups:", full.duplicated(['household_key','snapshot_day']).sum())
print("demo coverage:", full['has_demographics'].mean())
print("peer coverage:", full['peer_recent28'].notna().mean())
print("wk_sin coverage:", full['wk_sin'].notna().mean())
path = api.save_table(full, "e019_everything")
print(path)