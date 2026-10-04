import agent_api, pandas as pd
base = agent_api.load_saved('e016_smoothed.parquet')
big = agent_api.load_saved('e017_disc_seasonal.parquet')
newcols = ['disc_net_84','coupon_disc_84','match_disc_84','coupon_redemptions_84','dsl_coupon','wk_sin','wk_cos','wk_sin2','wk_cos2']
X = big[['household_key','snapshot_day']+newcols]
print(X.shape)
m = base.merge(X, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'dup rows:', len(m)-len(base))
assert len(m)==len(base) and m.shape[1]==127+9
path = agent_api.save_table(m, 'e017_v2.parquet')
print(path)