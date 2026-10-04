import agent_api, pandas as pd
base = agent_api.load_saved('e016_smoothed.parquet')
X = agent_api.load_saved('e017_disc_seasonal.parquet')
m = base.merge(X.drop(columns=['index']), on=['household_key','snapshot_day'], how='left')
print(m.shape, len(m)-len(base))
path = agent_api.save_table(m, 'e017_disc_seasonal.parquet')
print(path)