import agent_api as api
import pandas as pd, numpy as np

# Check target transform viability: what fraction of targets are 0 or tiny?
tr = api.train_targets()
y = tr['future_spend_4w']
print("zeros:", (y==0).mean(), " <5:", (y<5).mean(), " <20:", (y<20).mean())
print(np.log1p(y).describe())

# Also check skew of spend_l1 (main predictor)
e16 = api.load_saved("e016_smoothed.parquet")
print(e16['spend_l1'].describe())
print("corr log1p(y) with log1p(spend_l1):", np.corrcoef(np.log1p(tr['future_spend_4w']), np.log1p(e16.merge(tr[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='right')['spend_l1'].fillna(0)))[0,1])