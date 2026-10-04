import agent_api as api

# peek at what the model sees: is the model gradient boosting? unknown; just check target distribution
tr = api.train_targets()
print(tr.shape, tr['future_spend_4w'].describe())
print("val snapshots:", api.snapshot_days())

# also check: how many households per snapshot, and spend scale
import numpy as np
print(tr.groupby('snapshot_day')['future_spend_4w'].mean())