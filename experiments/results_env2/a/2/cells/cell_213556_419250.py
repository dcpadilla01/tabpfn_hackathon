
import agent_api as api, pandas as pd, numpy as np

print(api.snapshot_days())
tt = api.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())

snap = api.snapshot()
tr = snap.transactions
print(tr.shape, tr.columns.tolist())
print(tr.head())
print("n households:", tr.household_key.nunique(), "day range:", tr.day.min(), tr.day.max())

bf = api.baseline_features()
print(bf.shape, bf.columns.tolist())
print(bf.head(3))
