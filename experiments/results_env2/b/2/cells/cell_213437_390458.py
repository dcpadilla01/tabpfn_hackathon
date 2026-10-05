import agent_api as api, pandas as pd, numpy as np
t = api.load_saved("rfm_cadence_v1.parquet")
print("E003 table:", t.shape)
print(list(t.columns))
tt = api.train_targets()
print(tt["future_spend_4w"].describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]))
