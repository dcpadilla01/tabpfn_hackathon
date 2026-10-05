import agent_api as A, pandas as pd, numpy as np
for name in ["e001_recent_behavior.parquet","e004_temporal.parquet"]:
    t = A.load_saved(name)
    print(name, t.shape)
    print(t.columns.tolist())
    print()
tt = A.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]))
