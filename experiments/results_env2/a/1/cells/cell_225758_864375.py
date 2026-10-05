
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
print(tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).head(20))
print("frac zero:", (tt.future_spend_4w==0).mean())
print(tt.future_spend_4w.describe())
