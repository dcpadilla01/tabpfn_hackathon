import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
print("mean/median target by train snapshot day:")
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.round(2))
# weekly aggregate spend seasonality (capped view at 459)
v = agent_api.snapshot(459)
tr = v.transactions
wk = tr.groupby("week_no").sales_value.sum()
print("\nweekly total spend, first 30 weeks:"); print(wk.head(30).round(0).to_dict())
print("weekly total spend, weeks 50-80:"); print(wk.loc[50:80].round(0).to_dict())
print("\nn weeks:", wk.shape[0], "min/max week:", wk.index.min(), wk.index.max())
