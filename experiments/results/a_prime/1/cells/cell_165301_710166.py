import numpy as np, pandas as pd
v = snapshot()
ct = v.campaign_targets
cnt = ct.groupby(["household_key","description"]).size().unstack(fill_value=0).reset_index()
print(cnt.head(3))
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d = d.merge(cnt, on="household_key", how="left").fillna({c:0 for c in ["TypeA","TypeB","TypeC"]})
for t in ["TypeA","TypeB","TypeC"]:
    g = d.groupby(t)["future_spend_4w"].agg(["mean","size"]).round(1)
    print(t); print(g)
    print("  spend_l1:", d.groupby(t)["spend_l1"].mean().round(1).to_dict())
    print("  spend_l123_mean:", d.groupby(t)["spend_l123_mean"].mean().round(1).to_dict())