import numpy as np, pandas as pd
v = snapshot()
ct = v.campaign_targets; camp = v.campaigns
# how many distinct campaigns per household per type (proxy for "high-value target")
cnt = ct.groupby(["household_key","description"]).size().unstack(fill_value=0)
print(cnt.describe().round(2))
# cross-tab with spend level
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d = d.merge(cnt, on="household_key", how="left").fillna(0)
for t in ["TypeA","TypeB","TypeC"]:
    d[t+"_n"] = d[t]
    print(t, d.groupby(t+"_n")["future_spend_4w"].agg(["mean","size"]).round(1).T.to_dict())
# also check: does spend_l1 differ by TypeA count? (i.e. is it just selection on past spend?)
print("spend_l1 by TypeA_n:")
print(d.groupby("TypeA_n")["spend_l1"].mean().round(1).to_dict())
print("spend_l1 by TypeB_n:")
print(d.groupby("TypeB_n")["spend_l1"].mean().round(1).to_dict())