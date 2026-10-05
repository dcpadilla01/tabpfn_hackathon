import agent_api as api, pandas as pd, numpy as np
v = api.snapshot(95)
hh = pd.Index(v.households)
print(type(v.households), type(hh), hh.dtype)
c = v.table("campaigns")[["campaign","start_day","end_day"]]
ct = v.table("campaign_targets")
cm = ct.merge(c, on="campaign", how="left")
cm = cm[cm.household_key.isin(hh)]
g = cm.groupby("household_key")
r = g.apply(lambda d: float((d.end_day >= 95).sum()))
print("apply result dtype:", r.dtype, type(r.index), r.index.dtype)
print(type(g.keys))
