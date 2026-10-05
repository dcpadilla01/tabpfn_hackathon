import agent_api as api
v = api.snapshot()
print(type(v.households), getattr(v.households, "dtype", None), len(v.households) if v.households is not None else None)
print(v.day, v.week)
hh = pd.Index(v.households)
print(hh.dtype)
c = v.table("campaigns")[["campaign","start_day","end_day"]]
ct = v.table("campaign_targets")
cm = ct.merge(c, on="campaign", how="left")
cm = cm[cm.household_key.isin(hh)]
g = cm.groupby("household_key")
r = g.apply(lambda d: float((d.end_day >= 459).sum()))
print("dtype:", r.dtype, "idx dtype:", r.index.dtype, "len:", len(r))
print(r.head())
