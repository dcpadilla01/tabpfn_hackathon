
v = agent_api.snapshot(95)
print("households:", type(v.households), len(v.households) if hasattr(v.households,'__len__') else '')
print("day/week:", v.day, v.week)
tx = v.table("transactions"); print(tx.dtypes.to_dict())
ct = v.table("campaign_targets"); print(ct.dtypes.to_dict())
red = v.table("coupon_redemptions"); print("redemptions", red.shape)
camp = v.table("campaigns"); print(camp.to_string())
b = agent_api.load_saved("e002_mix.parquet"); print(b.shape, b.columns.tolist()[:12])
