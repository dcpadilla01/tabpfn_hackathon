import agent_api as api, pandas as pd, numpy as np

v = api.snapshot()
c = v.table("campaigns")
print("campaigns", c.shape)
print(c.head(3).to_string())
print(c["description"].value_counts())
print(c[["start_day","end_day"]].describe().to_string())
ct = v.table("campaign_targets")
print("targets", ct.shape)
print(ct["description"].value_counts())
tx = v.table("transactions")
print(tx[["sales_value","retail_disc","coupon_disc","coupon_match_disc"]].describe().to_string())

base = api.load_saved("e006_newblock.parquet")
print("base", base.shape, base.columns[:6].tolist())
