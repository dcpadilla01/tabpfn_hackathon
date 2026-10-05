import agent_api as api, pandas as pd, numpy as np
v = api.snapshot()
tx = v.transactions
print(tx.shape)
print(tx[["sales_value","coupon_disc","coupon_match_disc","retail_disc","quantity","trans_time"]].describe().loc[["mean","min","max"]])
print(tx.head(3))
dem = v.table("demographics"); print(dem.shape); print(dem.head(3))
