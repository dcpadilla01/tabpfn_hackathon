import pandas as pd, numpy as np
t = agent_api.load_saved("e011_table.parquet")
print("e011:", t.shape)
print(list(t.columns))
for name in ["deal_v1","hazard_v1"]:
    try:
        d = agent_api.load_saved(name)
        print(name, d.shape, list(d.columns)[:30])
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
v = agent_api.snapshot()
tx = v.transactions
print(tx[["coupon_disc","coupon_match_disc","retail_disc","sales_value","quantity"]].describe().round(3))
