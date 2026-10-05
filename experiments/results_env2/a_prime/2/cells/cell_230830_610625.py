
import pandas as pd, numpy as np
t10 = agent_api.load_saved("e010_lifecycle.parquet")
print("E010 shape", t10.shape)
print("E010 cols", list(t10.columns))
t05 = agent_api.load_saved("e005_decay_gapcv.parquet")
print("E005 cols", list(t05.columns))
v = agent_api.snapshot()
tx = v.table("transactions")
print(tx.head(3).to_string())
print(tx[["sales_value","quantity","coupon_disc","coupon_match_disc","retail_disc"]].describe().to_string())
print("hh type", type(v.households))
try:
    print(v.households.head())
except Exception as e:
    print("hh err", e)
