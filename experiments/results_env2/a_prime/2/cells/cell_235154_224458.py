import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")
dept_spend = tr.merge(p[["product_id","department"]], on="product_id", how="left").groupby("department")["sales_value"].sum().sort_values(ascending=False)
print("dept spend (top 12):")
print(dept_spend.head(12).round(0))
print("\nn depts:", tr.merge(p[["product_id","department"]], on="product_id", how="left")["department"].nunique())

# top commodities
com = tr.merge(p[["product_id","commodity_desc"]], on="product_id", how="left").groupby("commodity_desc")["sales_value"].sum().sort_values(ascending=False)
print("\ntop commodities:")
print(com.head(15).round(0))

# store-level: n stores per household over 364d
hh_stores = tr[tr.day >= 459-364].groupby("household_key")["store_id"].nunique()
print("\nn_stores_364 distribution:", hh_stores.describe().round(2).to_dict())

# trans_time
print("\ntrans_time sample:", tr.trans_time.dropna().head().tolist())
