import agent_api as api, pandas as pd, numpy as np
v = api.snapshot()
tx = v.transactions
prod = v.table("products")
m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
dep = m.groupby("department").sales_value.sum().sort_values(ascending=False)
print("n departments:", len(dep)); print(dep.head(25))
print(m["brand"].value_counts(dropna=False).head())
camp = v.table("campaigns"); print(camp.description.value_counts())
ct = v.table("campaign_targets"); print(ct.description.value_counts()); print("hh targeted:", ct.household_key.nunique())
red = v.table("coupon_redemptions"); print("redemptions rows:", len(red), "hh:", red.household_key.nunique())
