import agent_api, pandas as pd
v = agent_api.snapshot()
tx = v.table("transactions")
prod = v.table("products")
m = tx.merge(prod[["product_id","commodity_desc"]], on="product_id", how="left")
print("null commodity:", m.commodity_desc.isna().mean())
g = m.groupby("commodity_desc").sales_value.agg(["sum","size"]).sort_values("sum", ascending=False)
print(g.head(30))
print("n commodities:", m.commodity_desc.nunique())
# how much do top-30 commodities cover
print("top30 share:", g["sum"].head(30).sum()/g["sum"].sum())
# household count
print("n households tx:", tx.household_key.nunique())
