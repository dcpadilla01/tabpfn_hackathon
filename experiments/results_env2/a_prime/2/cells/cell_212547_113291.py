v = agent_api.snapshot(459)
tx = v.transactions
prod = v.table("products")
print(tx.shape, prod.shape)
m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
print(m["department"].value_counts().head(20))
print(m["brand"].value_counts(dropna=False))
# spend share by dept overall
g = m.groupby("department")["sales_value"].sum().sort_values(ascending=False)
print(g.head(15))
print("total spend", g.sum())
