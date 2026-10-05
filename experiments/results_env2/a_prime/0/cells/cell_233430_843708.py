import agent_api, pandas as pd
for name in ["e013_storeprod.parquet","e012_outcome2.parquet","e007_te.parquet","e001_rfm.parquet","e002_composition.parquet"]:
    df = agent_api.load_saved(name)
    print("==", name, df.shape)
    print(list(df.columns))
print()
v = agent_api.snapshot()
tx = v.table("transactions")
prod = v.table("products")
print("tx", tx.shape, "prod", prod.shape)
m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
print(m.groupby("department").sales_value.sum().sort_values(ascending=False).head(25))
print(m.brand.value_counts(dropna=False).head())
