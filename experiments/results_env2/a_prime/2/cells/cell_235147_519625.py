import agent_api, pandas as pd, numpy as np

print("=== API ===")
print([x for x in dir(agent_api) if not x.startswith("_")])

print("\n=== E011 table ===")
try:
    t = agent_api.load_saved("e011_discounts.parquet")
    print("shape:", t.shape)
    print("cols:", list(t.columns))
except Exception as e:
    print("ERR", repr(e))

print("\n=== targets by snapshot ===")
tt = agent_api.train_targets()
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median","std","max"]))
print("zero share overall:", round((tt.future_spend_4w==0).mean(),4))

print("\n=== view format + departments ===")
v = agent_api.snapshot(459)
print(type(v.households), len(v.households), list(v.households[:3]))
tr_full = v.transactions
print("tr shape:", tr_full.shape)
p = v.table("products")
dept_spend = tr_full.merge(p[["product_id","department"]], on="product_id", how="left").groupby("department")["sales_value"].sum().sort_values(ascending=False)
print("top dept spend:", {k: round(x) for k,x in dept_spend.head(12).items()})
