import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")

print("sales_value stats:", tr.sales_value.describe().round(2).to_dict())
print("neg sales rows:", (tr.sales_value<0).sum(), " zero:", (tr.sales_value==0).sum())
print("quantity stats:", tr.quantity.describe().round(2).to_dict())
print("qty<=0 rows:", (tr.quantity<=0).sum())

# trans_time -> hour
tt = tr.trans_time.dropna()
hh_ = (tt//100).clip(0,23)
print("\nhour dist:", hh_.value_counts().sort_index().to_dict())

# day-of-week (day mod 7) trip counts
dow = tr.groupby(tr.day % 7)["basket_id"].nunique()
print("\nbaskets by day%7:", dow.to_dict())

# brand
print("\nbrand values:", p.brand.value_counts(dropna=False).to_dict())

# COUPON/MISC ITEMS commodity check
cm = tr.merge(p[["product_id","commodity_desc"]], on="product_id", how="left")
print("\nCOUPON/MISC stats:", cm[cm.commodity_desc=="COUPON/MISC ITEMS"].sales_value.describe().round(2).to_dict())
