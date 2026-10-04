import agent_api as A, pandas as pd, numpy as np
v = A.snapshot(459)
t = v.transactions
p = v.products
print("tx rows:", len(t), "hh:", t.household_key.nunique(), "day range:", t.day.min(), t.day.max())
print("products rows:", len(p), "unique pid:", p.product_id.nunique())
print("coverage:", t.product_id.isin(p.product_id).mean())
print("n departments:", p.department.nunique(), "n commodities:", p.commodity_desc.nunique(), "n manufacturers:", p.manufacturer.nunique())
m = t.merge(p[['product_id','department','commodity_desc','manufacturer','brand']], on='product_id', how='left')
sp = m.groupby('commodity_desc').sales_value.sum().sort_values(ascending=False)
print(sp.head(25))
print("share top20 commodities:", sp.head(20).sum()/sp.sum())
sp2 = m.groupby('manufacturer').sales_value.sum().sort_values(ascending=False)
print(sp2.head(10))
print("manufacturer NaN:", m.manufacturer.isna().mean())
tt = A.train_targets()
print(tt.future_spend_4w.describe())