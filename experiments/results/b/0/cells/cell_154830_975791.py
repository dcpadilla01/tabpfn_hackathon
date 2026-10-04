import pandas as pd, numpy as np
v = agent_api.snapshot()
t = v.transactions
print("TXN", t.shape)
print("neg:", int((t.sales_value<0).sum()), "zero:", int((t.sales_value==0).sum()))
print("day range", int(t.day.min()), int(t.day.max()))
cr = v.coupon_redemptions
print("CR", cr.shape); print(cr.head(3))
dm = v.display_mailer
print("DM", dm.shape); print(dm.head(3))
print("display:", dm.display.value_counts().sort_index().to_dict())
print("mailer:", dm.mailer.value_counts().sort_index().to_dict())
h = v.households
print("households", type(h), h.shape)
print(h.head(3))
print("week", v.week, "day", v.day)
e11 = agent_api.load_saved('e011_price.parquet')
print("E011", e11.shape)
print(list(e11.columns))