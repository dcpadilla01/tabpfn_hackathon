import pandas as pd, numpy as np
snap = agent_api.snapshot()
dm = snap.table("display_mailer"); tx = snap.table("transactions")
print("dm dtypes:\n", dm.dtypes)
print("tx dtypes:\n", tx.dtypes)
print("display uniques sample:", dm.display.unique()[:10])
print("mailer uniques:", dm.mailer.unique())
# dup rows example
d = dm[dm.duplicated(["product_id","store_id","week_no"], keep=False)]
print("dup rows:", len(d))
print(d.head(10).to_string())
# proper match test on last 28 days
sub = tx[tx.day >= 459-28].copy()
sub["week_no"] = (sub.day+8)//7
m = sub.merge(dm, on=["product_id","store_id","week_no"], how="left")
print("last28 match rate:", m.display.notna().mean())
# ignoring store
pw = dm.groupby(["product_id","week_no"]).size().rename("n").reset_index()
m2 = sub.merge(pw, on=["product_id","week_no"], how="left")
print("last28 match ignoring store:", m2.n.notna().mean())
print("product_id dtype in tx:", sub.product_id.dtype, "in dm:", dm.product_id.dtype)
