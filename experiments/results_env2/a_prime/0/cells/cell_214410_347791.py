import pandas as pd, numpy as np, time
snap = agent_api.snapshot()
dm = snap.table("display_mailer")
print("dm shape", dm.shape)
print("n products", dm.product_id.nunique(), "n stores", dm.store_id.nunique(), "n weeks", dm.week_no.nunique())
print("dup (prod,store,week):", dm.duplicated(["product_id","store_id","week_no"]).sum())
t0=time.time()
sub = dm[dm.week_no >= 50]
print("weeks>=50 rows:", len(sub), "t=%.1fs" % (time.time()-t0))
agg = sub.groupby("week_no").size()
print(agg.tail(20))
# product-week level size
pw = sub.groupby(["product_id","week_no"]).agg(nd=("store_id","nunique")).reset_index()
print("prod-week rows (weeks>=50):", len(pw))
# display/mailr non-zero share
print("display!=0 rows:", (dm.display!=0).mean().round(3), "mailer!=0:", (dm.mailer!="0").mean().round(3))
tx = snap.table("transactions")
print("tx shape", tx.shape)
t0=time.time()
w = (tx.day+8)//7
m = tx.assign(week_no=w).merge(pw, on=["product_id","week_no"], how="left")
print("merge time %.1fs, matched %.3f" % (time.time()-t0, m.nd.notna().mean()))
