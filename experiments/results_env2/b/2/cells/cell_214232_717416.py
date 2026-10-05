import pandas as pd, numpy as np, time
v = agent_api.snapshot(459)
tx = v.table("transactions")
print("tx rows", len(tx))
t0=time.time()
dm = v.table("display_mailer")
print("dm load", time.time()-t0, dm.shape)
# subset to recent weeks
wk = (459+8)//7
print("snapshot week", wk)
t0=time.time()
dmr = dm[(dm.week_no >= wk-7)]
print("dm recent", dmr.shape, "load+filter", time.time()-t0)
# test join
txr = tx[tx.day >= 459-56].copy()
txr['week_no'] = (txr.day+8)//7
t0=time.time()
m = txr.merge(dmr, on=['product_id','store_id','week_no'], how='left')
print("merge", time.time()-t0, m.shape)
print("match rate", m.display.notna().mean())
print("spend share on displayed", (m.sales_value*(m.display!='0').fillna(False)).sum()/m.sales_value.sum())