import pandas as pd, numpy as np
v = agent_api.snapshot()
t = v.transactions; dm = v.display_mailer
print(t.dtypes.to_dict()); print(dm.dtypes.to_dict())
chk = ((t.week_no == (t.day+8)//7).mean())
print("week_no consistent:", chk)
tg = t.groupby(['product_id','store_id','week_no']).size().reset_index(name='n')
print("tx combos", len(tg), "stores", t.store_id.nunique(), "prods", t.product_id.nunique())
mg = tg.merge(dm, on=['product_id','store_id','week_no'], how='left')
print("match rate", round(mg.display.notna().mean(),4))
sub = mg[mg.display.notna()]
dmap = {c:i for i,c in enumerate(['0','1','2','3','4','5','6','7','9','A'])}
mmap = {c:i for i,c in enumerate(['0','A','C','D','F','H','J','L','P','X','Z'])}
sub2 = sub.copy()
sub2['dv'] = sub2.display.map(dmap); sub2['mv'] = sub2.mailer.map(mmap)
print("dv dist", sub2.dv.value_counts().sort_index().to_dict())
print("mv dist", sub2.mv.value_counts().sort_index().to_dict())
print("display>0 share among matched:", round((sub2.dv>0).mean(),4), "mailer>0:", round((sub2.mv>0).mean(),4))