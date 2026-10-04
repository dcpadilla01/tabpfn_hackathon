import numpy as np, pandas as pd, math
sd = agent_api.snapshot_days(); print("snapdays", sd)
v = agent_api.snapshot(459)
t = v.transactions
print("day range", t.day.min(), t.day.max(), "rows", len(t), "hh", t.household_key.nunique())
t = t.assign(wk=(t.day+8)//7)
wk = t.groupby('wk').agg(sales=('sales_value','sum'), hh=('household_key','nunique'))
wk['per_hh'] = wk.sales/wk.hh
print("weekly per_hh stats:", wk.per_hh.describe().round(2).to_dict())
print(wk.per_hh.round(1).to_string())
tt = agent_api.train_targets()
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median']).round(1).to_string())
for d in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    w=(d+8)//7
    print("snap", d, "week", w, "phase", (w-1)%52+1, "winweeks", (d+9)//7, (d+36)//7)
e = agent_api.load_saved('e013_stock.parquet')
print("e013", e.shape, "ncols", len(e.columns))
dm = v.display_mailer
print("dm shape", dm.shape, "weeks", dm.week_no.min(), dm.week_no.max())
print("display>0 frac", round((dm.display>0).mean(),4), "mailer>0 frac", round((dm.mailer>0).mean(),4))