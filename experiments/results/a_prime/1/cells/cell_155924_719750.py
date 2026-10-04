import numpy as np, pandas as pd, agent_api as api
s = api.snapshot()  # capped at 459
tx = s.transactions
wk = tx.groupby('week_no').sales_value.sum()
hh = tx.groupby('week_no').household_key.nunique()
per_hh = (wk/hh)
print('weeks', wk.index.min(), wk.index.max())
print('weekly total spend: first 20 weeks:'); print(wk.head(20).round(0).to_dict())
print('per-household weekly spend, weeks 1..52:'); print(per_hh.head(52).round(1).to_dict())
print('overall per-hh mean %.1f std %.1f' % (per_hh.mean(), per_hh.std()))
# same-window-last-year anchor feasibility: for snapshot d, mean per-household spend in [d-27,d] and [d-363,d-336]
days = api.snapshot_days()['train']+api.snapshot_days()['validation']
print('val days', api.snapshot_days()['validation'])
tx2 = tx[['household_key','day','sales_value']]
for d in [95, 207, 431, 459]:
    w0 = tx2[(tx2.day>=d-27)&(tx2.day<=d)].groupby('household_key').sales_value.sum()
    w1 = tx2[(tx2.day>=d-363)&(tx2.day<=d-336)].groupby('household_key').sales_value.sum()
    print(d, 'recent4w mean %.1f  lastyear4w mean %.1f  n_ly %d' % (w0.mean(), w1.mean(), len(w1)))