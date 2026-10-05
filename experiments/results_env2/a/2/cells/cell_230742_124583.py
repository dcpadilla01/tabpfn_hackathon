import pandas as pd, numpy as np
from agent_api import snapshot, train_targets, KEYS, TARGET

tr = snapshot().transactions
print('transactions rows:', len(tr), 'max day:', tr.day.max(), 'n hh:', tr.household_key.nunique())
dm = snapshot().display_mailer
print('display_mailer rows:', len(dm))

# verify label computation: future_spend_4w = sum sales on days d+1..d+28
tt = train_targets()
spend = tr.groupby(['household_key','day']).sales_value.sum().reset_index()
def lab(d):
    w = spend[(spend.day > d) & (spend.day <= d+28)]
    return w.groupby('household_key').sales_value.sum()
ok, tot = 0, 0
for d in [95, 235, 431]:
    L = lab(d).rename('mylab')
    sub = tt[tt.snapshot_day==d].merge(L, on='household_key', how='left').fillna({'mylab':0})
    diff = (sub[TARGET] - sub.mylab).abs()
    print('day', d, 'n:', len(sub), 'max abs diff:', round(diff.max(),4), 'mismatch:', int((diff>0.01).sum()))
    ok += int((diff<=0.01).sum()); tot += len(sub)
print('verified rows:', ok, '/', tot)
