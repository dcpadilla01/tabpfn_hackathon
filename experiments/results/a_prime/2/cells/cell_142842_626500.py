import numpy as np, pandas as pd
tt = agent_api.train_targets()
v = agent_api.snapshot()
tr = v.transactions
print(tr.day.max(), tr.shape)
# recent 28d spend per household per snapshot day
def recent_spend(day, w=28):
    return tr[(tr.day>day-w)&(tr.day<=day)].groupby('household_key').sales_value.sum()
out=[]
for day in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    r = recent_spend(day).rename('recent')
    t = tt[tt.snapshot_day==day].set_index('household_key')
    j = t.join(r).fillna({'recent':0.0})
    out.append(j[['future_spend_4w','recent']])
allj = pd.concat(out)
print(allj[['future_spend_4w','recent']].corr())
print(allj.corr())
# MAE of predicting recent spend as target
from numpy import abs
mae = (allj.future_spend_4w-allj.recent).abs().mean()
print('MAE recent-as-pred:', mae)
# scaled variants
for k in [0.5,0.7,0.8,0.9,1.0,1.1,1.2]:
    print(k, (allj.future_spend_4w-k*allj.recent).abs().mean())
