
import agent_api as A, pandas as pd, numpy as np
from collections import defaultdict
tt = A.train_targets()
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index
    lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag2y = np.array([lag_fast(h,s,728) for h,s in zip(hk,sd)])
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
med = np.median(y)
def mae(p): return float(np.mean(np.abs(np.asarray(p,float)-y)))
for nm,arr in [('lag1y',lag1y),('lag2y',lag2y),('lag13',lag13)]:
    print(nm,'mean',round(arr.mean(),1),'corr',round(np.corrcoef(arr,y)[0,1],3),'MAE',round(mae(arr),2))
print('blend sp28+lag1y', round(mae(0.5*m['sp28'].fillna(0)+0.5*lag1y),2))
print('blend sp28+.3lag1y', round(mae(0.7*m['sp28'].fillna(0)+0.3*lag1y),2))
byh=defaultdict(list)
for h,s,vv in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w): byh[h].append((s,vv))
tsmean=np.full(len(y),np.nan)
for i,(h,s) in enumerate(zip(hk,sd)):
    past=[vv for s2,vv in byh[h] if s2<s]
    if past: tsmean[i]=np.mean(past)
okt=~np.isnan(tsmean)
print('tsmean MAE',round(mae(np.where(okt,tsmean,med)),2))
print('blend sp28+tsmean',round(mae(0.5*m['sp28'].fillna(0)+0.5*np.where(okt,tsmean,med)),2))
print('blend sp28+tsmean+lag1y',round(mae((m['sp28'].fillna(0)+np.where(okt,tsmean,med)+lag1y)/3),2))
print('ridge proxy check on lag1y alone: corr', round(np.corrcoef(np.log1p(lag1y),np.log1p(y))[0,1],3))
