import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
# Prototype E014 fn and time it at the largest snapshot
def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    wk = tx.groupby('w').sales_value.sum()
    wk = wk.reindex(range(1, week_max+1)).ffill()
    lvl = wk/wk.iloc[:8].mean()
    hh = view.households
    g = tx.groupby(['household_key','w']).sales_value.sum()
    idx = pd.MultiIndex.from_product([hh, range(max(1,week-25), week+1)], names=['household_key','w'])
    g = g.reindex(idx).fillna(0.0)
    z = g.xs(hh, level=0) if False else None
    arr = np.zeros((len(hh), 26))
    for i,h in enumerate(hh):
        arr[i] = g.xs(h, level=0).values
    return None
week_max = 70
import agent_api, time
t0=time.time()
# skip prototype; instead time the reindex approach on full tx at day 459
s = agent_api.snapshot()
tx = s.table('transactions')
tx = tx[tx.sales_value.notna()].copy()
tx['w'] = (tx.day+8)//7
g = tx.groupby(['household_key','w']).sales_value.sum()
hh = tx.household_key.unique()
t0=time.time()
idx = pd.MultiIndex.from_product([hh, range(41, 67)], names=['household_key','w'])
gg = g.reindex(idx).fillna(0.0)
mat = gg.unstack()  # hh x weeks
print('unstack time', round(time.time()-t0,2), mat.shape)
t0=time.time()
roll = mat.rolling(4, axis=1).mean()
print('rolling time', round(time.time()-t0,2))
# holiday flags
wmod = np.array([( (w-1)%52 )+1 for w in range(41,67)])
hol = ((wmod>=51)|(wmod<=1)).astype(float)
print('holiday weeks in window:', hol)
