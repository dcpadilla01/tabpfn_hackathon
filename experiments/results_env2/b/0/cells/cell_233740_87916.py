import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
s = agent_api.snapshot()
tx = s.table('transactions')
# Market-level trend: avg weekly spend per active household, and avg unit price by week
tx['week'] = (tx.day+8)//7
wk = tx.groupby('week').agg(total=('sales_value','sum'), hh=('household_key','nunique'), lines=('sales_value','size'))
wk['per_hh'] = wk.total/wk.hh
print(wk[['per_hh','lines']].describe())
print(wk.iloc[::6][['per_hh']].T.to_string())
# price proxy: sales_value per line over time (monthly)
tx['m'] = tx.day//28
mp = tx.groupby('m').sales_value.mean()
print('mean line value by 28d month:'); print(mp.iloc[::3].round(3).to_string())
# week alignment check
for d in [95,123,459,487,515,543]:
    print(d, 'week', (d+8)//7, 'mod52', ((d+8)//7-1)%52+1)
# max day in data
print('max day', tx.day.max())
