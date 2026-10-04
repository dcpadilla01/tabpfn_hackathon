import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
print('households:', v.households)
print('day:', v.day, 'week:', v.week)
# maybe households derivable from transactions at this snapshot
t = v.transactions
print('txn shape:', t.shape, 'max day:', t['day'].max())
hh = t.groupby('household_key')['day'].min()
print('n households ever:', len(hh))
# eligibility rule: first purchase >= 84 days before snapshot
elig = hh[hh <= v.day - 84]
print('eligible at 459:', len(elig))
# compare with a saved table's row count per snapshot
e11 = A.load_saved('e011_price.parquet')
print('e11 rows at 459:', (e11.snapshot_day==459).sum())
# are the eligible keys exactly the e11 keys at 459?
e11h = set(e11[e11.snapshot_day==459]['household_key'])
print('match:', set(elig.index.astype(e11.household_key.dtype)) == e11h)
