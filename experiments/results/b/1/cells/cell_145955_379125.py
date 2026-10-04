import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
v459 = agent_api.snapshot()  # max view
t459 = v459.transactions
fp_full = t459.groupby('household_key').day.min()   # first purchase over all visible data
print('households with any txn <=459:', len(fp_full))
for d in [95, 123, 207, 403, 431, 459]:
    rows = set(tt[tt.snapshot_day==d].household_key)
    elig_full = set(fp_full[fp_full <= d-84].index)          # full-data first purchase
    v = agent_api.snapshot(as_of_day=d)
    tv = v.transactions
    elig_trunc = set(tv.groupby('household_key').day.min().pipe(lambda s: s[s<=d-84]).index)  # truncated
    print(f'day {d}: rows={len(rows)} elig_full={len(elig_full)} elig_trunc={len(elig_trunc)} '
          f'rows==elig_full:{rows==elig_full} rows==elig_trunc:{rows==elig_trunc} '
          f'rows_no_txn_yet={len(rows - set(tv.household_key.unique()))}')
print()
print('type of v.households at 403:', type(agent_api.snapshot(as_of_day=403).households))
print('type at 95:', type(agent_api.snapshot(as_of_day=95).households))
print('type at 459:', type(v459.households))
hh403 = agent_api.snapshot(as_of_day=403).households
if hh403 is not None:
    print('n at 403:', len(hh403), '== rows?', set(hh403)==set(tt[tt.snapshot_day==403].household_key))