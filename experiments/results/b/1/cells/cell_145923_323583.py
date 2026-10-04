import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
print('tt shape:', tt.shape)
print('tt per-day counts:'); print(tt.groupby('snapshot_day').size())
t = agent_api.load_saved('e001_history.parquet')
print('\nE001 rows per snapshot_day:'); print(t.groupby('snapshot_day').size())
# what is v.households at various days?
for d in [95, 207, 403, 459]:
    v = agent_api.snapshot(as_of_day=d)
    print(f'\nas_of_day={d}: v.households n={len(v.households)}')
    # households with first purchase <= d-84 among transactions up to d
    tr = v.transactions
    fp = tr.groupby('household_key').day.min()
    n84 = int((fp <= d-84).sum())
    print('  households with first purchase <= d-84 (in txns up to d):', n84)
    print('  households in txns up to d:', tr.household_key.nunique())
    print('  tt count at d (if train):', int((tt.snapshot_day==d).sum()))
    hv = set(v.households); tv = set(tt[tt.snapshot_day==d].household_key)
    print('  v.households subset of tt?', hv <= tv, ' tt - v:', len(tv-hv), ' v - tt:', len(hv-tv))