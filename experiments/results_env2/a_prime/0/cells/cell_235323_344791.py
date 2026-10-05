
import pandas as pd, numpy as np

names = ['e013_storeprod','e016_batch2','e016_merged','e017_phase','e017_merged','e018_dyn','e018_merged','e001_rfm']
tabs = {}
for nm in names:
    try:
        tabs[nm] = load_saved(nm + '.parquet')
        print(nm, tabs[nm].shape)
    except Exception as e:
        print(nm, 'ERR', type(e).__name__, str(e)[:120])

key = ['household_key','snapshot_day']
base = tabs['e013_storeprod']
bset = set(base.columns)
print('base features:', len(bset - {'household_key','snapshot_day'}))

bk = base[key].drop_duplicates().reset_index(drop=True)
for nm in ['e016_merged','e017_merged','e018_merged']:
    t = tabs[nm]
    tk = t[key].drop_duplicates()
    print(nm, 'rows', len(t), 'keys', len(tk), 'inner-with-base', len(bk.merge(tk, on=key)))

for nm in ['e016_batch2','e017_phase','e018_dyn']:
    t = tabs[nm]
    ex = [c for c in t.columns if c not in bset and c not in key]
    print('===', nm, 'extras', len(ex))
    print(ex)

print('=== e001_rfm cols not in base:')
print([c for c in tabs['e001_rfm'].columns if c not in bset])
