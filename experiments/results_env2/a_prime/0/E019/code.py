
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


# ---- cell ----

import pandas as pd, numpy as np
key = ['household_key','snapshot_day']
base = load_saved('e013_storeprod.parquet')
b16  = load_saved('e016_batch2.parquet')
b17  = load_saved('e017_phase.parquet')
b18  = load_saved('e018_dyn.parquet')

m = base.merge(b16, on=key, how='inner', suffixes=('','_x'))
m = m.merge(b17, on=key, how='inner')
m = m.merge(b18, on=key, how='inner')
print('union shape', m.shape)
print('dup cols check:', [c for c in m.columns if c.endswith('_x')][:5])
assert len(m) == 36426
m = m.sort_values(key).reset_index(drop=True)
path = save_table(m, 'e019_union')
print(path)
print('n features:', m.shape[1]-2)
print(m[key].head(3))
