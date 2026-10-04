import numpy as np, pandas as pd
from agent_api import load_saved, save_table, train_targets, snapshot_days

names = ['e017_gapfill.parquet','e015_base.parquet','e004_long_hist.parquet','e011_display.parquet','e005_seasonal_peer.parquet','e001_history.parquet']
tabs = {}
for n in names:
    try:
        tabs[n] = load_saved(n)
        print(n, tabs[n].shape)
    except Exception as e:
        print('ERR', n, repr(e))

e17 = tabs['e017_gapfill.parquet']; e15 = tabs['e015_base.parquet']
print('\ne17 cols:', list(e17.columns))
print('e17 snapdays:', sorted(e17.snapshot_day.unique()))
print('snapshot_days:', snapshot_days())
print('dup keys:', e17.duplicated(['household_key','snapshot_day']).sum())
print('const cols:', [c for c in e17.columns if e17[c].nunique(dropna=False)<=1])
print('nan:', {c:int(e17[c].isna().sum()) for c in e17.columns if e17[c].isna().any()})
print('dtypes:', e17.dtypes.value_counts().to_dict())
same_keys = set(map(tuple, e17[['household_key','snapshot_day']].values)) == set(map(tuple, e15[['household_key','snapshot_day']].values))
print('keys match e15:', same_keys)
for n in names[2:]:
    t = tabs[n]
    ov = sorted(set(t.columns) & set(e17.columns))
    print(n, 'nfeat', t.shape[1]-2, 'overlap_with_e17:', ov)
p = save_table(e17, 'e017r')
print('saved path:', repr(p))