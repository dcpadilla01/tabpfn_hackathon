
import agent_api as A
import pandas as pd, numpy as np

for name in ['e019_everything.parquet','e019_full_merged.parquet','e018_union_full.parquet']:
    t = A.load_saved(name)
    print('='*20, name, t.shape)
    print('dup keys:', t.duplicated(['household_key','snapshot_day']).sum())
    print('snapshots:', sorted(t.snapshot_day.unique()))
    num = t.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in t.columns if c not in num]
    print('n numeric:', len(num), 'n other:', len(cat), cat[:20])
    na = t.isna().mean().sort_values(ascending=False)
    print('cols with >50% NaN:', (na>0.5).sum(), '| all-NaN cols:', (na==1.0).sum())
    print(na.head(5))
