import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved('feats_v3.parquet')
cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for c in cols: print(c)
