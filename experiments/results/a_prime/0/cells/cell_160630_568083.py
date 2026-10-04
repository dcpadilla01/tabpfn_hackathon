
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
cols = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for c in cols: print(c)
