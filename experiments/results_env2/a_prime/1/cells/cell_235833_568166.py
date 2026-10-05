
import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
cols = [c for c in e012.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
print(cols)
r = agent_api.load_saved('rawrec')
print(r.shape, r.columns.tolist()[:20])
