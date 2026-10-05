
import agent_api, pandas as pd, numpy as np
e2 = agent_api.load_saved('e002_mix.parquet')
cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
print(cols)
