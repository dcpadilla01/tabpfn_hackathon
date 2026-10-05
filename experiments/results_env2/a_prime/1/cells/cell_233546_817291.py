import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
cols = [c for c in T.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))
