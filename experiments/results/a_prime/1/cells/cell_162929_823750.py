import pandas as pd, numpy as np
import agent_api as A
base = A.load_saved('e016_smoothed.parquet')
feat = [c for c in base.columns if c not in ('household_key','snapshot_day')]
for i in range(0, len(feat), 8):
    print(', '.join(feat[i:i+8]))
tt = A.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
print('zero frac', (tt.future_spend_4w==0).mean())
