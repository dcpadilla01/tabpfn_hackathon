
import agent_api as A
import pandas as pd, numpy as np
tt = A.train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(2))
# recent-spend level by snapshot (spend_l1 from union table) to see drift vs target
t = A.load_saved('e018_union_full.parquet')
m = t.groupby('snapshot_day')[['spend_l1','spend_total','tenure_x']].mean()
print(m.round(2))
