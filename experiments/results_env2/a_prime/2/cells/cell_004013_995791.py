
import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
print('shape', t.shape)
for i, c in enumerate(t.columns):
    print(i, c, t[c].dtype)

tt = A.train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','std','count']))
print(A.snapshot_days())
