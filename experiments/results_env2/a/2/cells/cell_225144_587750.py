import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, KEYS, TARGET, snapshot_days

feats = load_saved('feats_v3.parquet')
print('feats shape:', feats.shape)
print('cols:', list(feats.columns)[:120])
tt = train_targets()
print('targets shape:', tt.shape)
y = tt[TARGET]
print(y.describe())
print('zero rate:', round(float((y==0).mean()), 3))
print(tt.groupby('snapshot_day')[TARGET].agg(['mean','count']).round(1))
