import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

print('snapdays', snapshot_days())
t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('E017 shape', t.shape, 'n_feat', len(feat_cols))
print('cols:', sorted(feat_cols))

df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape, 'missing targets', int(df['future_spend_4w'].isna().sum()))
df = df[df['future_spend_4w'].notna()]
y = df['future_spend_4w'].astype(float).values
days = df['snapshot_day'].values
print('y: mean %.2f med %.2f std %.2f zero %.3f' % (y.mean(), np.median(y), y.std(), (y==0).mean()))
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(2))
