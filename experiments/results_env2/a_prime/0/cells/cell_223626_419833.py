import agent_api, numpy as np, pandas as pd

df = agent_api.load_saved('e007_te.parquet')
print('shape', df.shape)
print(df.dtypes.value_counts())
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', d.shape)
y = d['future_spend_4w'].values
print('target mean/med/zero-frac (train):', 
      round(np.mean(y),1), round(np.median(y),1), round(np.mean(y==0),3))

feat = [c for c in d.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n feat', len(feat))
print('columns:', feat)
