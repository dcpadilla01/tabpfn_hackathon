
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print("feats shape:", feats.shape)
print("feats cols:", feats.columns.tolist())
m = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
t = m['future_spend_4w']
print(t.describe())
print("zero frac:", (t==0).mean(), "| <10 frac:", (t<10).mean())
print(m.groupby('snapshot_day')['future_spend_4w'].agg([('mean','mean'),('med','median'),('zero', lambda s:(s==0).mean())]))
num = feats.drop(columns=['household_key','snapshot_day']).select_dtypes(include=[np.number])
corrs = num.corrwith(m['future_spend_4w']).sort_values()
print("corr with target (bottom 15):"); print(corrs.head(15))
print("corr with target (top 15):"); print(corrs.tail(15))
print("NaN counts>0:", num.isna().sum()[num.isna().sum()>0].to_dict())
