import agent_api as A, pandas as pd, numpy as np
f4 = A.load_saved("feats_v3.parquet")
print(f4.columns.tolist()[40:])
t = A.train_targets()
m = f4.merge(t, on=['household_key','snapshot_day'])
print(m.shape)
num = m.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','index'], errors='ignore')
cor = num.corr()['future_spend_4w'].sort_values(key=abs, ascending=False)
print(cor.head(25).round(3))
