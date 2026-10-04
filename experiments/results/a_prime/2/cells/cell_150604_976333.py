import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
print(tt.shape, tt['future_spend_4w'].describe())
print("zero share:", (tt['future_spend_4w']==0).mean())
print("by snapshot day:")
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median', lambda x:(x==0).mean()]))

f = A.load_saved("churn_seasonality.parquet")
m = tt.merge(f, on=['household_key','snapshot_day'], how='left')
num = [c for c in f.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m['future_spend_4w']).sort_values(key=np.abs, ascending=False)
print(cor.head(40))