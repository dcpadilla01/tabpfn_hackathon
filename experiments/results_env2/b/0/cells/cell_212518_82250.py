import agent_api as A
tt = A.train_targets()
print(tt.shape, tt.head())
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))
