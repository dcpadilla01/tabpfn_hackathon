import pandas as pd, numpy as np
t = agent_api.load_saved('e001_history.parquet')
print(t.shape)
print(sorted(t.columns))
tt = agent_api.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']
print('target: mean %.1f  zeros %.1f%%' % (y.mean(), 100*(y==0).mean()))
num = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes(include=[np.number])
corr = num.corrwith(y)
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(35))
