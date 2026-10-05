
import agent_api, pandas as pd, numpy as np

t = agent_api.train_targets()
print(t.shape, t.columns.tolist())
print(t['future_spend_4w'].describe())
print("zero share:", (t['future_spend_4w']==0).mean())

e2 = agent_api.load_saved('e002_mix.parquet')
print(e2.shape)
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
num = [c for c in e2.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m['future_spend_4w']).sort_values()
print("Top positive corr:")
print(corr.tail(25))
print("Top negative corr:")
print(corr.head(10))
print("n features:", len(num))
