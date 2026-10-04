import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e013_denoise.parquet')
print('shape', t.shape)
print('cols:', list(t.columns))
tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe().round(2))
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
print(cor.reindex(cor.abs().sort_values(ascending=False).index).head(45).round(3))