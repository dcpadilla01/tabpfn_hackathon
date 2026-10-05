
import agent_api, pandas as pd, numpy as np
e13 = agent_api.load_saved('e013_stationary.parquet')
e15 = agent_api.load_saved('e015_norm_windows.parquet')
e17 = agent_api.load_saved('e017_reversion.parquet')
print('shapes', e13.shape, e15.shape, e17.shape)
print('cols e13:', sorted(e13.columns))
print('dropped by E015 (in e13 not e15):', sorted(set(e13.columns)-set(e15.columns)))
print('e17 adds:', sorted(set(e17.columns)-set(e13.columns)))
tt = agent_api.train_targets()
y = tt.future_spend_4w
print('n=%d zero=%.3f mean=%.1f med=%.1f p90=%.0f max=%.0f' % (len(tt),(y==0).mean(),y.mean(),y.median(),y.quantile(.9),y.max()))
print(tt.groupby('snapshot_day').future_spend_4w.agg(['size','mean','median']).round(1))
print('e13 rows per snapshot:'); print(e13.snapshot_day.value_counts().sort_index().to_dict())
m = e13.merge(tt, on=['household_key','snapshot_day'])
num = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes('number')
pc = num.corrwith(m.future_spend_4w)
print('pearson with target, bottom15:'); print(pc.sort_values().head(15).round(3).to_string())
print('top15:'); print(pc.sort_values().tail(15).round(3).to_string())
