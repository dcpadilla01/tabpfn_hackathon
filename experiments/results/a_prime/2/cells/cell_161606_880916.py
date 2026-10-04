import agent_api, pandas as pd, numpy as np
sd = agent_api.snapshot_days(); print(sd)
tt = agent_api.train_targets()
print('targets:', tt.shape); print(tt.head(3))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','std','count']).round(1))
e009 = agent_api.load_saved('e009_ewma_longlags.parquet')
print('e009:', e009.shape)
print(e009.groupby('snapshot_day').size())
v = agent_api.snapshot(as_of_day=207)
print('households type:', type(v.households))
hh207 = set(np.asarray(v.households).ravel().tolist())
fd = v.transactions.groupby('household_key')['day'].min()
pred = set(fd[fd <= 207-84].index)
print('hh207', len(hh207), 'pred', len(pred), 'only_view', len(hh207-pred), 'only_pred', len(pred-hh207))
sub = e009[e009.snapshot_day==207]
print('tlag NaN rates:', sub[['spend_28','tlag_2','tlag_5','tlag_13']].isna().mean().round(3).to_dict())
print(sub.groupby(sub['tlag_13'].isna())['tenure_days'].agg(['mean','count']))
demo = agent_api.snapshot().demographics
print('demo:', demo.shape); print(demo.head(3))
