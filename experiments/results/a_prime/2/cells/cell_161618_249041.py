import agent_api, pandas as pd, numpy as np
# Probe: are there households with no purchase in the 28d before snapshot? (churn/zero-window)
# and what does the target look like for them vs active ones
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
print('merged:', m.shape)
for c in ['spend_28','spend_56','spend_84','days_since_last','tenure_days']:
    print(c, 'NaN:', m[c].isna().mean().round(4))
# zero-window rate by snapshot day
m['zero28'] = (m['spend_28']==0).astype(int)
print(m.groupby('snapshot_day')['zero28'].mean().round(3))
# target stats by zero28
print(m.groupby('zero28')['future_spend_4w'].agg(['mean','std','count']).round(1))
# corr of spend_28 with target
print('corr spend_28 vs target:', m[['spend_28','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_84 vs target:', m[['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_364 vs target:', m[['spend_364','future_spend_4w']].corr().iloc[0,1].round(3))
# distribution of target
print(m['future_spend_4w'].describe().round(1))
print('target quantiles:', m['future_spend_4w'].quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).round(0).to_dict())
print('zero target share:', (m['future_spend_4w']==0).mean().round(3))
