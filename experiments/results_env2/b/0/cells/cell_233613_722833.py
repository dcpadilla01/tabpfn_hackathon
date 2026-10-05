import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w==0).mean())
# How predictable is future spend from lag1 (spend in [s-27,s])? quick sanity on train rows
base = agent_api.load_saved('e013_stationary.parquet')
print('base shape', base.shape)
m = tt.merge(base[['household_key','snapshot_day','spend_28','spend_84','z_rate84','dec_28']], on=['household_key','snapshot_day'])
print('merged', m.shape)
for c in ['spend_28','spend_84','z_rate84','dec_28']:
    print(c, 'corr', np.corrcoef(m[c].fillna(0), m.future_spend_4w)[0,1])
