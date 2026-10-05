import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda x:(x==0).mean()])
g.columns=['mean','median','zero_frac']
print(g.round(2))
# feature drift: spend_28 by snapshot day
f3 = agent_api.load_saved('feats_v3.parquet')
g2 = f3.groupby('snapshot_day')[['spend_28','spend_84','baskets_28']].mean().round(2)
print(g2)
# correlation of snapshot_day with target within household (drift per household)
m = f3[['household_key','snapshot_day','spend_28']].merge(tt,on=['household_key','snapshot_day'])
print('corr(spend_28, target)', m.spend_28.corr(m.future_spend_4w).round(3))
print('corr(snapshot_day, target)', m.snapshot_day.corr(m.future_spend_4w).round(3))
