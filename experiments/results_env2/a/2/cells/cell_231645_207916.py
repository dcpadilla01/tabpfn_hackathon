
import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p4 = agent_api.load_saved('pred_e004.parquet')
p5 = agent_api.load_saved('pred_e005.parquet')
p6 = agent_api.load_saved('pred_e006.parquet')

tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
base = tr[['household_key','snapshot_day','future_spend_4w','spend_28','spend_56','spend_84']]
print("base:", base.shape)
m = base.merge(p6.rename(columns={'prediction':'pred6'}), on=['household_key','snapshot_day'], how='inner')
print("after p6:", m.shape)
m = m.merge(p4.rename(columns={'prediction':'pred4'}), on=['household_key','snapshot_day'], how='inner')
print("after p4:", m.shape)
m = m.merge(p5.rename(columns={'prediction':'pred5'}), on=['household_key','snapshot_day'], how='inner')
print("after p5:", m.shape)
print(m.head())
