
import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p6 = agent_api.load_saved('pred_e006.parquet')

tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("tr keys sample:", tr[['household_key','snapshot_day']].head())
print("p6 keys sample:", p6[['household_key','snapshot_day']].head())
print("tr snapshot days:", sorted(tr['snapshot_day'].unique()))
print("p6 snapshot days:", sorted(p6['snapshot_day'].unique()))
print("tr hh max:", tr['household_key'].max(), "p6 hh max:", p6['household_key'].max())
k1 = set(zip(tr['household_key'], tr['snapshot_day']))
k2 = set(zip(p6['household_key'], p6['snapshot_day']))
print("key overlap:", len(k1 & k2), "tr keys:", len(k1), "p6 keys:", len(k2))
