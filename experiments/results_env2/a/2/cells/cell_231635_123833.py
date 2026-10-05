
import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
p6 = agent_api.load_saved('pred_e006.parquet')
print(feats.dtypes.head(3))
print(t.dtypes)
print(p6.dtypes)
print(feats['household_key'].head(), t['household_key'].head(), p6['household_key'].head())
print("pred6 cols:", p6.columns.tolist())
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("train rows after merge:", tr.shape)
