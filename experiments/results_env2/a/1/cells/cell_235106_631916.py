import numpy as np, pandas as pd
f4 = agent_api.load_saved('e004_features.parquet')
f5 = agent_api.load_saved('e005_newfeats.parquet')
F = f4.merge(f5, on=['household_key','snapshot_day'], how='left')
tt = agent_api.train_targets()
F = F.merge(tt, on=['household_key','snapshot_day'], how='left')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('merged', F.shape, 'n feats', len(FEATS))
path = agent_api.save_table(F, 'allF.parquet')
print(path)
