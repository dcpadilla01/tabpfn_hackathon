
import agent_api as A, numpy as np, pandas as pd
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); cand = A.load_saved('cand_train.parquet')
print('feats dtypes non-numeric:')
for c in feats.columns:
    if not np.issubdtype(feats[c].dtype, np.number): print('  ', c, feats[c].dtype)
print('tt days', sorted(tt.snapshot_day.unique()))
print('cand days', sorted(cand.snapshot_day.unique()))
print('cand sizes:'); print(cand.snapshot_day.value_counts().sort_index())
print('tt sizes:'); print(tt.snapshot_day.value_counts().sort_index())
k1 = set(map(tuple, tt[['household_key','snapshot_day']].values))
k2 = set(map(tuple, cand[['household_key','snapshot_day']].values))
print('tt pairs', len(k1), 'cand pairs', len(k2), 'tt-not-in-cand', len(k1-k2), 'cand-not-in-tt', len(k2-k1))
print('dup cand pairs:', cand.duplicated(['household_key','snapshot_day']).sum())
