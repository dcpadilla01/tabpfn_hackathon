import agent_api as A
import pandas as pd, numpy as np
tt = A.train_targets()
print(tt.dtypes)
print(tt.head())
m = A.load_saved('e006_temporal.parquet')
print(m.dtypes.head(3))
print(m[['household_key','snapshot_day']].head())
print('tt keys sample', tt['household_key'].head().tolist(), m['household_key'].head().tolist())
print('overlap', m.set_index(['household_key','snapshot_day']).index.isin(tt.set_index(['household_key','snapshot_day']).index).sum(), 'of', len(tt))