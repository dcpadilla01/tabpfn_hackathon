import numpy as np, pandas as pd
L = agent_api.load_saved('lagfeats.parquet')
print(L.dtypes)
print(L.groupby('snapshot_day')[['lag28_56','lag56_84','lag84_112','lag112_140']].mean().round(2))
v = L[L.snapshot_day==431]
print('431 std:', v[['lag28_56','lag56_84','lag84_112','lag112_140']].std().values)
print(v.head())
