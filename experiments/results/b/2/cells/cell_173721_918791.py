import agent_api, pandas as pd, numpy as np
tab = agent_api.load_saved('e018_hinge_prune.parquet')
print('shape:', tab.shape)
print(tab.dtypes.value_counts())
obj = [c for c in tab.columns if tab[c].dtype == object]
print('obj cols:', obj)
print('columns:', list(tab.columns))
tt = agent_api.train_targets()
print('targets:', tt.shape)
print(tt.head(3))
print(agent_api.snapshot_days())
print('rows per snapshot:')
print(tab.groupby('snapshot_day').size())