import agent_api, pandas as pd, numpy as np
b = agent_api.load_saved('e009_ewma_longlags.parquet')
print('shape', b.shape)
print(b['snapshot_day'].value_counts().sort_index())
cols = list(b.columns)
for pat in ['spend','tlag','ewma','trip','index']:
    print(pat, [c for c in cols if pat in c.lower()][:40])
print('dtypes', b.dtypes.value_counts().to_dict())
t = agent_api.train_targets()
print(t.shape, t.columns.tolist())
print(t.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))
