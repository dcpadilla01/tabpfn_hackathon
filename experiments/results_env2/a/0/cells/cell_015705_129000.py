
import agent_api, pandas as pd, numpy as np

for f in ['feats_v1','feats_v2','feats_v3','feats_v4']:
    d = agent_api.load_saved(f + '.parquet')
    print(f, d.shape, 'snapdays:', sorted(d.snapshot_day.unique()))
    print('  cols:', list(d.columns))

tt = agent_api.train_targets()
print('tt days:', sorted(tt.snapshot_day.unique()), tt.shape)
