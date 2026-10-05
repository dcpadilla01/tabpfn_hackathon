
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
print(feats.dtypes[feats.dtypes=='object'].index.tolist())
print(feats.dtypes.tail(12))
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
print([ (c, str(m[c].dtype)) for c in m.columns if m[c].dtype not in (np.number,) ][:20])
