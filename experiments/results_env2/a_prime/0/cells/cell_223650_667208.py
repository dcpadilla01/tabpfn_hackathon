import agent_api, numpy as np, pandas as pd
df = agent_api.load_saved('e008_spendproc.parquet')
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('E008 n feat', len(feat))
new = [c for c in feat if c not in set(agent_api.load_saved('e007_te.parquet').columns)]
print('E008 added vs E007:', new)
