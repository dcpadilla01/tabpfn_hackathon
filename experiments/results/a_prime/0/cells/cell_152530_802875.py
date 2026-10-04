import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
demo = agent_api.snapshot().demographics
print(demo.dtypes, flush=True)
dd = demo.copy()
cats = [c for c in dd.columns if c!='household_key']
D = df.merge(dd[['household_key']+cats], on='household_key', how='left')
print(D[cats[0]].dtype, D[cats[0]].unique()[:5], flush=True)
