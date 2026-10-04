import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
demo = agent_api.snapshot().demographics
cats = [c for c in demo.columns if c!='household_key']

D = df.merge(demo[['household_key']+cats], on='household_key', how='left')
tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=431])]
va = D[D.snapshot_day==431]
s = cats
n = feats
Xs_tr = pd.get_dummies(tr[s].astype(str), dummy_na=False).astype(float)
print("dummy shape:", Xs_tr.shape, "cols sample:", list(Xs_tr.columns[:8]), flush=True)
print("sum of dummies:", Xs_tr.values.sum(), "rows:", len(tr), flush=True)
