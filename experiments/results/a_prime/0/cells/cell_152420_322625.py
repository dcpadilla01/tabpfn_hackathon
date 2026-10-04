import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
tr = df[df.snapshot_day.isin([d for d in all_train_days if d!=431])]
va = df[df.snapshot_day==431]
print("shapes", tr.shape, va.shape, flush=True)
Xtr = pd.get_dummies(tr[feats].astype(object), dummy_na=False).astype(float)
print("dummies done", Xtr.shape, flush=True)
Xva = pd.get_dummies(va[feats].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
print("va dummies done", flush=True)
ym = tr.future_spend_4w.mean()
A = (Xtr.T@Xtr).values/len(tr) + 1000*np.eye(Xtr.shape[1])
print("A done", flush=True)
w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
print("w done", flush=True)
pv = np.clip(Xva.values@w + ym, 0, None)
print("MAE431:", np.abs(pv-va.future_spend_4w.values).mean(), flush=True)
