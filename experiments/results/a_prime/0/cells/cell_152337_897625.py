import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
print("start", df.shape, len(feats), flush=True)

def ridge_one(vd, feat_list, alpha=1000):
    tr = df[df.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = df[df.snapshot_day==vd]
    Xtr = pd.get_dummies(tr[feat_list].astype(object), dummy_na=False).astype(float)
    Xva = pd.get_dummies(va[feat_list].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr).values/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
    pv = np.clip(Xva.values@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)
