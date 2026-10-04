import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
demo = agent_api.snapshot().demographics

def design(D, s_cols, n_cols):
    Xn = D[n_cols].fillna(0).astype(float).values if n_cols else np.zeros((len(D),0))
    Xs = pd.get_dummies(D[s_cols].astype(str), dummy_na=False).astype(float).values if s_cols else np.zeros((len(D),0))
    return np.hstack([Xn, Xs])

def ridge_one(vd, feat_list, extra=None, alpha=1000):
    D = df
    fl = list(feat_list)
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl += [c for c in extra.columns if c!='household_key']
    tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = D[D.snapshot_day==vd]
    s = [c for c in fl if str(D[c].dtype)=='category' or D[c].dtype==object]
    n = [c for c in fl if c not in s]
    Xtr, Xva = design(tr, s, n), design(va, s, n)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr)/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym))/len(tr))
    pv = np.clip(Xva@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

cats = [c for c in demo.columns if c!='household_key']
print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)
print("+demo:", [round(ridge_one(vd, feats, extra=demo[['household_key']+cats]),2) for vd in [431,403]], flush=True)
