import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def ridge_cv(feat_list, extra=None, alpha=1000):
    D = df.copy()
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl = feat_list + [c for c in extra.columns if c!='household_key']
    else:
        fl = list(feat_list)
    outs = []
    for vd in [431,403]:
        tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
        va = D[D.snapshot_day==vd]
        Xtr = pd.get_dummies(tr[fl].astype(object), dummy_na=False).astype(float)
        Xva = pd.get_dummies(va[fl].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
        ym = tr.future_spend_4w.mean()
        A = (Xtr.T@Xtr).values/len(tr) + alpha*np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
        pv = np.clip(Xva.values@w + ym, 0, None)
        outs.append(np.abs(pv-va.future_spend_4w.values).mean())
    return outs

base = ridge_cv(feats)
print("feats-only:", [round(x,2) for x in base], flush=True)

demo = agent_api.snapshot().demographics
dd = demo.copy()
cat_cols = [c for c in dd.columns if c!='household_key']
with_demo = ridge_cv(feats, extra=dd[['household_key']+cat_cols])
print("+demographics:", [round(x,2) for x in with_demo], flush=True)
