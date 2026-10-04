import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

# demographics
demo = agent_api.snapshot().demographics
print("demo rows:", len(demo), "cols:", list(demo.columns))
print(demo.head(3).to_string())

def ridge_cv(feat_list, alpha=1000, extra=None):
    D = df.copy()
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl = feat_list + [c for c in extra.columns if c!='household_key']
    else:
        fl = feat_list
    outs = []
    for vd in [431,403]:
        tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
        va = D[D.snapshot_day==vd]
        Xtr = pd.get_dummies(tr[fl].fillna('NA').astype(object), dummy_na=False)
        Xva = pd.get_dummies(va[fl].fillna('NA').astype(object)).reindex(columns=Xtr.columns, fill_value=0)
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
        w = np.linalg.solve((Xtr.values.T@Xtr.values)/len(tr) + alpha*np.eye(Xtr.shape[1]), Xtr.values.T@(tr.future_spend_4w.values - tr.future_spend_4w.mean())/len(tr))
        pv = np.clip(Xva.values@w + tr.future_spend_4w.mean(), 0, None)
        outs.append(np.abs(pv-va.future_spend_4w.values).mean())
    return outs

base = ridge_cv(feats)
print("internal ridge val431/403 (feats only):", [round(x,2) for x in base])
# add demographics one-hot
dd = demo.copy()
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
    dd[c] = dd[c].astype(str)
with_demo = ridge_cv(feats, extra=dd[['household_key']+ [c for c in dd.columns if c!='household_key']])
print("internal ridge + demographics:", [round(x,2) for x in with_demo])
