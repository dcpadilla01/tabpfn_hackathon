import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
def load(name):
    t = agent_api.load_saved(name)
    return t.merge(tt, on=['household_key','snapshot_day'], how='inner')

def ev(df, alpha=100.0):
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

base = load('e009_ewma_longlags.parquet')
b = ev(base); print('E009  proxy: train %.2f inner %.2f  (harness 62.292)'%b)
known = {'e011_pruned.parquet':('E011',62.399), 'e013_full.parquet':('E013',62.518),
         'e014_full_pool.parquet':('E014',62.766), 'e016_dec_predictors.parquet':('E016',62.680)}
for nm,(hid,hm) in known.items():
    r = ev(load(nm))
    print('%s proxy: train %.2f inner %.2f  d_inner %+.2f | harness d %+.3f'%(hid, r[0], r[1], r[1]-b[1], hm-62.292))