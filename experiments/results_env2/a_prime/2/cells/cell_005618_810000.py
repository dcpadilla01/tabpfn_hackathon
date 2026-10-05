import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']

def prep(path):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number): X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float), t['future_spend_4w'].values.astype(float), t['snapshot_day'].values

paths = {'e001':'e001_recent_behavior.parquet','e010':'e010_lifecycle.parquet',
         'e011':'e011_discounts.parquet','e015':'e015_full_superset.parquet',
         'e016':'e016_churn_gapratio.parquet','e017':'e017_xsec_rank.parquet',
         'e009':'e009_ar_season.parquet','e012':'e012_hh_target_enc.parquet'}
harness = {'e001':63.574,'e010':62.702,'e011':62.651,'e015':62.647,
           'e016':62.535,'e017':62.526,'e009':66.683,'e012':72.176}
data = {k: prep(v) for k,v in paths.items()}

def pseudo_mae(X, y, d, lam, holdout=(403,431)):
    tr = ~np.isin(d, holdout); va = np.isin(d, holdout)
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    Zt = np.hstack([Z[tr], np.ones((tr.sum(),1))]); Zv = np.hstack([Z[va], np.ones((va.sum(),1))])
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1] -= lam  # no penalty on intercept
    w = np.linalg.solve(A, Zt.T@y[tr])
    return np.abs(Zv@w - y[va]).mean()

for lam in [100, 300, 1000, 3000, 10000]:
    row = sorted((pseudo_mae(*data[k], lam), k) for k in paths)
    print('lam %6d:' % lam, '  '.join('%s:%.1f' % (k,m) for m,k in row))
print('harness:  ', '  '.join('%s:%.1f' % (k,v) for k,v in sorted(harness.items(), key=lambda kv: kv[1])))
